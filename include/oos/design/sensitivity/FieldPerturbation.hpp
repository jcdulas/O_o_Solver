#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/model/FieldTargets.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace oos::design
{

// The central-difference step of a designed field v: h = FIELD_STEP_RELATIVE·|v|, or
// FIELD_STEP_AT_ZERO for a field at zero, the derivative extrapolated from the central differences at h and h/2
// (Richardson):
//     D(h) = (f(v + h) − f(v − h)) / 2h,   f'(v) ≈ (4·D(h/2) − D(h)) / 3,
// whose truncation is of order h⁴·f⁽⁵⁾ (none for a polynomial of degree 4 or less: a shell's stiffness is cubic in its
// thickness, a rod's linear in its area and modulus) and whose rounding is of order ε·|f|/h.
//
// Why 1e-2: the rounding is what matters. A pseudo-load −∂K/∂v·u sums terms of the size of K·u, which cancel down to
// the size of the loads: the rounding of ∂K/∂v, ε·|K|/h, is amplified by that cancellation (1e4 to 1e6 for a plate in
// bending, more for a stiff connector between two plates), and it shows in full on a derivative that is small against
// its response. Measured on the models of the gradient tests, against the
// finite differences of the whole analysis: d(von Mises)/dE of a ShellTria3 plate, 1.3e-5 off with a plain central
// difference at 1e-5, 1e-8 with the extrapolated one at 1e-3; d(tip displacement)/dD of a Weld lap joint, 7e-3 off
// at 1e-4, 3e-3 at 1e-3, 1e-5 at 1e-2 (the limit of the reference). With the extrapolation, the truncation at 1e-2 is
// of order 1e-8·f⁽⁵⁾/f' and below 1e-10 for the smooth rational dependences (a bar's shear flexibility in I); none for
// the polynomial ones.
//
// A field at zero (an offset ZOFFS, an angle THETA, an expansion coefficient) has no relative step: FIELD_STEP_AT_ZERO
// in the units of the field. The dependences on such fields are polynomial (an offset enters the stiffness
// quadratically, an expansion coefficient the thermal loads linearly), which the extrapolation differentiates exactly
// whatever the step, or trigonometric in degrees (an angle), smooth over a thousandth of a degree; so the step is taken
// large enough that the rounding stays small. A field not at zero always takes the relative step, however small it is
// (an inertia of 1e-7 m⁴ must not be moved by an absolute step, which could make it negative).
inline constexpr double FIELD_STEP_RELATIVE = 1.0e-2;
inline constexpr double FIELD_STEP_AT_ZERO = 1.0e-3;

[[nodiscard]] double fieldStep(double value) noexcept;

// A field whose own value is no measure of its size: an off-diagonal term of a AnisoMat2D (G12, G13, G23)
// is zero or a rounding of zero in a balanced laminate (A16 = A26 = 0: a lamination parameter at 2.8e-17 gave G13 at
// 8.8e-13 where G11 is 5.7e4), so that its relative step, 1e-2·|v|, moved the stiffness by 1.5e-19 of its matrix and
// the differences of the section loads were rounding (764 for 44 on the composite tiny barrel); and its step at zero,
// FIELD_STEP_AT_ZERO in its units, moved it by 2e-8 of its matrix. Its step is taken from the size of its matrix,
// `scale` (fieldStepScale: the largest diagonal term of the AnisoMat2D at the design point), where its value is below
// FIELD_STEP_SCALE_FRACTION of it: h = FIELD_STEP_RELATIVE·max(|v|, FIELD_STEP_SCALE_FRACTION·scale), at least
// 1e-4 of the matrix; a term above that fraction keeps its relative step and its bits (the dependence on a term is a
// rational function of the matrix, as smooth over 1e-4 of it as over 1e-2 of the term). `scale` 0: fieldStep(value).
inline constexpr double FIELD_STEP_SCALE_FRACTION = 1.0e-2;

[[nodiscard]] double fieldStep(double value, double scale) noexcept;

// The size of the matrix a field is a term of, for fieldStep(value, scale): the largest |G11|, |G22|, |G33| of the
// AnisoMat2D row of a "stiffness" cell of `target` in `point` (the design point's snapshot); 0 for any other field, or
// a AnisoMat2D whose diagonal is zero.
[[nodiscard]] double fieldStepScale(const input::ModelSnapshot& point, const FieldTarget& target);

// The sides of a perturbation: v + h, v − h, v + h/2, v − h/2.
enum class FieldSide : std::uint8_t
{
    Plus,
    Minus,
    HalfPlus,
    HalfMinus
};
inline constexpr std::size_t FIELD_SIDE_COUNT = 4;

// One field of a batch of perturbations (FieldPerturbation's fourth constructor): its target, its value at the design
// point, its relation and its step.
struct BatchedField
{
    const FieldTarget* target = nullptr;
    double value = 0.0;
    std::size_t relation = 0;
    double step = 0.0;
};

// The snapshots of a design point on which one designed quantity is moved to v ± h and v ± h/2, every other quantity at
// its value of the point, with their indices and element sets: what ElementDerivatives and the explicit parts of the
// responses difference. A designed quantity is a field (the value of a relation: its cells are moved) or a shape (the
// shape value s of a shaped variable: every grid point of its basis is moved by ±h·T(g) together, so that the
// differences are the derivatives along the basis vector, Σ_g ∂/∂g · T(g), in one perturbation whatever the number of
// grid points). The quantities are indexed as PseudoLoads indexes them: the fields (one per relation), then the shapes
// (one per shape basis). Each side is an overlay of the base snapshot (FieldTargets::overlayOf, as DesignPoint makes
// the snapshot of a point), so a perturbation costs the copy of the designed columns and four indices, never a copy of
// the model.
//
// Thread-safety: immutable once built. Ownership: owns the snapshots (which keep the base alive), their indices and
// element sets; borrows the registry and the budget, which must outlive it.
class FieldPerturbation
{
public:
    // `base`: the snapshot the targets were bound to; `values`: the field and shape values of the design point;
    // `quantity`: the quantity moved (a relation index, or the number of relations plus a shape basis index); `step`:
    // h, positive. Throws as ColumnOverlay::applyTo and SnapshotIndex; std::invalid_argument for a quantity out of
    // range or a step that is not positive.
    FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& base, const DesignModel& model,
                      const FieldTargets& targets, const DesignValues& values, std::size_t quantity, double step,
                      const elements::ElementRegistry& registry, base::MemoryBudget& budget);
    // The field of relation `relation` moved by fieldStep() of its value, for a design model without shape variables
    // (the plain field perturbations). std::invalid_argument when the targets move grid points: use the constructor
    // above, which keeps them at the design point.
    FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& base, const FieldTargets& targets,
                      const std::vector<FieldValue>& fields, std::size_t relation,
                      const elements::ElementRegistry& registry, base::MemoryBudget& budget);
    // The field of relation `relation`, of value `value` at a design point, moved by `step`, each side made from the
    // point's own snapshot `point` (the overlay of the base by FieldTargets::overlayOf with every field and shape of
    // the point: the snapshot the analysis read) with only the cells of that relation (`target`) overlaid again: the
    // same values cell by cell as the first constructor gives for a field, every other field and every moved position
    // being the point's already. The sides share the lookups of `pointIndex`, the index of `point` (a field changes
    // no ID nor position: SnapshotIndex's second constructor).
    // A field of a designed laminate or library section moves its card's derived cells too (the ply stack,
    // the section properties: FieldTargets::addDerivedCells), from `targets` and the point's field values
    // `pointFields` (one per relation), the relation's own at each side; a section's from `sections`, its properties
    // linearised at the point (FieldTargets::linearSections: one call of the library for every section), or
    // from the library with the point's dimensions as reference when null.
    // PERFORMANCE: the first constructor overlays every designed cell and moved position of the design on the base
    // and builds four indices of the whole model per quantity: O(variables + grid points) per variable, so that the
    // full Jacobian of V variables cost O(V²) (100 000 overlay cells and an index of the 170 000 grid points four
    // times per variable at 1 M DOFs); this one copies the relation's columns and shares the index.
    // Measured: 200 thickness variables on 250 000 DOFs, the perturbations 0.78 s → 0.35 s, the same bits (2.7 ms per
    // variable at 1 M DOFs).
    FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& point,
                      const elements::SnapshotIndex& pointIndex, const FieldTarget& target, double value,
                      std::size_t relation, double step, const elements::ElementRegistry& registry,
                      const FieldTargets* targets = nullptr, std::span<const double> pointFields = {},
                      const SectionLinearisation* sections = nullptr);

    // The fields of several relations moved together, each by its own step and to the values the third constructor
    // gives it, on four sides they share: for relations whose cells and reached elements are disjoint, an element of
    // one of them reads on these sides exactly what it reads on its relation's own (the other relations' cells are
    // read by none of its computations), so the derivatives are the same bits. member(k) is the perturbation of the
    // k-th field (its relation and the distances between its sides), sharing the sides. Not for the fields of a
    // designed laminate or library section, whose derived cells belong to their card.
    // PERFORMANCE: a perturbation builds four snapshots of the point (each table imported again and assembled: some
    // 90 us whatever the model's size); a variable per element moves thousands of disjoint fields, one batch for all.
    // Measured: a 2 162-bar truss, its fields 0.92 s to 0.16 s.
    FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& point,
                      const elements::SnapshotIndex& pointIndex, std::span<const BatchedField> fields,
                      const elements::ElementRegistry& registry);

    // The perturbation of the k-th field of a batch (the fourth constructor), sharing its sides.
    [[nodiscard]] FieldPerturbation member(std::size_t k) const;

    // The quantity moved (a relation index for a field).
    [[nodiscard]] std::size_t relation() const noexcept { return m_quantity; }
    [[nodiscard]] std::size_t quantity() const noexcept { return m_quantity; }
    [[nodiscard]] const assembly::ElementSet& set(FieldSide side) const noexcept { return *sideOf(side).set; }
    [[nodiscard]] const std::shared_ptr<const input::ModelSnapshot>& snapshot(FieldSide side) const noexcept
    {
        return sideOf(side).snapshot;
    }

    // The derivative of a quantity from its values on the four sides (Plus, Minus, HalfPlus, HalfMinus), by the
    // extrapolation above, with the distances between the sides as they are stored (v ± h rounded).
    [[nodiscard]] double derivative(double plus, double minus, double halfPlus, double halfMinus) const noexcept;
    // The same, element by element of four arrays of the same size, into `derivative`.
    void derivatives(std::span<const double> plus, std::span<const double> minus, std::span<const double> halfPlus,
                     std::span<const double> halfMinus, std::span<double> derivative) const noexcept;

private:
    // Shared by the members of a batch (the element set borrows the index, which borrows the snapshot: the members
    // keep all three).
    struct Side
    {
        std::shared_ptr<const input::ModelSnapshot> snapshot;
        std::shared_ptr<const elements::SnapshotIndex> index;
        std::shared_ptr<const assembly::ElementSet> set;
    };

    // A member of a batch: its quantity and widths on the batch's sides.
    FieldPerturbation(std::size_t quantity, double width, double halfWidth,
                      const std::array<Side, FIELD_SIDE_COUNT>& sides);

    [[nodiscard]] const Side& sideOf(FieldSide side) const noexcept { return m_sides[static_cast<std::size_t>(side)]; }
    [[nodiscard]] static Side makeSide(const std::shared_ptr<const input::ModelSnapshot>& base,
                                       const input::ColumnOverlay& overlay, const elements::ElementRegistry& registry,
                                       base::MemoryBudget& budget);

    std::size_t m_quantity;
    double m_width = 0.0;     // (v + h) − (v − h) as stored
    double m_halfWidth = 0.0; // (v + h/2) − (v − h/2) as stored
    std::array<Side, FIELD_SIDE_COUNT> m_sides;
    // The members of a batch: their relations and widths (empty for one quantity).
    struct Member
    {
        std::size_t relation = 0;
        double width = 0.0;
        double halfWidth = 0.0;
    };
    std::vector<Member> m_members;
};

} // namespace oos::design
