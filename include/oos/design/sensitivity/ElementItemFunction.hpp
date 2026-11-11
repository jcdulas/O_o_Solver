#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/stress_strain_force/ItemCodes.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

// One item of an element response: its type, item code, location in the result tables and lamina (0 when none).
struct ElementItemSpec
{
    ResponseType type = ResponseType::Stress;
    std::int32_t code = 0;
    ItemLocation location;
    std::int32_t lamina = 0;
};

// How an element item depends on the displacements, where its column says so:
// - Affine: a component of a shell's stresses or strains (σx, σy, τxy; a ply's σ1, σ2, τ12, τ1z, τ2z; the strains
//   likewise; the fibre distance) or of a rod's (axial, torsion): linear in u, affine under a thermal strain;
// - PlaneVonMises: a shell's or a ply's von Mises stress or strain, the function of its row's in-plane components it is
//   (ShellRecovery: the von Mises of the principal values of (a, b, k·c)): vm = scale·√(a² + b² − a·b + 3·(k·c)²), a, b
//   the normal and c the shear component (σx, σy, τxy; or the strains εx, εy, γxy with k = ½ and scale = ⅔, the
//   equivalent strain), each affine in u;
// - Other: any other item (a principal value, an angle, a margin, a strain energy, a failure index, another family).
struct ItemForm
{
    enum class Kind : std::uint8_t
    {
        Other,
        Affine,
        PlaneVonMises
    };
    Kind kind = Kind::Other;
    std::array<double, 3> components{}; // a, b, c of a PlaneVonMises item
    double shearFactor = 1.0;           // k
    double scale = 1.0;
};

// A writable copy of the model's displacements (six per grid point row, then the scalar points), for the recoveries at
// unit displacements of ElementItemFunction::slopes.
struct ScratchDisplacements
{
    std::span<double> grid;
    std::span<double> scalar;
};

// The values of an element's items and their derivatives with respect to the DOFs of the element, from the slopes of
// its recoveries (ElementItemFunction::slopes).
struct ItemSlopes
{
    std::vector<double> values;             // every item, as ElementItemFunction::values gives them
    std::vector<elements::ElementDof> dofs; // the DOFs of the element's stiffness
    std::vector<double> weights;            // items × DOFs: ∂item/∂u_d
    std::int64_t familySlopes = 0;          // the outputs whose slopes the family gave (one call each)
    std::int64_t unitRecoveries = 0;        // the recoveries at unit displacements of the outputs it did not give
};

// The second derivatives of a weighted sum Σ_k y_k·r_k of an element's items (ElementItemFunction::secondOrder),
// over the element's DOFs and the designed fields that reach it (row-major):
// the displacement block ∂²/∂u² (empty when every weighted item is affine in u: zero), the mixed block ∂²/∂u∂v (DOFs ×
// fields) and the field block ∂²/∂v² (fields × fields).
struct ItemSecondOrder
{
    std::vector<elements::ElementDof> dofs; // the DOFs of the element's stiffness
    std::vector<double> displacementBlock;  // dofs × dofs, or empty
    std::vector<double> mixedBlock;         // dofs × fields
    std::vector<double> fieldBlock;         // fields × fields
};

// What ElementItemFunction::secondOrder forms of an element that does not depend on the displacements: per output, the
// slopes of its recovery and, with a field, the slopes of their rates. Kept by the caller between the
// calls for one element and one designed field (its subcases), so that they are formed once; the same values either
// way (each a function of the element and the field alone).
struct ItemSlopeCache
{
    struct Output
    {
        elements::RecoverySlopes slopes;
        elements::RecoverySlopes slopeRates;
        bool hasSlopeRates = false;
    };
    std::map<std::string, Output, std::less<>> byKind; // by the output's table kind
};

// The items of an element response on one element as a function of the displacements and of the snapshot the element
// reads (a stress depends on the designed fields directly, as on the displacements), so that the
// sensitivities can difference them with respect to either: the same computation as the analysis (the family's
// recover() and recoverLayers(), the row of the place and layer ResultLookup reads, the strength ratio of
// DesignResp1Response, the strain energy ½·u_eᵀ·K_e·u_e of StiffnessRecovery), on any view of the element.
//
// ESE item 3 (the percentage of the total) is given as the element's strain energy (item 2): the total depends on
// every element, so the caller forms the percentage from it.
//
// Thread-safety: used on one thread (it caches the volume of ESE item 4). Ownership: owns its item list; borrows
// the family.
class ElementItemFunction
{
public:
    // The weighted mixed field derivative at fixed displacement. Nonlinear columns are chained through their
    // linear components; an unavailable field or column returns no value, so the caller keeps its numerical path.
    [[nodiscard]] std::optional<double> mixedFieldRate(const elements::ElementBatchView& view,
                                                       const elements::NodalDisplacements& displacements,
                                                       std::span<const double> weights,
                                                       const elements::DesignedField& first,
                                                       const elements::DesignedField& second) const;
    ElementItemFunction(const elements::ElementFamily& family, std::int64_t row, std::vector<ElementItemSpec> items)
        : m_family(&family), m_row(row), m_items(std::move(items)), m_columns(columnsOf(family, m_items))
    {
    }

    [[nodiscard]] const std::vector<ElementItemSpec>& items() const noexcept { return m_items; }
    [[nodiscard]] std::int64_t row() const noexcept { return m_row; }

    // The value of every item, in order, on `view` (of `snapshot`, which owns it: the volume of ESE item 4 and the
    // laminate of CSTRAT are read there) under `displacements`. Throws OptimisationError when a row the item reads is
    // not recovered (the analysis would have failed on it too).
    // With `forms`, also the form of each item (ItemForm, one per item: the components of a plane von Mises).
    [[nodiscard]] std::vector<double> values(const elements::ElementBatchView& view,
                                             const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                             const elements::NodalDisplacements& displacements,
                                             base::MemoryBudget& budget, std::vector<ItemForm>* forms = nullptr) const;

    // The values of the items and their derivatives with respect to the element's DOFs without differences,
    // or nothing when an item is not given by them (a column the family does not declare Linear or PlaneVonMises, a
    // strength ratio): the caller then differences the job. A Linear item's weights are the slopes of its row and
    // column (exact); a PlaneVonMises item's are chained from those of its components at their recovered values; a
    // strain energy E = ½·u_eᵀ·K_e·u_e has ∂E/∂u_e = K_e·u_e (the density E/V and the percentage alike: the caller
    // forms the percentage). The slopes come from the family (ElementFamily::recoverySlopes, its formulation built
    // once), else from recoveries at unit displacements without temperatures: S(·, d) = rows(e_d) − rows(0), exact for
    // the Linear columns, on a writable copy of the model's displacements that `scratch` gives (asked for only then:
    // the element's DOFs are moved in it and restored).
    [[nodiscard]] std::optional<ItemSlopes> slopes(const elements::ElementBatchView& view,
                                                   const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                                   const elements::NodalDisplacements& displacements,
                                                   const std::function<ScratchDisplacements()>& scratch,
                                                   base::MemoryBudget& budget) const;

    // The explicit parts of the items with respect to the designed field `field` at fixed displacements, in closed form
    // (ElementFamily::recoveryFieldRates; a ply's, ElementFamily::layerRecoveryFieldRates), or
    // nothing when an item is not given by them (its family has no closed form for the field, a column it gives no rate
    // of, a strength ratio or a strain energy): the caller then differences the job's recoveries on the perturbed
    // snapshots.
    [[nodiscard]] std::optional<std::vector<double>> fieldRates(const elements::ElementBatchView& view,
                                                                const elements::NodalDisplacements& displacements,
                                                                const elements::DesignedField& field) const;
    // fieldRates for several fields, the family asked once per table for all of them
    // (ElementFamily::recoveryFieldRatesOfFields): result[k] is fieldRates(view, displacements,
    // fields[k]), bit for bit. Safe to call concurrently (it reads the function only).
    [[nodiscard]] std::vector<std::optional<std::vector<double>>> fieldRatesOfFields(
        const elements::ElementBatchView& view, const elements::NodalDisplacements& displacements,
        std::span<const elements::DesignedField> fields) const;

    // The second derivatives of Σ_k weights[k]·r_k (one weight per item; a zero leaves the item out) at `displacements`
    // and the fields of `view`, in closed form, or nothing when a weighted item or a field is not
    // given by them: an item that is not a Linear or PlaneVonMises column of a stress, strain or force (a ply, a
    // strength ratio, a strain energy, another form), more than one field (their mixed derivatives are not given), or a
    // family without the closed forms of the field (ElementFamily::recoveryFieldRates, recoveryFieldSecondRates; the
    // slopes of the rates from recoverySlopeRates, else from the rates at unit displacements on `scratch`, exact). For
    // an item r linear in u, r = S·u + r₀: ∂²r/∂u² = 0, ∂²r/∂u∂v = ∂S/∂v, ∂²r/∂v² its row's second rate; a plane von
    // Mises chained from its components (planeVonMisesSecondRate).
    // `cache` (may be null): the slopes of this element and field formed by an earlier call (its other subcases), used
    // and completed.
    [[nodiscard]] std::optional<ItemSecondOrder> secondOrder(const elements::ElementBatchView& view,
                                                             const elements::NodalDisplacements& displacements,
                                                             std::span<const double> weights,
                                                             std::span<const elements::DesignedField> fields,
                                                             const std::function<ScratchDisplacements()>& scratch,
                                                             ItemSlopeCache* cache = nullptr) const;

    // The magnitude of the products of K_e·u_e, the largest Σ_c |K_rc·u_c| over the rows of the element's stiffness:
    // the scale of the rounding of its recovered forces (ValueCheck).
    [[nodiscard]] double forceProducts(const elements::ElementBatchView& view,
                                       const elements::NodalDisplacements& displacements) const;

    // The DOFs of the element (those of its stiffness matrix): the displacements its items read.
    [[nodiscard]] std::vector<elements::ElementDof> dofs(const elements::ElementBatchView& view) const;

private:
    // The output of a table (the composite outputs apart), recovered once for every item that reads it.
    using OutputKey = std::pair<bool, std::string>;
    using Outputs = std::map<OutputKey, elements::ElementResultBlock>;
    [[nodiscard]] const elements::ElementResultBlock& recoveredOutput(const elements::ElementBatchView& view,
                                                                      const ElementItemSpec& item,
                                                                      const elements::NodalDisplacements& displacements,
                                                                      Outputs& outputs) const;
    // The position of each item's column in its output, found by its name in the family's columns (the largest
    // std::size_t when the output does not have it, or for a strain energy).
    [[nodiscard]] static std::vector<std::size_t> columnsOf(const elements::ElementFamily& family,
                                                            const std::vector<ElementItemSpec>& items);
    // The value of a non-energy item at `column` of its output (values()), or the strength ratio of a CSTRAT item;
    // `forms` gives the form of the item when not null.
    [[nodiscard]] double itemValue(const elements::ElementBatchView& view, const ElementItemSpec& item,
                                   std::size_t column, const elements::NodalDisplacements& displacements,
                                   Outputs& outputs, ItemForm* form) const;
    [[nodiscard]] double volumeOf(const elements::ElementBatchView& view,
                                  const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                  base::MemoryBudget& budget) const;

    const elements::ElementFamily* m_family;
    std::int64_t m_row;
    std::vector<ElementItemSpec> m_items;
    std::vector<std::size_t> m_columns; // columnsOf(m_items)
    // The volume of the element on the last snapshot it was asked on (ESE item 4): it does not depend on the
    // displacements, which the differences move many times. A weak reference: a snapshot that died never matches
    // another one made at its address.
    mutable std::weak_ptr<const input::ModelSnapshot> m_volumeSnapshot;
    mutable double m_volume = 0.0;
};

} // namespace oos::design
