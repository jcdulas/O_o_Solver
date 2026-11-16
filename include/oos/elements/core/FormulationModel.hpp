#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/EntityLayout.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace oos::elements
{

class ElementRegistry;
class SnapshotIndex;
struct NodalDisplacements;

// One term of a relation of the formulation: a DOF of an element (grid, scalar or entity) and its coefficient.
struct FormulationRelationTerm
{
    ElementDof dof;
    double coefficient = 0.0;
};

// A component of a grid point held by a support of the constraint set, in the displacement system CD of the grid
// point (component 1 to 6), with its enforced value (0 when none).
struct HeldGridComponent
{
    std::int32_t gridRow = 0;
    int component = 1;
    double value = 0.0;
};

// A support of an entity DOF that the formulation derives from the supports of the grid points (an edge whose two
// grid points hold a translation holds the displacement moments of the edge along it, ...), with its value and the
// map that carries its reaction back to the grid points: the reaction r of the derived DOF adds r·c_j to each grid
// component of `carriedTo` (the transpose of the map from the grid values to the entity value, exact for rigid
// motions, so that the reported support forces balance the loads).
struct DerivedSupport
{
    ElementDof dof;
    double value = 0.0;
    std::vector<FormulationRelationTerm> carriedTo;
};

// What an element formulation whose unknowns are not all at grid points tells the rest of the solver (the mixed
// hybrid formulation: docs/mixed-hybrid-formulation-spec.md sections 5.1 to 5.7). The assembly, the constraints and
// the solutions see the formulation through this interface only; the element families read their own model.
//
// Thread-safety: immutable once built; concurrent calls are safe. Ownership: built once per snapshot index
// (SnapshotIndex::computedOnce), it lives as long as the index.
class FormulationModel
{
public:
    virtual ~FormulationModel() = default;

    // The entities of an analysis of the given kind (section 5.2).
    [[nodiscard]] virtual const EntityLayout& layout(EntityLayout::Kind kind) const = 0;

    // The problems of the model as a whole (missing or unknown geometry declarations, cards of two formulations on one
    // edge...); the problems of one element are reported by its family's check().
    virtual void reportProblems(base::DiagnosticSink& sink) const = 0;

    // The ties of the formulation for a constraint set (section 3.4): dependent DOF = Σ terms. `held` are the grid
    // components the set holds (a tie is not written on a held unknown).
    virtual void ties(EntityLayout::Kind kind, std::span<const HeldGridComponent> held,
                      std::vector<ElementDof>& dependents, std::vector<std::size_t>& offsets,
                      std::vector<FormulationRelationTerm>& terms, base::DiagnosticSink& sink) const = 0;

    // The grid components the formulation determines from its own unknowns: the rotations of the grid points of its
    // shells (the mean rotation of the shells at their corners) and the translations of the mid-side nodes of its
    // shells (the point of their edge), section 5.4. Each is either made dependent by an observer relation (when
    // something else uses it), held (when a support holds it: read along the edges), or excluded from the free set and
    // filled after the solution with determinedRow().
    virtual void determinedGridComponents(EntityLayout::Kind kind,
                                          std::vector<HeldGridComponent>& components) const = 0;
    // The row of a determined grid component: its value = Σ terms (in its CD system), for an observer relation.
    // Precondition: the component is one of determinedGridComponents().
    virtual void determinedRow(EntityLayout::Kind kind, std::int32_t gridRow, int component,
                               std::vector<FormulationRelationTerm>& terms) const = 0;
    // The values of every determined grid component from a solution (the excluded ones are filled with them after the
    // solution): write(gridRow, component, value), in the CD systems. Computed element by element, without the rows
    // (a corner rotation reads every unknown of its elements: too many terms to keep for a large model).
    virtual void fillDeterminedComponents(EntityLayout::Kind kind, const NodalDisplacements& displacements,
                                          const std::function<void(std::int32_t, int, double)>& write) const = 0;

    // The supports of entity DOFs derived from the grid components `held` (section 5.6).
    virtual void deriveSupports(EntityLayout::Kind kind, std::span<const HeldGridComponent> held,
                                std::vector<DerivedSupport>& supports, base::DiagnosticSink& sink) const = 0;

    // The values on the entity DOFs (layout(kind) order, EntityLayout::totalDofCount() values) of the affine
    // displacement u(x) = b + A·x in the basic system, the rotation unknowns taken from the skew part of A (section
    // 5.7). The grid components are the caller's.
    virtual void affineField(EntityLayout::Kind kind, const base::Mat3& gradient, const base::Vec3& translation,
                             std::span<double> entityValues) const = 0;

    // The stresses at the grid points of the solids of the formulation for the displacements of a case (section
    // 10.3): at each grid point, the mean of the corner stresses of the solids that have it among those `takes`
    // accepts (card, element ID), in the basic system (Voigt order σx, σy, σz, τxy, τyz, τzx). None by default.
    struct GridPointStress
    {
        std::int32_t gridRow = 0;
        std::array<double, 6> stress{};
    };
    // The element of the formulation that has the face of the grid points `gridRows` (node table rows, 3 or 4, in the
    // order of a pressure given by grid points: PointsPressureLoad, whose force is P times the vector area of that
    // order), for that pressure to act through its pressure facet (section 7): its card, its batch row, and the sign
    // that turns P into the pressure of the element (a shell's along its normal, a solid face's inward). Empty when no
    // element of the formulation has that face (the pressure then acts at the grid points).
    struct FormulationFace
    {
        std::string_view card;
        std::int64_t batchRow = 0;
        double sign = 1.0;
    };
    [[nodiscard]] virtual std::optional<FormulationFace> faceOf(std::span<const std::int32_t> gridRows) const
    {
        static_cast<void>(gridRows);
        return std::nullopt;
    }

    using SolidSelection = std::function<bool(std::string_view card, std::int32_t elementId)>;
    virtual void gridPointStresses(EntityLayout::Kind kind, const NodalDisplacements& displacements,
                                   const SolidSelection& takes, std::vector<GridPointStress>& stresses) const
    {
        static_cast<void>(kind);
        static_cast<void>(displacements);
        static_cast<void>(takes);
        static_cast<void>(stresses);
    }
};

// The formulation model of the snapshot of `index` in the formulation `registry` chose, or null when the formulation
// has no unknowns outside the grid points (every formulation but MixedHybridFormulation). Built on the first call.
[[nodiscard]] const FormulationModel* formulationModelOf(const SnapshotIndex& index, const ElementRegistry& registry);

} // namespace oos::elements
