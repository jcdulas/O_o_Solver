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
    // The component is not held by a support: a relation of the model makes it dependent (a rigid element), and the
    // edges read it as they read a held one (assembly::FormulationConstraints).
    bool followsRelation = false;
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

    // The unit normal that the shells of the formulation share at the grid point of node table row `gridRow`, in its CD
    // system (section 5.6): a rotation about it is the rotation of the membrane field in its plane, which a support
    // of a rotation of that grid point does not hold (as the drilling rotation of a shell without drilling stiffness).
    // Empty when the grid point is no corner of shells or when its shells meet at an angle. None by default.
    [[nodiscard]] virtual std::optional<base::Vec3> commonShellNormal(std::int32_t gridRow) const
    {
        static_cast<void>(gridRow);
        return std::nullopt;
    }

    // The edges of the shells of the formulation, each as the node table rows of its two grid points, appended to
    // `edges`. None by default.
    virtual void shellEdges(std::vector<std::array<std::int32_t, 2>>& edges) const { static_cast<void>(edges); }

    // The supports of entity DOFs derived from the grid components `held` (section 5.6).
    virtual void deriveSupports(EntityLayout::Kind kind, std::span<const HeldGridComponent> held,
                                std::vector<DerivedSupport>& supports, base::DiagnosticSink& sink) const = 0;

    // What the grid components `held` hold along a direction that is no component of the unknowns of an edge or a
    // face (a translation held in a displacement system turned from the basic one, the same direction at every grid
    // point of the edge or face: a plane of symmetry that is not along the basic axes), section 5.6: each row is a
    // constraint Σ terms = 0 on entity DOFs and on the held grid components, appended to `rows`. They are no support
    // of one DOF: the constraints eliminate them as relations (assembly::FormulationConstraints). None by default.
    virtual void supportRows(EntityLayout::Kind kind, std::span<const HeldGridComponent> held,
                             std::vector<std::vector<FormulationRelationTerm>>& rows) const
    {
        static_cast<void>(kind);
        static_cast<void>(held);
        static_cast<void>(rows);
    }

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
    // The IDs of the elements whose matrices act on the entity of row `row` of layout(kind), increasing: the element of
    // an interior or of a line end, the elements that have every grid point of an edge or a face. Empty by default.
    [[nodiscard]] virtual std::vector<std::int32_t> entityElements(EntityLayout::Kind kind, std::int32_t row) const
    {
        static_cast<void>(kind);
        static_cast<void>(row);
        return {};
    }

    // The grid points (node table rows) of the entity of row `row` of layout(kind): the two of an edge, those of a
    // face, the corners of the element of an interior, the ends of the line of a line end. The forces on the entity's
    // DOFs are reported at them (assembly::EntityForceCarry). Empty by default.
    [[nodiscard]] virtual std::vector<std::int32_t> entityGridRows(EntityLayout::Kind kind, std::int32_t row) const
    {
        static_cast<void>(kind);
        static_cast<void>(row);
        return {};
    }

    // The grid points (node table rows, increasing) whose positions the matrices of the element of card `card` at
    // batch row `batchRow` read beyond its own: those of the elements around it whose geometry the formulation reads
    // (describeSurroundings' two rings). Empty by default, and for an element the formulation does not have.
    [[nodiscard]] virtual std::vector<std::int32_t> readGridRows(std::string_view card, std::int64_t batchRow) const
    {
        static_cast<void>(card);
        static_cast<void>(batchRow);
        return {};
    }

    // What the matrices of the elements at the grid points `gridRows` (node table rows) read around them beyond their
    // own cards, grid points and relations: the geometry the formulation reads from the neighbouring elements, given to
    // `mix` as numbers in a fixed order. The condensation cache mixes them into the key of a unit
    // (reduction::unitInputStamp), so that an edit beside a unit, which the unit's own stamps do not see, condenses it
    // again. Nothing by default: the elements of ClassicFormulation read their own cards and grid points only.
    virtual void describeSurroundings(std::span<const std::int32_t> gridRows,
                                      const std::function<void(double)>& mix) const
    {
        static_cast<void>(gridRows);
        static_cast<void>(mix);
    }

    [[nodiscard]] virtual std::optional<FormulationFace> faceOf(std::span<const std::int32_t> gridRows) const
    {
        static_cast<void>(gridRows);
        return std::nullopt;
    }

    // The motion of a point of an element of the formulation as rows on its DOFs (section 5.10): the translations
    // (rows 0 to 2) and the rotations (rows 3 to 5) of the point in the basic system; the grid components in their CD
    // systems, the entity DOFs as they are.
    struct PointMotion
    {
        std::array<std::vector<FormulationRelationTerm>, 6> rows;
        // The point of the surface of the element the rows are those of (basic), when the formulation gives it
        // (facePointMotion of a shell: what is carried from it follows its rotation).
        std::optional<base::Vec3> point;
    };
    // One point of a shell that a fastener or a weld reads: the shell (its ID), the weight of the point, and what
    // places the point on the shell, as the fastener piercing table gives it: grid points of its card (`gridRows`,
    // node table rows) and their shape functions at the point.
    struct ShellSample
    {
        std::int32_t elementId = 0;
        double weight = 0.0;
        std::span<const std::int32_t> gridRows;
        std::span<const double> shapeValues;
    };
    // The motion of the point `point` (basic) of the shell of ID `elementId` that a fastener or a weld pierces: the
    // weighted mean over `samples` (the footprint of the fastener on the shells,
    // input::FastenerPiercingTable::meanFootprint, or the piercing point alone) of the motion of the shell at each
    // sample carried rigidly to `point`, u + θ × (point − X) and θ, X the point of the surface of the sample. A
    // sample on a shell the formulation does not have is left out.
    // Empty when the formulation does not have the shell `elementId` or no sample is on a shell of its own: the grid
    // points of the element then carry the point (elements/fasteners/PiercingInterpolation). None by default.
    [[nodiscard]] virtual std::optional<PointMotion> shellPointMotion(std::int32_t elementId, const base::Vec3& point,
                                                                      std::span<const ShellSample> samples) const
    {
        static_cast<void>(elementId);
        static_cast<void>(point);
        static_cast<void>(samples);
        return std::nullopt;
    }
    // The motion of a point of a face of an element of the formulation (a shell, or a face of a solid) given by the
    // grid points of the face (node table rows: its corners, then its mid-side points, −1 for an absent one) and
    // their shape functions at the point: the primary face of a glued contact. A shell gives its translations (rows
    // 0 to 2), its rotation (rows 3 to 5) and the point of its surface there; a solid its translations alone (the
    // rotations empty, no point). Empty when no element of the formulation has that face. None by default.
    [[nodiscard]] virtual std::optional<PointMotion> facePointMotion(std::span<const std::int32_t> faceGridRows,
                                                                     std::span<const double> shapeValues) const
    {
        static_cast<void>(faceGridRows);
        static_cast<void>(shapeValues);
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

// The entities that formulation numbers for an analysis of `kind` (FormulationModel::layout), or null without one:
// what the element set and the DOF map of an analysis take (assembly::ElementSet, assembly::DofMap).
[[nodiscard]] const EntityLayout* formulationEntitiesOf(const SnapshotIndex& index, const ElementRegistry& registry,
                                                        EntityLayout::Kind kind);

} // namespace oos::elements
