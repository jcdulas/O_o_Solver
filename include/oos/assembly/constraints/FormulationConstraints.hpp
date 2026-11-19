#pragma once

#include "oos/assembly/constraints/FormulationRelations.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/FormulationModel.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <vector>

namespace oos::assembly
{

// The grid components an SPC set holds (sb and sg), with their enforced values, for the formulation.
[[nodiscard]] std::vector<elements::HeldGridComponent> heldGridComponentsOf(const SpcSet& spc, const DofMap& dofMap);

// `spc` with the supports the formulation derives from it (FormulationModel::deriveSupports: the entity DOFs of the
// edges and faces whose grid points hold a direction, docs/mixed-hybrid-formulation-spec.md section 5.6); `spc`
// itself when `model` is null.
[[nodiscard]] SpcSet withFormulationSupports(const SpcSet& spc, const DofMap& dofMap,
                                             const elements::FormulationModel* model, elements::EntityLayout::Kind kind,
                                             base::MemoryBudget& budget, base::DiagnosticSink& sink);

class GlueConstraints;

// The g-DOFs that something other than the element formulation uses, for the observer relations of section 5.4: the
// grid components a concentrated load of any set acts on, those an element of another formulation stiffens (the
// stiffened components of the node-block connectivity: the formulation's own shells and solids stiffen no rotation),
// and those a relation of the model names as an independent DOF. One flag per g-DOF.
struct UsedElsewhere
{
    std::vector<std::uint8_t> flags;
    // The components a concentrated load acts on (a subset of `flags`; empty: none).
    std::vector<std::uint8_t> loaded;
    // Marks the components a concentrated load row of the model acts on: the translations of the grid point of a
    // force card, its rotations for a moment card.
    void markConcentratedLoads(const input::ModelSnapshot& snapshot, const DofMap& dofMap);
    // Marks the components of the grid points whose bit is set in `stiffenedComponents` (one byte per node, bit c for
    // component c + 1).
    void markStiffened(std::span<const std::uint8_t> stiffenedComponents, const DofMap& dofMap);
    // Marks the independent DOFs of the relations of `gm` (their own terms, before the chains are resolved).
    void markIndependentTerms(const RigidElimination& gm);
};

// What the element formulation adds to the constraints of one constraint set (sections 3.4, 5.4 and 5.6): its ties,
// the observer relations of the grid components it determines that something else uses, and the excluded set x of the
// determined components nothing else uses, with the rows that fill them after the solution.
//
// The grid components a relation of the model makes dependent (a rigid element, a multipoint constraint) are read
// along the edges and faces as those a support holds are: an edge whose two grid points have a component held or
// dependent follows them, its entity DOFs made dependent on those grid components by the map that carries the values
// of a support along the edge (the relations "that follow", after the ties). The grid points a glued contact ties
// are left out: each follows its own point of a face, and the side between two of them keeps its own unknowns.
// Carried along their chords, the sides of a glued sheet stay straight on a face that bends: a cantilever of two
// glued layers was 1.4 times too stiff, 2.5 times when the two layers bend side by side
// (wasm/test/validation/glue.test.mjs, section 5: 1.5 % and 0.01 % off the closed forms with the sides left free).
//
// A determined component that a support holds or a relation makes dependent, and that no edge reads (the rotation at
// one grid point of shells alone: the clamp that holds a free body, a rigid element joined to the shells at one grid
// point), is tied: the mean rotation of the shells there is made equal to it by a relation written on one unknown of
// the shells (the "tied rows", last). A held rotation is tied about the axes of the plane of the shells only, a
// dependent one in each of its components. Where no unknown of the shells is free to take the row, a dependent
// rotation stays the relation's own and a warning says so (rotation-not-transmitted).
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: owns its rows (bounded by the ties and
// the determined components: std::vector is the right container).
class FormulationConstraints
{
public:
    // `spc`: the set with its derived supports (withFormulationSupports). `userRelations`: the eliminations of the
    // model's own relations (rigid elements, MPCs, fasteners, glue), to know which DOFs they make dependent.
    // `glue`: the glued contact among them (nullptr: none). The grid points it ties each follow their own point of
    // a face: the side between two of them is not carried along its chord as between two grid points of a rigid
    // element (it would stay straight on a face that bends), its own unknowns stay free.
    FormulationConstraints(const elements::FormulationModel& model, elements::EntityLayout::Kind kind,
                           const DofMap& dofMap, const SpcSet& spc, const UsedElsewhere& usedElsewhere,
                           const RigidElimination& userRelations, base::DiagnosticSink& sink,
                           const GlueConstraints* glue);

    // The ties and the observer relations, for RigidElimination.
    [[nodiscard]] const FormulationRelations& relations() const noexcept { return m_relations; }
    // The excluded set x, increasing.
    [[nodiscard]] std::span<const base::DofIndex> excludedDofs() const noexcept { return m_excluded; }
    // Fills the excluded DOFs of a g-set displacement vector from its other values (after RigidElimination::
    // expandToG: the formulation reads its unknowns, which may be dependent): FormulationModel::
    // fillDeterminedComponents.
    void fillExcluded(std::span<double> displacementsG) const;
    // The transpose of fillExcluded on a g-set vector of weights (an adjoint's right-hand side, a response's
    // gradient): the weight of each excluded DOF moved onto the DOFs its value is filled from
    // (FormulationModel::determinedRow), the excluded DOF's set to zero; so that wᵀ·u of a filled u is the moved
    // weights times its other values.
    void moveExcludedWeights(std::span<double> weightsG) const;

    // The rows of this set's ties and observer relations on another formulation model of the same layout (the model of
    // a shape perturbation of the design sensitivities: its grid points moved), on the same dependent DOFs in the same
    // order: the ties asked of `model` with this set's held components, the observer relations from its
    // determinedRow. Their differences over the perturbations give the derivatives of the relations with respect to
    // the positions. Throws std::logic_error when `model` ties other DOFs than this set (another layout).
    [[nodiscard]] FormulationRelations relationsOn(const elements::FormulationModel& model,
                                                   base::DiagnosticSink& sink) const;
    // fillExcluded with the rows of another formulation model of the same layout (relationsOn).
    void fillExcludedOn(const elements::FormulationModel& model, std::span<double> displacementsG) const;

private:
    // A tied row: Σ_s direction[s]·(mean value of the component firstComponent + s of the shells at the grid point −
    // that grid component) = 0.
    struct TiedRow
    {
        std::int32_t gridRow = 0;
        int firstComponent = 4; // 4: the rotations of a corner; 1: the translations of a mid-side grid point
        std::array<double, 3> direction{};
    };
    using SparseRow = std::map<base::DofIndex, double>;
    // A constraint Σ_j a_j·u_j = 0 eliminated as a relation on one of its unknowns: its coefficients by g-DOF, and
    // whether it reaches grid components the model's relations move (its unknown must then be read by no relation).
    struct ConstraintRow
    {
        SparseRow coefficients;
        bool chains = false;
    };
    // The unknown a row is written on, from its coefficients and its index (−1: none, the row is left out).
    using PivotChoice = std::function<base::DofIndex(const ConstraintRow&, std::size_t index)>;
    // The rows to eliminate on `model`, in a fixed order: what the supports hold along directions that are no
    // component of an edge or a face (FormulationModel::supportRows), then the tied rows (m_tiedRows).
    [[nodiscard]] std::vector<ConstraintRow> constraintRowsOn(const elements::FormulationModel& model) const;
    // Appends the relations of `rows` to `relations`, each on the unknown `pivots` gives it; with `choosePivot` the
    // unknowns are chosen and stored in `pivots`.
    void eliminateRows(std::vector<ConstraintRow>& rows, std::vector<base::DofIndex>& pivots,
                       FormulationRelations& relations, const PivotChoice* choosePivot) const;
    // The row of a grid DOF a relation of the model makes dependent (empty for any other DOF), and whether it is one.
    [[nodiscard]] std::span<const ConstraintTerm> userRowOf(base::DofIndex dof) const noexcept;
    [[nodiscard]] bool isUserDependent(base::DofIndex dof) const noexcept;
    // Replaces the DOFs of `coefficients` that the model's relations make dependent by those relations' rows.
    void substituteUserRows(SparseRow& coefficients) const;
    // Appends u_dependent = Σ row to `relations`, its terms on DOFs the model's relations make dependent replaced by
    // those relations' rows: no relation of the formulation then chains through a relation of the model, which the
    // condensed path may eliminate elsewhere (inside a unit, when the formulation's is at the interface).
    void addRelation(FormulationRelations& relations, const elements::ElementDof& dependent,
                     std::span<const FormulationTerm> row) const;

    const elements::FormulationModel* m_model;
    elements::EntityLayout::Kind m_kind;
    const DofMap* m_dofMap;
    FormulationRelations m_relations;
    std::vector<base::DofIndex> m_excluded;
    // The grid components the set holds, and those with the ones the model's relations make dependent (the ties and
    // the edges that follow are asked with them); the relations in order: m_tieCount ties, the edges and faces that
    // follow a relation of the model (their dependent DOFs in m_followedDofs), the observer relations, then from
    // m_firstTiedRow the tied rows (m_tiedRows): relationsOn.
    std::vector<elements::HeldGridComponent> m_held;
    std::vector<elements::HeldGridComponent> m_followed;
    std::size_t m_tieCount = 0;
    std::vector<base::DofIndex> m_followedDofs;
    std::size_t m_firstTiedRow = 0;
    std::vector<TiedRow> m_tiedRows;
    // The rows eliminated from m_firstTiedRow on (constraintRowsOn): the first m_supportRowCount are the supports',
    // the others the tied rows; the unknown of each (−1: left out).
    std::size_t m_supportRowCount = 0;
    std::vector<base::DofIndex> m_rowPivots;
    // The rows of the grid DOFs the model's relations make dependent, on their independent DOFs (RigidElimination::
    // row of the model's relations alone): m_userDofs increasing, row i's terms in m_userTerms from m_userOffsets[i].
    // A shape of the design sensitivities never changes them (one that deforms a rigid element is refused).
    std::vector<base::DofIndex> m_userDofs;
    std::vector<std::size_t> m_userOffsets;
    std::vector<ConstraintTerm> m_userTerms;
};

} // namespace oos::assembly
