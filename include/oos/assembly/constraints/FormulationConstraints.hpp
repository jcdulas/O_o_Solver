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

#include <cstdint>
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

// What the element formulation adds to the constraints of one constraint set (sections 3.4 and 5.4): its ties, the
// observer relations of the grid components it determines that something else uses, and the excluded set x of the
// determined components nothing else uses, with the rows that fill them after the solution. A determined component a
// support holds is left to the support (the formulation reads it along the edges); one a relation of the model makes
// dependent is the relation's own, and a warning says the shells are not tied to it (rotation-not-transmitted).
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: owns its rows (bounded by the ties and
// the determined components: std::vector is the right container).
class FormulationConstraints
{
public:
    // `spc`: the set with its derived supports (withFormulationSupports). `userRelations`: the eliminations of the
    // model's own relations (rigid elements, MPCs, fasteners, glue), to know which DOFs they make dependent.
    FormulationConstraints(const elements::FormulationModel& model, elements::EntityLayout::Kind kind,
                           const DofMap& dofMap, const SpcSet& spc, const UsedElsewhere& usedElsewhere,
                           const RigidElimination& userRelations, base::DiagnosticSink& sink);

    // The ties and the observer relations, for RigidElimination.
    [[nodiscard]] const FormulationRelations& relations() const noexcept { return m_relations; }
    // The excluded set x, increasing.
    [[nodiscard]] std::span<const base::DofIndex> excludedDofs() const noexcept { return m_excluded; }
    // Fills the excluded DOFs of a g-set displacement vector from its other values (after RigidElimination::
    // expandToG: the formulation reads its unknowns, which may be dependent): FormulationModel::
    // fillDeterminedComponents.
    void fillExcluded(std::span<double> displacementsG) const;

private:
    const elements::FormulationModel* m_model;
    elements::EntityLayout::Kind m_kind;
    const DofMap* m_dofMap;
    FormulationRelations m_relations;
    std::vector<base::DofIndex> m_excluded;
};

} // namespace oos::assembly
