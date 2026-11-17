#pragma once

#include "oos/assembly/constraints/FormulationRelations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// What every subcase of a static analysis shares, on which a condensation that serves them all is made (the plan,
// CondensationPlanner, and the condensed solution, CondensedStatics, read it the same way):
// - how many subcases constrain each DOF (SinglePointConstr, SinglePointConstrList, SinglePointConstrUnion,
//   EnforcedMotionConstr and GRID PS): a DOF held in every subcase is removed from the units, one held in some only
//   puts its point on a boundary;
// - the MPC cards every subcase selects (a MultiPointConstrUnion resolved to the cards it holds, from the source_sid
//   column of multiPointConstrTable, MpcEquations::sourceSetIdsOf), whose equations a unit may eliminate, and those
//   that vary between subcases, which stay in the connection system;
// - the contact table every subcase selects (glued contact): one for every subcase, as one MPC set; subcases selecting
//   different ones are reported ("glued-contact-varies", an error: the condensed path takes one);
// - the element formulation's constraints (docs/mixed-hybrid-formulation-spec.md section 5.11), from each subcase's
//   assembly::FormulationConstraints as the full path builds them: its derived supports and its excluded DOFs (the
//   x set) counted with the supports in constrainedCount (an excluded DOF of every subcase is removed from the units
//   as a held one, without a reaction to report); its relations every subcase has with the same terms (eliminated
//   inside the units); and the nodes of the relations and excluded DOFs some subcases only have, which go on the
//   boundaries (the interface eliminates them per constraint set).
//
// Thread-safety: a value. Ownership: owns its lists (bounded by the subcases and their cards) and the counts
// (budgeted, one per g-set DOF).
struct CommonConstraints
{
    std::vector<base::EntityId> commonMpcCards; // sorted
    std::optional<base::EntityId> anyMpcSet;    // a set some subcase selects (its equations of the common cards)
    std::vector<base::EntityId> varyingMpcSets; // sorted
    std::optional<base::EntityId> commonContact;
    base::BudgetedBuffer<std::int32_t> constrainedCount; // per g-set DOF: the subcases that hold it
    std::int32_t subcaseCount = 1;
    // The element formulation's relations every subcase has (null without a formulation that has relations).
    std::shared_ptr<const assembly::FormulationRelations> formulationRelations;
    // The DOFs every subcase excludes (the formulation's x set), increasing.
    std::vector<base::DofIndex> formulationExcluded;
    // The nodes of the formulation's relations (dependent and terms) and excluded DOFs some subcases only have, sorted.
    std::vector<std::int32_t> formulationVaryingNodes;

    [[nodiscard]] bool isHeldEverywhere(base::DofIndex dof) const noexcept
    {
        return constrainedCount[static_cast<std::size_t>(dof)] == subcaseCount;
    }
    // The MPC set to read the equations of the common cards from (none when no card is common).
    [[nodiscard]] std::optional<base::EntityId> commonMpcSet() const noexcept
    {
        return commonMpcCards.empty() ? std::optional<base::EntityId>{} : anyMpcSet;
    }

    // From the solved subcases (no SUBCOM: they constrain nothing of their own; none: the permanent constraints of the
    // grid points alone). Reports the problems of the SPC sets and of the contact tables to `sink` (those of the
    // formulation's constraints are the constraint sets' own, reported by the analysis). `stiffenedComponents`: the
    // connectivity's (assembly::ElementConnectivity::stiffenedComponents), which the formulation's observer relations
    // read; empty without a formulation.
    [[nodiscard]] static CommonConstraints of(const input::ModelSnapshot& snapshot, const assembly::DofMap& dofMap,
                                              std::span<const StaticSubcase> subcases, base::MemoryBudget& budget,
                                              base::DiagnosticSink& sink,
                                              std::span<const std::uint8_t> stiffenedComponents = {});
};

} // namespace oos::solutions
