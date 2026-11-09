#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// What every subcase of a static analysis shares, on which a condensation that serves them all is made (the plan,
// CondensationPlanner, and the condensed solution, CondensedStatics, read it the same way):
// - how many subcases constrain each DOF (SPC, SinglePointConstrList, SinglePointConstrUnion, EnforcedMotionConstr and
// GRID PS): a DOF held in every subcase is removed
//   from the units, one held in some only puts its point on a boundary;
// - the MPC cards every subcase selects (an MultiPointConstrUnion resolved to the cards it holds, from the source_sid
// column of
//   multiPointConstrTable, MpcEquations::sourceSetIdsOf), whose equations a unit may eliminate, and those that vary
//   between subcases, which stay in the connection system;
// - the contact table every subcase selects (glued contact): one for every subcase, as one MPC set; subcases selecting
//   different ones are reported ("glued-contact-varies", an error: the condensed path takes one).
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
    // grid points alone). Reports the problems of the SPC sets and of the contact tables to `sink`.
    [[nodiscard]] static CommonConstraints of(const input::ModelSnapshot& snapshot, const assembly::DofMap& dofMap,
                                              std::span<const StaticSubcase> subcases, base::MemoryBudget& budget,
                                              base::DiagnosticSink& sink);
};

} // namespace oos::solutions
