#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/ConstraintTable.hpp"
#include "oos/input/tables/LoadTable.hpp"
#include "oos/input/tables/SubcaseTable.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::assembly
{

// An enforced value of a constrained DOF (SPC field D, EnforcedMotionConstr): the displacement (or rotation, in
// radians) it is held at, in the displacement system CD of its grid point.
struct EnforcedDisplacement
{
    base::DofIndex dof = 0;
    double value = 0.0;
};

// The single-point constraints of one subcase, from the singlePointConstrTable of O_o_Reader
// (SPCADD expanded: the rows of an SinglePointConstrUnion set carry its ID in `setId`) and the permanent constraints of
// the grid points:
// - sg: the PS field of every GRID (read from the node table; singlePointConstrTable repeats them as rows "GRID.PS"
// with a null setId,
//   which are skipped here);
// - sb: the components of the SPC and SinglePointConstrList rows of the SPC set the subcase selects, and of the
// EnforcedMotionConstr rows of its LOAD set
//   (singlePointConstrTable gives EnforcedMotionConstr under the ID of the load set; the DOFs of an
//   EnforcedMotionConstr become members of the s set, and in statics its SID is selected by LOAD); the
//   EnforcedMotionConstr of a LOAD combination are those of the sets it reaches times their factors
//   (input::LoadTable::termsOf: O_o_Reader lists each set's EnforcedMotionConstr once);
// - the enforced values: field D of SPC (0 when blank), and the value of EnforcedMotionConstr, which replaces the SPC
// value of the same
//   DOF. Only nonzero values are kept: every other s DOF is held
//   at 0.
//
// A scalar point is constrained with component 0 or blank. Problems are Diagnostics: an unknown point
// ("spc-unknown-point"), a component the point does not have ("spc-bad-component"), two different SPC values on one
// DOF ("spc-conflicting-values", the first is kept).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its arrays (budgeted); the budget must outlive
// it. Copyable: its arrays shared (withLoadSet).
class SpcSet
{
public:
    // The constraints of the SPC set `spcSetId` and of the EnforcedMotionConstr rows of the load set `loadSetId`
    // (either may be absent) on top of the permanent constraints. `constraints` may be null for a model without
    // constraint tables, `loads` for a model without load tables (the EnforcedMotionConstr of `loadSetId` are then its
    // own rows); with them, the EnforcedMotionConstr are those of the terms of `loadSetId` (LoadTable::termsOf), each
    // value times its term's factor.
    SpcSet(const input::ConstraintTable* constraints, const input::LoadTable* loads, const DofMap& dofMap,
           std::optional<base::EntityId> spcSetId, std::optional<base::EntityId> loadSetId, base::MemoryBudget& budget,
           base::DiagnosticSink& sink);

    // The constraints of a subcase: the sets its SPC and LOAD commands select (subcasesTable, inheritance applied by
    // O_o_Reader). Throws std::invalid_argument when the subcase is not in the table.
    [[nodiscard]] static SpcSet forSubcase(const input::ConstraintTable* constraints, const input::LoadTable* loads,
                                           const input::SubcaseTable& subcases, const DofMap& dofMap,
                                           base::EntityId subcase, base::MemoryBudget& budget,
                                           base::DiagnosticSink& sink);

    // The IDs of the sets that have EnforcedMotionConstr rows in `constraints` (null: none), increasing, each once.
    [[nodiscard]] static std::vector<base::EntityId> enforcedSetIds(const input::ConstraintTable* constraints);

    // The terms of the load set `loadSetId` whose sets have EnforcedMotionConstr rows (`enforcedSets`: enforcedSetIds),
    // in the order of LoadTable::termsOf: two subcases with the same SPC set and the same such terms have the same
    // SpcSet but for its load set (the owner's 6 000 cases share one, each made once instead of a pass over the g set
    // per case; at the section's 1/32 with 6 000 cases, 26 s).
    [[nodiscard]] static std::vector<input::LoadTerm> enforcedTermsOf(const input::LoadTable* loads,
                                                                      std::span<const base::EntityId> enforcedSets,
                                                                      std::optional<base::EntityId> loadSetId);

    // This set's constraints and values under another load set, whose EnforcedMotionConstr terms are this one's
    // (enforcedTermsOf): its arrays shared, not copied (6 000 cases one copy).
    [[nodiscard]] SpcSet withLoadSet(std::optional<base::EntityId> loadSetId) const;

    [[nodiscard]] std::optional<base::EntityId> spcSetId() const noexcept { return m_spcSetId; }
    [[nodiscard]] std::optional<base::EntityId> loadSetId() const noexcept { return m_loadSetId; }

    // The DOFs of each set, increasing, each once. A DOF may be in both (DofSets keeps it in sg).
    [[nodiscard]] std::span<const base::DofIndex> sbDofs() const noexcept { return m_arrays->sb.span(); }
    [[nodiscard]] std::span<const base::DofIndex> sgDofs() const noexcept { return m_arrays->sg.span(); }
    // The nonzero enforced values, by increasing DOF.
    [[nodiscard]] std::span<const EnforcedDisplacement> enforcedDisplacements() const noexcept
    {
        return m_arrays->enforced.span();
    }

    // Whether two subcases constrain the same DOFs (the enforced values may differ: they are loads). Subcases with the
    // same constraints share one factorisation (SOL 101).
    [[nodiscard]] bool hasSameConstrainedDofs(const SpcSet& other) const noexcept;

private:
    SpcSet() = default;

    std::optional<base::EntityId> m_spcSetId;
    std::optional<base::EntityId> m_loadSetId;
    // The arrays, shared by the sets made by withLoadSet (immutable).
    struct Arrays
    {
        base::BudgetedBuffer<base::DofIndex> sb;
        base::BudgetedBuffer<base::DofIndex> sg;
        base::BudgetedBuffer<EnforcedDisplacement> enforced;
    };
    std::shared_ptr<const Arrays> m_arrays;
};

} // namespace oos::assembly
