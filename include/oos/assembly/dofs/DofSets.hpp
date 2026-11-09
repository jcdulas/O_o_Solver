#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace oos::assembly
{

// The mutually exclusive sets a g-set DOF belongs to (the exclusive sets of the structural DOFs that SOL 101, 103 and
// 105 use): m (dependent of an MPC or a rigid element), sb (single-point constraint of the subcase: SPC,
// SinglePointConstrList, SinglePointConstrUnion, EnforcedMotionConstr), sg (permanent constraint GRID PS), autoSpc
// (constrained by the grid point singularity processor; reported in sb), o (omitted), r (SUPORT reference) and b (the
// rest: the a-set DOFs that no card names belong to the exclusive set b, written "B" in the USET and singularity
// tables).
enum class ExclusiveSet : std::uint8_t
{
    M,
    Sb,
    Sg,
    AutoSpc,
    O,
    R,
    B
};

// The sets of the format, exclusive and combined: g = n + m, n = f + s, s = sb + sg (+ the AUTOSPC DOFs, reported in
// sb), f = a + o, a = l + r (with no q set here), l = b here (no c set yet).
enum class DofSet : std::uint8_t
{
    G,
    M,
    N,
    S,
    Sb,
    Sg,
    F,
    O,
    A,
    R,
    L
};

// The name of a set, upper case as the text report prints it ("G", "SB"...).
[[nodiscard]] std::string_view nameOf(DofSet set) noexcept;
// The name the text report prints for an exclusive set ("M", "SB", "SG", "O", "R", "B"; an AUTOSPC DOF prints "SB").
[[nodiscard]] std::string_view nameOf(ExclusiveSet set) noexcept;
// Whether a DOF of an exclusive set is a member of a set.
[[nodiscard]] bool isMemberOf(ExclusiveSet exclusive, DofSet set) noexcept;

// The DOFs that the cards of a subcase put in each exclusive set; every span may hold a DOF more than once, in any
// order.
struct DofSetInputs
{
    std::span<const base::DofIndex> m;  // RigidElimination::mDofs()
    std::span<const base::DofIndex> sb; // SpcSet::sbDofs()
    std::span<const base::DofIndex> sg; // SpcSet::sgDofs()
    std::span<const base::DofIndex> o;  // OmittedDofSet, OmittedDofSet1 (refused: no o-set reduction before step 5)
    std::span<const base::DofIndex> r;  // FreeBodySupport, SUPORT1
};

// The DOF sets of one subcase: the exclusive set of every g-set DOF, and the counts and
// members of the combined sets under their names. The g set is reduced to the f set: the m DOFs are
// eliminated through Gm (RigidElimination), the s DOFs are removed (with their enforced values, SpcSet), and the a set
// is the f set (no o set).
//
// The rules applied when the inputs overlap (in a valid deck the exclusive sets do not overlap):
// - m with sb, sg or r: an error Diagnostic "dof-in-exclusive-sets" (such a deck is invalid); the DOF stays
//   in m;
// - sb with sg: accepted, the DOF is in sg (a subcase SPC on a permanently constrained component changes nothing);
// - r with sb or sg: an error, the DOF stays in s;
// - any o DOF: an error "unsupported-omitted-dofs" (the o-set reduction is not implemented;
// AnalysisDofSet/OmittedDofSet decks are refused).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its exclusive set array (budgeted); the budget
// must outlive it. Movable, not copyable (withAutoSpc() makes the sets after singularity processing).
class DofSets
{
public:
    DofSets(const DofMap& dofMap, const DofSetInputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    // The sets once the grid point singularity processor has constrained `autoSpcDofs`. Throws std::invalid_argument
    // when one of them is not in the a set (b) of these sets.
    [[nodiscard]] DofSets withAutoSpc(std::span<const base::DofIndex> autoSpcDofs) const;
    // The same sets with `supports` as the r set in place of these sets' (theirs back in b): the sets a free body is
    // factorised on when it is held by other supports than its FreeBodySupport DOFs (assembly::
    // chooseConditionedSupports). Throws std::invalid_argument when a support is not in b or r of these sets.
    [[nodiscard]] DofSets withSupports(std::span<const base::DofIndex> supports) const;

    [[nodiscard]] base::DofIndex gSetSize() const noexcept { return static_cast<base::DofIndex>(m_exclusive.size()); }
    [[nodiscard]] ExclusiveSet exclusiveSetOf(base::DofIndex dof) const noexcept
    {
        return static_cast<ExclusiveSet>(m_exclusive[static_cast<std::size_t>(dof)]);
    }
    [[nodiscard]] bool contains(DofSet set, base::DofIndex dof) const noexcept
    {
        return isMemberOf(exclusiveSetOf(dof), set);
    }
    [[nodiscard]] bool isAutoSpc(base::DofIndex dof) const noexcept
    {
        return exclusiveSetOf(dof) == ExclusiveSet::AutoSpc;
    }
    [[nodiscard]] std::int64_t count(DofSet set) const noexcept;
    [[nodiscard]] std::int64_t count(ExclusiveSet set) const noexcept
    {
        return m_counts[static_cast<std::size_t>(set)];
    }

    // The DOFs of a set, increasing.
    [[nodiscard]] base::BudgetedBuffer<base::DofIndex> members(DofSet set, base::MemoryBudget& budget) const;
    // One flag per g-set DOF, nonzero for the DOFs outside `kept`: the isEliminated argument of linalg::DofSelection
    // (kept = DofSet::F for a static solve on the f set).
    [[nodiscard]] base::BudgetedBuffer<std::uint8_t> eliminationFlags(DofSet kept, base::MemoryBudget& budget) const;

private:
    DofSets(base::BudgetedBuffer<std::uint8_t> exclusive, base::MemoryBudget& budget);
    void countMembers() noexcept;

    base::MemoryBudget* m_budget;
    base::BudgetedBuffer<std::uint8_t> m_exclusive; // one ExclusiveSet per g-set DOF
    std::array<std::int64_t, 7> m_counts{};         // per ExclusiveSet
};

} // namespace oos::assembly
