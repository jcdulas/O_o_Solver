#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::linalg
{

// A subtree of the supernode tree factorised by one task, serially: in a postorder a subtree is the range of
// consecutive supernodes [first, last], last being its root.
struct FactorSubtree
{
    std::int32_t first = 0;
    std::int32_t last = 0;
    // In WorkspacePool's class sizes of the blocks.
    std::int64_t frontValues = 0;        // the largest front of the subtree, (c + r)²: one reused buffer (indefinite)
    std::int64_t stackValues = 0;        // the peak of its stack of update matrices (r² each)
    std::int64_t contributionValues = 0; // the update matrix of its root, handed to the parent (r²)
    // What the subtree holds at its peak, its root's update matrix included: in indefinite mode its front buffer and
    // its stack at their peaks; in positive definite mode the largest stack a supernode's update matrix joins while
    // its children's are still on it (no front buffer).
    std::int64_t heldValues = 0;
    double flops = 0.0; // its factorisation's (supernodeFlops)
};

// The plan of a multifrontal factorisation (internal to linalg/supernodal): which
// supernodes run in which task, the children of each supernode, and the bound of the working storage, all from the
// symbolic factor before any numeric work.
//
// Tasks: the tree is cut into independent subtrees (each factorised serially by one task, with the kernels on one
// thread) and the top part above them (one task per supernode, with the kernels' threads). The cut depends on the
// tree and its flops only, never on the number of threads: a subtree of more than 1/SUBTREE_SPLIT_COUNT of the total
// work (and at least SUBTREE_MIN_FLOPS) is split into its children, its root moving to the top part. Each supernode is
// computed the same way whatever task runs it, so the cut never changes a bit of the result.
//
// Working storage (the fronts and the update matrices waiting for their parent; the panels of the factor are counted
// apart): at most threadCount subtrees run at once (the driver enforces it: a subtree that starts beyond them reserves
// its own front and stack in the budget, MultifrontalDriver), each holding at most its front, its stack and its root's
// update matrix, and a finished subtree holds its root's update matrix; then the top part runs in waves (TaskGraph),
// all the supernodes of a wave holding their front and update matrix at once, with every update matrix not yet
// consumed. The bound is the largest of these states: an upper bound of any interleaving the pool may produce. Without
// delayed pivots it covers every allocation of the driver; delayed pivots grow fronts beyond it, and the driver
// reserves the growth when it happens. Each front and update matrix is counted at the size class of its block
// (WorkspacePool::classSize), so the bound covers the blocks as allocated, and the blocks kept for reuse within it.
//
// A positive definite front needs no front buffer (assembled in place), and its bound counts what that mode allocates:
// in a subtree, the update matrix each supernode builds while its children's are still on the
// stack; at the top, each supernode's update matrix beside the ones alive. The indefinite bound counts the front
// buffers besides; it covers the positive definite mode too.
//
// With TopOrder::Serial (the streamed factorisation): the top part runs one supernode after the other in
// increasing order on the calling thread (StreamedFactorisation::runTop), not in waves, so its bound follows that
// order: each supernode's update matrix beside the ones alive, those of its children released after it.
//
// Thread-safety: immutable once built. Ownership: owns its arrays.
class FactorSchedule
{
public:
    static constexpr std::int32_t NO_SLOT = -1;
    static constexpr double SUBTREE_SPLIT_COUNT = 64.0;
    // About a millisecond of work: below it a task costs more to schedule than it computes.
    static constexpr double SUBTREE_MIN_FLOPS = 2.0e6;

    // How the top part runs: in waves of independent supernodes (MultifrontalDriver), or one after the other in
    // increasing order (StreamedFactorisation).
    enum class TopOrder : std::uint8_t
    {
        Waves,
        Serial
    };

    // With excludesTrailing, the trailing supernode is not factorised: the driver assembles it last, as the Schur
    // complement of a partial factorisation, outside the working storage.
    FactorSchedule(const SymbolicFactor& symbolic, bool excludesTrailing, std::size_t threadCount,
                   PivotingMode pivoting, base::MemoryBudget& budget, TopOrder topOrder = TopOrder::Waves);

    [[nodiscard]] std::span<const std::int32_t> children(std::int32_t supernode) const noexcept;
    [[nodiscard]] const std::vector<FactorSubtree>& subtrees() const noexcept { return m_subtrees; }
    // The subtrees' indices, the most work first (ties: the first): the order their tasks are given to the pool, so
    // that the largest do not start last and leave the other threads waiting (longest processing time first, Graham
    // 1969). The order decides only when a subtree runs, never what it computes.
    [[nodiscard]] std::vector<std::size_t> subtreesByWork() const;
    // The supernodes of the top part, children before parents (increasing index), the excluded trailing one last.
    [[nodiscard]] const std::vector<std::int32_t>& topSupernodes() const noexcept { return m_topSupernodes; }
    // The slot of the update matrix of a subtree root or a top supernode (handed between tasks), or NO_SLOT.
    [[nodiscard]] std::int32_t slotOf(std::int32_t supernode) const noexcept
    {
        return m_slot[static_cast<std::size_t>(supernode)];
    }
    [[nodiscard]] std::int32_t slotCount() const noexcept { return m_slotCount; }
    // The trailing supernode left unfactorised, or -1.
    [[nodiscard]] std::int32_t excludedTrailing() const noexcept { return m_excludedTrailing; }
    // The bound of the working storage, in values.
    [[nodiscard]] std::int64_t workspaceValues() const noexcept { return m_workspaceValues; }
    // The flops of the factorisation of one supernode (SymbolicFactor::flopCount for its columns).
    [[nodiscard]] static double supernodeFlops(std::int64_t columnCount, std::int64_t rowCount) noexcept;

private:
    void splitTree(const SymbolicFactor& symbolic, base::MemoryBudget& budget);
    void boundWorkspace(const SymbolicFactor& symbolic, std::size_t threadCount, PivotingMode pivoting,
                        TopOrder topOrder);

    base::BudgetedBuffer<std::int32_t> m_childStart; // supernodeCount + 1
    base::BudgetedBuffer<std::int32_t> m_children;
    base::BudgetedBuffer<std::int32_t> m_slot; // per supernode
    std::vector<FactorSubtree> m_subtrees;     // bounded by SUBTREE_SPLIT_COUNT per level of the top part
    std::vector<std::int32_t> m_topSupernodes;
    std::int32_t m_slotCount = 0;
    std::int32_t m_excludedTrailing = -1;
    std::int64_t m_workspaceValues = 0;
};

} // namespace oos::linalg
