#include "FactorSchedule.hpp"

#include "WorkspacePool.hpp"

#include "oos/base/errors/InternalError.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>

namespace oos::linalg
{
namespace
{

std::int64_t squared(std::int64_t value) noexcept
{
    return value * value;
}

} // namespace

double FactorSchedule::supernodeFlops(std::int64_t columnCount, std::int64_t rowCount) noexcept
{
    // Per column with m entries below the diagonal: m(m + 1) + 2m (SymbolicFactor::flopCount).
    double flops = 0.0;
    for (std::int64_t m = rowCount; m < rowCount + columnCount; ++m)
        flops += static_cast<double>(m) * static_cast<double>(m + 1) + 2.0 * static_cast<double>(m);
    return flops;
}

FactorSchedule::FactorSchedule(const SymbolicFactor& symbolic, bool excludesTrailing, std::size_t threadCount,
                               PivotingMode pivoting, base::MemoryBudget& budget, TopOrder topOrder)
{
    const std::int32_t count = symbolic.supernodeCount();
    const auto total = static_cast<std::size_t>(count);
    const Supernodes& supernodes = symbolic.supernodes();
    if (excludesTrailing && symbolic.trailingDofCount() > 0)
        m_excludedTrailing = count - 1;

    // Children in increasing order (a counting sort on the parents).
    m_childStart = base::BudgetedBuffer<std::int32_t>(budget, total + 1, "factorisation plan: child starts");
    m_children = base::BudgetedBuffer<std::int32_t>(budget, total, "factorisation plan: children");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = supernodes.parent(s);
        if (parent != Supernodes::ROOT)
            ++m_childStart[static_cast<std::size_t>(parent) + 1];
    }
    for (std::size_t s = 0; s < total; ++s)
        m_childStart[s + 1] += m_childStart[s];
    {
        base::BudgetedBuffer<std::int32_t> cursor(budget, total, "factorisation plan: cursors");
        std::copy_n(m_childStart.data(), total, cursor.data());
        for (std::int32_t s = 0; s < count; ++s)
        {
            const std::int32_t parent = supernodes.parent(s);
            if (parent != Supernodes::ROOT)
                m_children[static_cast<std::size_t>(cursor[static_cast<std::size_t>(parent)]++)] = s;
        }
    }

    m_slot = base::BudgetedBuffer<std::int32_t>(budget, total, "factorisation plan: slots");
    std::fill_n(m_slot.data(), total, NO_SLOT);
    splitTree(symbolic, budget);
    boundWorkspace(symbolic, std::max<std::size_t>(threadCount, 1), pivoting, topOrder);
}

std::span<const std::int32_t> FactorSchedule::children(std::int32_t supernode) const noexcept
{
    const auto begin = static_cast<std::size_t>(m_childStart[static_cast<std::size_t>(supernode)]);
    const auto end = static_cast<std::size_t>(m_childStart[static_cast<std::size_t>(supernode) + 1]);
    return m_children.span().subspan(begin, end - begin);
}

void FactorSchedule::splitTree(const SymbolicFactor& symbolic, base::MemoryBudget& budget)
{
    const std::int32_t count = symbolic.supernodeCount();
    const auto total = static_cast<std::size_t>(count);
    const Supernodes& supernodes = symbolic.supernodes();

    // The flops of every subtree and its first supernode (in a postorder the subtree of s is [first(s), s]).
    base::BudgetedBuffer<double> subtreeFlops(budget, total, "factorisation plan: subtree flops");
    base::BudgetedBuffer<std::int32_t> firstOfSubtree(budget, total, "factorisation plan: subtree starts");
    double totalFlops = 0.0;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        if (s == m_excludedTrailing)
            continue;
        subtreeFlops[at] += supernodeFlops(symbolic.columnCount(s), symbolic.rowCount(s));
        totalFlops += supernodeFlops(symbolic.columnCount(s), symbolic.rowCount(s));
        firstOfSubtree[at] = s;
        for (const std::int32_t child : children(s))
            firstOfSubtree[at] = std::min(firstOfSubtree[at], firstOfSubtree[static_cast<std::size_t>(child)]);
        const std::int32_t parent = supernodes.parent(s);
        if (parent != Supernodes::ROOT)
            subtreeFlops[static_cast<std::size_t>(parent)] += subtreeFlops[at];
    }

    // Split the heaviest subtree while it is above the limit (ties by the higher index: a deterministic order).
    const double limit = std::max(totalFlops / SUBTREE_SPLIT_COUNT, SUBTREE_MIN_FLOPS);
    using Candidate = std::pair<double, std::int32_t>;
    std::priority_queue<Candidate> heaviest;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = supernodes.parent(s);
        if (s == m_excludedTrailing)
            continue;
        if (parent == Supernodes::ROOT || parent == m_excludedTrailing)
            heaviest.emplace(subtreeFlops[static_cast<std::size_t>(s)], s);
    }
    std::vector<std::int32_t> subtreeRoots;
    while (!heaviest.empty())
    {
        const auto [flops, s] = heaviest.top();
        heaviest.pop();
        if (flops > limit && !children(s).empty())
        {
            m_topSupernodes.push_back(s);
            for (const std::int32_t child : children(s))
                heaviest.emplace(subtreeFlops[static_cast<std::size_t>(child)], child);
        }
        else
            subtreeRoots.push_back(s);
    }
    std::sort(subtreeRoots.begin(), subtreeRoots.end());
    std::sort(m_topSupernodes.begin(), m_topSupernodes.end());
    if (m_excludedTrailing >= 0)
        m_topSupernodes.push_back(m_excludedTrailing);

    for (const std::int32_t root : subtreeRoots)
    {
        FactorSubtree subtree;
        subtree.first = firstOfSubtree[static_cast<std::size_t>(root)];
        subtree.last = root;
        subtree.flops = subtreeFlops[static_cast<std::size_t>(root)];
        m_subtrees.push_back(subtree);
        m_slot[static_cast<std::size_t>(root)] = m_slotCount++;
    }
    for (const std::int32_t s : m_topSupernodes)
        m_slot[static_cast<std::size_t>(s)] = m_slotCount++;
}

std::vector<std::size_t> FactorSchedule::subtreesByWork() const
{
    std::vector<std::size_t> order(m_subtrees.size());
    for (std::size_t k = 0; k < order.size(); ++k)
        order[k] = k;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return m_subtrees[a].flops > m_subtrees[b].flops; });
    return order;
}

void FactorSchedule::boundWorkspace(const SymbolicFactor& symbolic, std::size_t threadCount, PivotingMode pivoting,
                                    TopOrder topOrder)
{
    // A positive definite front is assembled in place (MultifrontalDriver::assemble): no front buffer.
    const bool isInPlace = pivoting == PivotingMode::PositiveDefinite;
    // The blocks of the working storage come in the size classes of WorkspacePool, which the bound counts.
    const auto frontOf = [&](std::int32_t s)
    { return WorkspacePool::classSize(squared(symbolic.columnCount(s) + symbolic.rowCount(s))); };
    const auto updateOf = [&](std::int32_t s) { return WorkspacePool::classSize(squared(symbolic.rowCount(s))); };

    // Each subtree run serially: one front buffer of its largest front (indefinite mode), and its stack of update
    // matrices (the children of a supernode are the top entries when it is assembled; they are released once it is
    // assembled, then its own is pushed). In place, a supernode's update matrix is allocated by its assembly while its
    // children's are still on the stack: the stack before it plus its own.
    std::vector<std::int64_t> excess; // what a running subtree holds at its peak, its root's update matrix included
    std::int64_t finishedUpdates = 0;
    for (FactorSubtree& subtree : m_subtrees)
    {
        std::int64_t stack = 0;
        std::int64_t inPlacePeak = 0;
        for (std::int32_t s = subtree.first; s <= subtree.last; ++s)
        {
            subtree.frontValues = std::max(subtree.frontValues, frontOf(s));
            inPlacePeak = std::max(inPlacePeak, stack + updateOf(s));
            for (const std::int32_t child : children(s))
                stack -= updateOf(child);
            stack += updateOf(s);
            subtree.stackValues = std::max(subtree.stackValues, stack);
        }
        OOS_ASSERT(stack == updateOf(subtree.last));
        subtree.contributionValues = updateOf(subtree.last);
        subtree.heldValues = isInPlace ? inPlacePeak : subtree.frontValues + subtree.stackValues;
        finishedUpdates += subtree.contributionValues;
        excess.push_back(subtree.heldValues);
    }
    // At most threadCount subtrees run at once: the largest excesses, over the update matrices of all of them.
    std::sort(excess.begin(), excess.end(), std::greater<>());
    std::int64_t peak = finishedUpdates;
    for (std::size_t k = 0; k < std::min(threadCount, excess.size()); ++k)
        peak += excess[k];

    if (topOrder == TopOrder::Serial)
    {
        // PERFORMANCE: the top part one supernode after the other in increasing order (the streamed factorisation):
        // its update matrix made beside the ones alive (and, not in place, its front), its children's released after
        // it; the waves of TaskGraph would count every supernode of a wave at once.
        // Measured: the interface problem of a large model's dissection units.
        std::int64_t alive = finishedUpdates;
        for (const std::int32_t s : m_topSupernodes)
        {
            if (s == m_excludedTrailing)
                continue; // assembled into the Schur complement, budgeted by the caller
            peak = std::max(peak, alive + (isInPlace ? 0 : frontOf(s)) + updateOf(s));
            alive += updateOf(s);
            for (const std::int32_t child : children(s))
                alive -= updateOf(child);
        }
        m_workspaceValues = peak;
        return;
    }
    // The top part by waves, as TaskGraph runs it: wave(s) = 1 + the largest wave of its child tasks (subtrees: 0).
    std::vector<std::int32_t> waveOfSlot(static_cast<std::size_t>(m_slotCount), 0);
    std::int32_t waveCount = 0;
    std::vector<std::int32_t> waveOfTop;
    for (const std::int32_t s : m_topSupernodes)
    {
        std::int32_t wave = 1;
        for (const std::int32_t child : children(s))
            wave = std::max(wave, waveOfSlot[static_cast<std::size_t>(slotOf(child))] + 1);
        waveOfSlot[static_cast<std::size_t>(slotOf(s))] = wave;
        waveOfTop.push_back(wave);
        waveCount = std::max(waveCount, wave);
    }
    std::vector<std::int64_t> waveHeld(static_cast<std::size_t>(waveCount) + 1, 0);
    std::vector<std::int64_t> waveProduced(static_cast<std::size_t>(waveCount) + 1, 0);
    std::vector<std::int64_t> waveConsumed(static_cast<std::size_t>(waveCount) + 1, 0);
    for (std::size_t k = 0; k < m_topSupernodes.size(); ++k)
    {
        const std::int32_t s = m_topSupernodes[k];
        const auto wave = static_cast<std::size_t>(waveOfTop[k]);
        for (const std::int32_t child : children(s))
            waveConsumed[wave] += updateOf(child);
        if (s == m_excludedTrailing)
            continue; // assembled into the Schur complement, budgeted by the caller
        waveHeld[wave] += (isInPlace ? 0 : frontOf(s)) + updateOf(s);
        waveProduced[wave] += updateOf(s);
    }
    std::int64_t alive = finishedUpdates;
    for (std::size_t wave = 1; wave < waveHeld.size(); ++wave)
    {
        peak = std::max(peak, alive + waveHeld[wave]);
        alive += waveProduced[wave] - waveConsumed[wave];
    }
    m_workspaceValues = peak;
}

} // namespace oos::linalg
