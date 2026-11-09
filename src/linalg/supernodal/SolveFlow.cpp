#include "SolveFlow.hpp"

#include "../dense/LdltArithmetic.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <span>
#include <utility>

namespace oos::linalg::sweeps
{
namespace
{

using TaskId = base::TaskFlow::TaskId;
constexpr TaskId NO_TASK = std::numeric_limits<TaskId>::max();
// The flow's priorities: the earlier tiles first (so that a tile ends and its slot takes the next one); in a tile, the
// tasks on the chains of the triangles (the critical path of a large supernode) before the other products.
constexpr std::int64_t TILE_PRIORITY_STEP = 2;
constexpr std::int64_t CHAIN_PRIORITY = 1;

using Clock = std::chrono::steady_clock;

// The boundaries of the diagonal blocks of a panel's eliminated rows: TOP_ROW_BLOCK rows each, one more where a 2 × 2
// pivot would straddle a boundary.
std::vector<std::size_t> diagonalBlocks(const FactorPanel& panel)
{
    const std::size_t k = panel.eliminatedCount();
    const BlockDiagonalView d = panel.pivots();
    std::vector<std::size_t> bounds{0}; // bounded by a front
    while (bounds.back() < k)
    {
        std::size_t end = std::min(k, bounds.back() + TOP_ROW_BLOCK);
        if (end < k && startsTwoByTwoBlock(d, end - 1))
            ++end;
        bounds.push_back(end);
    }
    return bounds;
}

// The distinct tasks of `tasks` (NO_TASK left out), appended to `into`.
void addDistinct(std::vector<TaskId>& into, std::span<const TaskId> tasks)
{
    TaskId previous = NO_TASK;
    for (const TaskId task : tasks)
        if (task != NO_TASK && task != previous)
        {
            into.push_back(task);
            previous = task;
        }
    std::sort(into.begin(), into.end());
    into.erase(std::unique(into.begin(), into.end()), into.end());
}

RowProductOperands productOn(double* top, std::size_t ldx)
{
    RowProductOperands operands;
    operands.x = top;
    operands.ldx = ldx;
    operands.width = ldx;
    return operands;
}

RowTriangleOperands triangleOn(double* top, std::size_t ldx, const std::int32_t* rows, std::size_t first,
                               std::size_t end, const FactorPanel& panel, bool isTransposed)
{
    RowTriangleOperands operands;
    operands.x = top;
    operands.ldx = ldx;
    operands.width = ldx;
    operands.rows = rows + first;
    operands.size = end - first;
    operands.l = panel.values.pointer(first, first);
    operands.ldl = panel.values.leadingDimension;
    operands.isTransposed = isTransposed;
    return operands;
}

} // namespace

// The buffers of a tile in flight, taken by the next tile once it ends.
struct SolveFlow::Slot
{
    std::vector<double> accumulators; // the subtrees' outside rows × the tile's columns (forward sweeps)
    std::vector<double> top;          // the top block: its rows × the tile's columns
    std::vector<TaskId> lastWriter;   // per row of the top block, while the tile's tasks are made
};

// Makes the tasks of a tile (the class comment of SolveFlow); its last task makes those of the tile that takes its
// slot.
class SolveFlow::TileBuilder
{
public:
    TileBuilder(const SolveFlow& owner, base::TaskFlow& flow, std::span<const Tile> tiles,
                std::span<const std::vector<std::uint8_t>> isActive, std::span<Slot> slots, BlockPool& blocks,
                bool isForward, bool isBackward, bool isDiagonalFused)
        : m_owner(owner),
          m_flow(flow),
          m_tiles(tiles),
          m_isActive(isActive),
          m_slots(slots),
          m_blocks(blocks),
          m_isForward(isForward),
          m_isBackward(isBackward),
          m_isDiagonalFused(isDiagonalFused)
    {
    }

    void add(std::size_t t)
    {
        Slot& slot = m_slots[t % m_slots.size()];
        const std::int64_t priority = -static_cast<std::int64_t>(t) * TILE_PRIORITY_STEP;
        const SolvePlan& plan = m_owner.m_plan;
        const std::span<const SolvePlan::Subtree> subtrees = plan.subtrees();
        const std::span<const SupernodeRange> cut = m_owner.m_factor.subtrees();
        const std::size_t rowCount = m_owner.m_topRows.size();
        std::fill(slot.lastWriter.begin(), slot.lastWriter.end(), NO_TASK);
        const auto isSubtreeActive = [&](std::size_t g)
        { return m_isActive[t][static_cast<std::size_t>(cut[g].last)] != 0; };

        std::vector<TaskId> subtreeTasks; // bounded by the subtrees
        if (m_isForward)
        {
            for (std::size_t g = 0; g < subtrees.size(); ++g)
                if (isSubtreeActive(g))
                    subtreeTasks.push_back(m_flow.add([this, t, g] { return forwardSubtree(t, g); }, {}, priority));
        }
        // The tasks the slot's next tile waits for: every task of this one comes before a write-back, a subtree's
        // backward sweep or a gather.
        std::vector<TaskId> ends; // bounded by the pieces and the subtrees
        if (m_isForward || m_isBackward)
            for (std::size_t first = 0; first < rowCount; first += TOP_COPY_ROWS)
            {
                const std::size_t end = std::min(rowCount, first + TOP_COPY_ROWS);
                const TaskId id = m_flow.add([this, t, first, end] { return gatherTop(t, first, end); }, subtreeTasks,
                                             priority + CHAIN_PRIORITY);
                ends.push_back(id);
                std::fill(slot.lastWriter.begin() + static_cast<std::ptrdiff_t>(first),
                          slot.lastWriter.begin() + static_cast<std::ptrdiff_t>(end), id);
            }
        const std::span<const std::int32_t> supernodes = m_owner.m_topSupernodes;
        if (m_isForward)
            for (std::size_t q = 0; q < supernodes.size(); ++q)
                if (m_isActive[t][static_cast<std::size_t>(supernodes[q])] != 0)
                    addForward(t, slot, q, priority);
        if (m_isBackward)
            for (std::size_t q = supernodes.size(); q-- > 0;)
                addBackward(t, slot, q, priority);

        // The rows written back: the top supernodes' eliminated rows after a backward sweep (the trailing rows of a
        // partial factor are only read there), every row after a forward sweep alone.
        const std::size_t writtenRows = m_isBackward ? m_owner.m_topEliminatedRowCount : rowCount;
        std::vector<TaskId> written; // the write-backs
        std::vector<TaskId> dependencies;
        for (std::size_t first = 0; first < writtenRows; first += TOP_COPY_ROWS)
        {
            const std::size_t end = std::min(writtenRows, first + TOP_COPY_ROWS);
            dependencies.clear();
            addDistinct(dependencies, std::span<const TaskId>(slot.lastWriter).subspan(first, end - first));
            written.push_back(m_flow.add([this, t, first, end] { return scatterTop(t, first, end); }, dependencies,
                                         priority + CHAIN_PRIORITY));
        }
        ends.insert(ends.end(), written.begin(), written.end());
        // A factor without top rows (a forest of subtrees: no top supernode and no outside row) has no gather and no
        // write-back: the subtrees' forward sweeps are then what a backward sweep and the slot's next tile wait for
        // (their rows and the slot's accumulators), as the write-backs make them wait otherwise.
        std::vector<TaskId> beforeBackward = written;
        beforeBackward.insert(beforeBackward.end(), subtreeTasks.begin(), subtreeTasks.end());
        ends.insert(ends.end(), subtreeTasks.begin(), subtreeTasks.end());
        if (m_isBackward)
            for (std::size_t g = 0; g < subtrees.size(); ++g)
                ends.push_back(m_flow.add([this, t, g] { return backwardSubtree(t, g); }, beforeBackward, priority));
        if (t + m_slots.size() < m_tiles.size())
            m_flow.add(
                [this, t]
                {
                    add(t + m_slots.size());
                    return true;
                },
                ends, priority);
    }

private:
    [[nodiscard]] SweepContext alone() const
    {
        return SweepContext{m_owner.m_factor, m_owner.m_plan,         m_owner.m_kernels,
                            nullptr,          m_owner.m_cancellation, m_isDiagonalFused};
    }
    [[nodiscard]] double* topOf(std::size_t t) const { return m_slots[t % m_slots.size()].top.data(); }
    [[nodiscard]] std::span<const std::int32_t> positionsOf(std::size_t q) const
    {
        const std::size_t first = static_cast<std::size_t>(m_owner.m_positionOffset[q]);
        return std::span<const std::int32_t>(m_owner.m_positions)
            .subspan(first, static_cast<std::size_t>(m_owner.m_positionOffset[q + 1]) - first);
    }

    // The forward sweep of subtree g into its accumulator (FactorSolve's sweep by subtrees).
    bool forwardSubtree(std::size_t t, std::size_t g) const
    {
        const Tile& tile = m_tiles[t];
        const SolvePlan::Subtree& subtree = m_owner.m_plan.subtrees()[g];
        BlockLease block(m_blocks);
        double* accumulator = m_slots[t % m_slots.size()].accumulators.data() +
                              static_cast<std::size_t>(subtree.firstOutsideRow) * tile.ldx;
        std::fill_n(accumulator, static_cast<std::size_t>(subtree.outsideRowCount) * tile.ldx, 0.0);
        const std::span<const SolvePlan::Unit> units = m_owner.m_plan.units();
        const SweepContext context = alone();
        for (std::int32_t u = subtree.firstUnit; u < subtree.endUnit; ++u)
            forwardUnit(context, tile, units[static_cast<std::size_t>(u)], accumulator, m_isActive[t], block.data());
        return true;
    }

    // The backward sweep of subtree g, from the rows of its ancestors written back.
    bool backwardSubtree(std::size_t t, std::size_t g) const
    {
        const Tile& tile = m_tiles[t];
        const SolvePlan::Subtree& subtree = m_owner.m_plan.subtrees()[g];
        BlockLease block(m_blocks);
        const std::span<const SolvePlan::Unit> units = m_owner.m_plan.units();
        const SweepContext context = alone();
        for (std::int32_t u = subtree.endUnit; u-- > subtree.firstUnit;)
            backwardUnit(context, tile, units[static_cast<std::size_t>(u)], block.data());
        return true;
    }

    // Rows [first, end) of the top block from the right-hand sides, then (forward) the subtrees' accumulators added to
    // them in subtree order: the sums of FactorSolve's addAccumulators, row by row.
    bool gatherTop(std::size_t t, std::size_t first, std::size_t end) const
    {
        const Tile& tile = m_tiles[t];
        double* top = topOf(t);
        const std::span<const std::int32_t> rows = m_owner.m_topRows;
        loadRows(tile, rows, {}, nullptr, rows.size(), top, first, end);
        if (!m_isForward)
            return true;
        const std::span<const SolvePlan::Subtree> subtrees = m_owner.m_plan.subtrees();
        const std::span<const SupernodeRange> cut = m_owner.m_factor.subtrees();
        const double* accumulators = m_slots[t % m_slots.size()].accumulators.data();
        for (std::size_t g = 0; g < subtrees.size(); ++g)
        {
            if (m_isActive[t][static_cast<std::size_t>(cut[g].last)] == 0)
                continue;
            const auto outsideFirst = static_cast<std::size_t>(subtrees[g].firstOutsideRow);
            const std::span<const std::int32_t> positions =
                std::span<const std::int32_t>(m_owner.m_outsidePositions)
                    .subspan(outsideFirst, static_cast<std::size_t>(subtrees[g].outsideRowCount));
            // The outside rows are increasing, and so are their positions.
            const auto from = std::lower_bound(positions.begin(), positions.end(), static_cast<std::int32_t>(first));
            const auto to = std::lower_bound(from, positions.end(), static_cast<std::int32_t>(end));
            for (auto at = from; at != to; ++at)
            {
                const auto q = static_cast<std::size_t>(at - positions.begin());
                double* row = top + static_cast<std::size_t>(*at) * tile.ldx;
                const double* added = accumulators + (outsideFirst + q) * tile.ldx;
                for (std::size_t c = 0; c < tile.width; ++c)
                    row[c] += added[c];
            }
        }
        return true;
    }

    // Rows [first, end) of the top block back to the right-hand sides, without −0.
    bool scatterTop(std::size_t t, std::size_t first, std::size_t end) const
    {
        const std::span<const std::int32_t> rows = m_owner.m_topRows;
        storeRows(m_tiles[t], rows, {}, nullptr, rows.size(), topOf(t), first, end);
        return true;
    }

    // The distinct last writers of the rows `positions` of the slot's top block, added to `into`.
    static void writersOf(const Slot& slot, std::span<const std::int32_t> positions, std::vector<TaskId>& into)
    {
        std::vector<TaskId> writers; // bounded by a front
        writers.reserve(positions.size());
        for (const std::int32_t position : positions)
            writers.push_back(slot.lastWriter[static_cast<std::size_t>(position)]);
        addDistinct(into, writers);
    }

    static void setWriter(Slot& slot, std::span<const std::int32_t> positions, TaskId task)
    {
        for (const std::int32_t position : positions)
            slot.lastWriter[static_cast<std::size_t>(position)] = task;
    }

    // The forward tasks of the q-th top supernode.
    void addForward(std::size_t t, Slot& slot, std::size_t q, std::int64_t priority)
    {
        const std::int32_t s = m_owner.m_topSupernodes[q];
        const FactorPanel panel = m_owner.m_factor.panel(s);
        const std::size_t k = panel.eliminatedCount();
        const std::size_t r = panel.rows.size();
        if (k == 0)
            return;
        const std::span<const std::int32_t> positions = positionsOf(q);
        const Tile& tile = m_tiles[t];
        const auto work = (static_cast<std::uint64_t>(k) * (k + 1) / 2 + static_cast<std::uint64_t>(r - k) * k) *
                          static_cast<std::uint64_t>(tile.ldx);
        const DenseKernels& kernels = m_owner.m_kernels;
        const bool isFused = m_isDiagonalFused;
        const std::int32_t* pos = positions.data();
        const std::size_t ldx = tile.ldx;
        const std::size_t width = tile.width;
        double* const top = topOf(t);
        std::vector<TaskId> dependencies;
        if (work < TOP_TASK_MULTIPLY_ADDS)
        {
            writersOf(slot, positions, dependencies);
            const TaskId id = m_flow.add(
                [&kernels, s, k, r, pos, top, ldx, width, isFused, this]
                {
                    const FactorPanel p = m_owner.m_factor.panel(s);
                    kernels.solveRowTriangle(triangleOn(top, ldx, pos, 0, k, p, false));
                    if (r > k)
                    {
                        RowProductOperands below = productOn(top, ldx);
                        below.targets = pos + k;
                        below.targetCount = r - k;
                        below.sources = pos;
                        below.termCount = k;
                        below.l = p.values.pointer(k, 0);
                        below.targetStride = 1;
                        below.termStride = static_cast<std::ptrdiff_t>(p.values.leadingDimension);
                        kernels.subtractRowProducts(below);
                    }
                    if (isFused)
                        applyDiagonal(p, std::span<const std::int32_t>(pos, r), top, ldx, 0, width);
                    return true;
                },
                dependencies, priority + CHAIN_PRIORITY);
            setWriter(slot, positions, id);
            return;
        }
        const std::vector<std::size_t> bounds = diagonalBlocks(panel);
        const std::size_t m = bounds.size() - 1;
        const std::size_t belowBlocks = base::pieceCount(r - k, TOP_ROW_BLOCK);
        // The task each block's chain last went through (the rows' writers before the supernode at first).
        std::vector<std::vector<TaskId>> previous(m + belowBlocks); // bounded by a front
        for (std::size_t i = 0; i < m; ++i)
            writersOf(slot, positions.subspan(bounds[i], bounds[i + 1] - bounds[i]), previous[i]);
        for (std::size_t c = 0; c < belowBlocks; ++c)
        {
            const std::size_t first = k + c * TOP_ROW_BLOCK;
            writersOf(slot, positions.subspan(first, std::min(r, first + TOP_ROW_BLOCK) - first), previous[m + c]);
        }
        const std::ptrdiff_t ld = static_cast<std::ptrdiff_t>(panel.values.leadingDimension);
        for (std::size_t j = 0; j < m; ++j)
        {
            const std::size_t aj = bounds[j];
            const std::size_t bj = bounds[j + 1];
            const TaskId triangle = m_flow.add(
                [&kernels, s, aj, bj, pos, top, ldx, this]
                {
                    kernels.solveRowTriangle(triangleOn(top, ldx, pos, aj, bj, m_owner.m_factor.panel(s), false));
                    return true;
                },
                previous[j], priority + CHAIN_PRIORITY);
            std::vector<TaskId> readers{triangle};
            const auto addProduct = [&](std::size_t target, std::size_t first, std::size_t end, bool isChain)
            {
                dependencies.assign(previous[target].begin(), previous[target].end());
                dependencies.push_back(triangle);
                const TaskId id = m_flow.add(
                    [&kernels, s, aj, bj, first, end, pos, top, ldx, ld, this]
                    {
                        const FactorPanel p = m_owner.m_factor.panel(s);
                        RowProductOperands operands = productOn(top, ldx);
                        operands.targets = pos + first;
                        operands.targetCount = end - first;
                        operands.sources = pos + aj;
                        operands.termCount = bj - aj;
                        operands.l = p.values.pointer(first, aj);
                        operands.targetStride = 1;
                        operands.termStride = ld;
                        kernels.subtractRowProducts(operands);
                        return true;
                    },
                    dependencies, priority + (isChain ? CHAIN_PRIORITY : 0));
                previous[target] = {id};
                readers.push_back(id);
            };
            for (std::size_t i = j + 1; i < m; ++i)
                addProduct(i, bounds[i], bounds[i + 1], i == j + 1);
            for (std::size_t c = 0; c < belowBlocks; ++c)
            {
                const std::size_t first = k + c * TOP_ROW_BLOCK;
                addProduct(m + c, first, std::min(r, first + TOP_ROW_BLOCK), false);
            }
            TaskId last = triangle;
            if (isFused)
                last = m_flow.add(
                    [s, aj, bj, pos, top, ldx, width, r, this]
                    {
                        applyDiagonal(m_owner.m_factor.panel(s), std::span<const std::int32_t>(pos, r), top, ldx, 0,
                                      width, aj, bj);
                        return true;
                    },
                    readers, priority);
            setWriter(slot, positions.subspan(aj, bj - aj), last);
        }
        for (std::size_t c = 0; c < belowBlocks; ++c)
        {
            const std::size_t first = k + c * TOP_ROW_BLOCK;
            OOS_ASSERT(previous[m + c].size() == 1);
            setWriter(slot, positions.subspan(first, std::min(r, first + TOP_ROW_BLOCK) - first),
                      previous[m + c].front());
        }
    }

    // The backward tasks of the q-th top supernode.
    void addBackward(std::size_t t, Slot& slot, std::size_t q, std::int64_t priority)
    {
        const std::int32_t s = m_owner.m_topSupernodes[q];
        const FactorPanel panel = m_owner.m_factor.panel(s);
        const std::size_t k = panel.eliminatedCount();
        const std::size_t r = panel.rows.size();
        if (k == 0)
            return;
        const std::span<const std::int32_t> positions = positionsOf(q);
        const Tile& tile = m_tiles[t];
        const auto work = (static_cast<std::uint64_t>(k) * (k + 1) / 2 + static_cast<std::uint64_t>(r - k) * k) *
                          static_cast<std::uint64_t>(tile.ldx);
        const DenseKernels& kernels = m_owner.m_kernels;
        const std::int32_t* pos = positions.data();
        const std::size_t ldx = tile.ldx;
        double* const top = topOf(t);
        const std::ptrdiff_t ld = static_cast<std::ptrdiff_t>(panel.values.leadingDimension);
        std::vector<TaskId> below; // the writers of the rows below
        writersOf(slot, positions.subspan(k), below);
        // x(F) ← x(F) − L_RFᵀ·x(R) on the eliminated rows [first, end), the terms in panel order.
        const auto fromBelow =
            [&kernels, k, r, pos, top, ldx, ld](const FactorPanel& p, std::size_t first, std::size_t end)
        {
            RowProductOperands operands = productOn(top, ldx);
            operands.targets = pos + first;
            operands.targetCount = end - first;
            operands.sources = pos + k;
            operands.termCount = r - k;
            operands.l = p.values.pointer(k, first);
            operands.targetStride = ld;
            operands.termStride = 1;
            kernels.subtractRowProducts(operands);
        };
        if (work < TOP_TASK_MULTIPLY_ADDS)
        {
            std::vector<TaskId> dependencies = below;
            writersOf(slot, positions.first(k), dependencies);
            const TaskId id = m_flow.add(
                [&kernels, s, k, r, pos, top, ldx, fromBelow, this]
                {
                    const FactorPanel p = m_owner.m_factor.panel(s);
                    if (r > k)
                        fromBelow(p, 0, k);
                    kernels.solveRowTriangle(triangleOn(top, ldx, pos, 0, k, p, true));
                    return true;
                },
                dependencies, priority + CHAIN_PRIORITY);
            setWriter(slot, positions.first(k), id);
            return;
        }
        const std::vector<std::size_t> bounds = diagonalBlocks(panel);
        const std::size_t m = bounds.size() - 1;
        std::vector<std::vector<TaskId>> previous(m); // bounded by a front
        for (std::size_t i = 0; i < m; ++i)
        {
            std::vector<TaskId> dependencies = below;
            writersOf(slot, positions.subspan(bounds[i], bounds[i + 1] - bounds[i]), dependencies);
            if (r == k)
            {
                previous[i] = std::move(dependencies);
                continue;
            }
            const std::size_t first = bounds[i];
            const std::size_t end = bounds[i + 1];
            previous[i] = {m_flow.add(
                [s, first, end, fromBelow, this]
                {
                    fromBelow(m_owner.m_factor.panel(s), first, end);
                    return true;
                },
                dependencies, priority)};
        }
        for (std::size_t j = m; j-- > 0;)
        {
            const std::size_t aj = bounds[j];
            const std::size_t bj = bounds[j + 1];
            const TaskId triangle = m_flow.add(
                [&kernels, s, aj, bj, pos, top, ldx, this]
                {
                    kernels.solveRowTriangle(triangleOn(top, ldx, pos, aj, bj, m_owner.m_factor.panel(s), true));
                    return true;
                },
                previous[j], priority + CHAIN_PRIORITY);
            setWriter(slot, positions.subspan(aj, bj - aj), triangle);
            for (std::size_t i = j; i-- > 0;)
            {
                const std::size_t ai = bounds[i];
                const std::size_t bi = bounds[i + 1];
                std::vector<TaskId> dependencies = previous[i];
                dependencies.push_back(triangle);
                previous[i] = {m_flow.add(
                    [&kernels, s, ai, bi, aj, bj, pos, top, ldx, ld, this]
                    {
                        // Term p is the row bj − 1 − p, with l(i, p) = L(bj − 1 − p, ai + i): q decreasing.
                        RowProductOperands operands = productOn(top, ldx);
                        operands.targets = pos + ai;
                        operands.targetCount = bi - ai;
                        operands.sources = pos + (bj - 1);
                        operands.sourceStep = -1;
                        operands.termCount = bj - aj;
                        operands.l = m_owner.m_factor.panel(s).values.pointer(bj - 1, ai);
                        operands.targetStride = ld;
                        operands.termStride = -1;
                        kernels.subtractRowProducts(operands);
                        return true;
                    },
                    dependencies, priority + (i + 1 == j ? CHAIN_PRIORITY : 0))};
            }
        }
    }

    const SolveFlow& m_owner;
    base::TaskFlow& m_flow;
    std::span<const Tile> m_tiles;
    std::span<const std::vector<std::uint8_t>> m_isActive;
    std::span<Slot> m_slots;
    BlockPool& m_blocks;
    bool m_isForward;
    bool m_isBackward;
    bool m_isDiagonalFused;
};

SolveFlow::SolveFlow(const SupernodalFactor& factor, const SolvePlan& plan, const DenseKernels& kernels,
                     base::ThreadPool& pool, const base::CancellationToken* cancellation)
    : m_factor(factor), m_plan(plan), m_kernels(kernels), m_pool(pool), m_cancellation(cancellation)
{
    OOS_ASSERT(plan.isByTree());
    for (const SolvePlan::Unit& unit : plan.units())
        if (unit.subtree == SolvePlan::NO_SUBTREE)
        {
            OOS_ASSERT(unit.firstSupernode == unit.lastSupernode);
            m_topSupernodes.push_back(unit.firstSupernode);
        }
    // The top block's rows: the top panels' rows and the subtrees' outside rows, increasing.
    std::vector<std::uint8_t> isEliminatedAtTop(static_cast<std::size_t>(factor.dimension()), 0); // per equation
    for (const std::int32_t s : m_topSupernodes)
    {
        const FactorPanel panel = factor.panel(s);
        m_topRows.insert(m_topRows.end(), panel.rows.begin(), panel.rows.end());
        for (const std::int32_t row : panel.rows.first(panel.eliminatedCount()))
            isEliminatedAtTop[static_cast<std::size_t>(row)] = 1;
    }
    for (const SolvePlan::Subtree& subtree : plan.subtrees())
    {
        const std::span<const std::int32_t> outside = plan.outsideRows(subtree);
        m_topRows.insert(m_topRows.end(), outside.begin(), outside.end());
    }
    std::sort(m_topRows.begin(), m_topRows.end());
    m_topRows.erase(std::unique(m_topRows.begin(), m_topRows.end()), m_topRows.end());
    // The top supernodes' eliminated rows come first (the trailing rows of a partial factor are the last equations).
    while (m_topEliminatedRowCount < m_topRows.size() &&
           isEliminatedAtTop[static_cast<std::size_t>(m_topRows[m_topEliminatedRowCount])] != 0)
        ++m_topEliminatedRowCount;
    for (std::size_t i = m_topEliminatedRowCount; i < m_topRows.size(); ++i)
        OOS_ASSERT(isEliminatedAtTop[static_cast<std::size_t>(m_topRows[i])] == 0 &&
                          m_topRows[i] >= factor.eliminatedCount());
    const std::int32_t firstRow = m_topRows.empty() ? 0 : m_topRows.front();
    std::vector<std::int32_t> positionOf(static_cast<std::size_t>(factor.dimension() - firstRow), -1); // per equation
    for (std::size_t i = 0; i < m_topRows.size(); ++i)
        positionOf[static_cast<std::size_t>(m_topRows[i] - firstRow)] = static_cast<std::int32_t>(i);
    m_positionOffset.push_back(0);
    for (const std::int32_t s : m_topSupernodes)
    {
        for (const std::int32_t row : factor.panel(s).rows)
            m_positions.push_back(positionOf[static_cast<std::size_t>(row - firstRow)]);
        m_positionOffset.push_back(static_cast<std::int64_t>(m_positions.size()));
    }
    for (const SolvePlan::Subtree& subtree : plan.subtrees())
        for (const std::int32_t row : plan.outsideRows(subtree))
            m_outsidePositions.push_back(positionOf[static_cast<std::size_t>(row - firstRow)]);
}

void SolveFlow::run(DenseMatrixView rightHandSides, std::size_t width, std::size_t inFlight, bool isForward,
                    bool isBackward, bool isDiagonalFused)
{
    const auto started = Clock::now();
    const std::size_t columnCount = rightHandSides.columns;
    const std::size_t lanes = m_kernels.rowLanes();
    const std::size_t tileCount = base::pieceCount(columnCount, width);
    std::vector<Tile> tiles; // one per tile
    for (std::size_t t = 0; t < tileCount; ++t)
    {
        const std::size_t first = t * width;
        const std::size_t columns = std::min(width, columnCount - first);
        tiles.push_back(
            Tile{rightHandSides.block(0, first, rightHandSides.rows, columns), columns, roundUp(columns, lanes)});
    }
    // The supernodes each tile's forward sweep visits, made before the flow (their scans are batches on the pool).
    std::vector<std::vector<std::uint8_t>> isActive(tileCount); // one per tile
    if (isForward)
    {
        const SweepContext pooled{m_factor, m_plan, m_kernels, &m_pool, m_cancellation, isDiagonalFused};
        for (std::size_t t = 0; t < tileCount; ++t)
        {
            canonicalizeTrailingRows(pooled, tiles[t]);
            isActive[t] = activeSupernodes(pooled, tiles[t]);
        }
    }
    const std::size_t ldx = roundUp(width, lanes);
    std::vector<Slot> slots(std::max<std::size_t>(1, std::min(inFlight, tileCount))); // the tiles in flight
    for (Slot& slot : slots)
    {
        if (isForward)
            slot.accumulators.assign(static_cast<std::size_t>(m_plan.outsideRowCount()) * ldx, 0.0);
        slot.top.assign(m_topRows.size() * ldx, 0.0);
        slot.lastWriter.assign(m_topRows.size(), NO_TASK);
    }
    // A block per running subtree task: the largest unit of a subtree.
    std::size_t subtreeUnitRows = 1;
    for (const SolvePlan::Unit& unit : m_plan.units())
        if (unit.subtree != SolvePlan::NO_SUBTREE)
            subtreeUnitRows = std::max(subtreeUnitRows, static_cast<std::size_t>(unit.rowCount));
    BlockPool blocks(subtreeUnitRows * ldx);
    base::TaskFlow flow;
    TileBuilder builder(*this, flow, tiles, isActive, slots, blocks, isForward, isBackward, isDiagonalFused);
    for (std::size_t t = 0; t < slots.size() && t < tileCount; ++t)
        builder.add(t);
    flow.run(&m_pool, m_cancellation);
    m_workSeconds = flow.workSeconds();
    m_wallSeconds = std::chrono::duration<double>(Clock::now() - started).count();
}

} // namespace oos::linalg::sweeps
