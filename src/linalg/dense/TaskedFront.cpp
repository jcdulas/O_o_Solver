#include "oos/linalg/dense/TaskedFront.hpp"

#include "BlockedLdlt.hpp"
#include "KernelFamily.hpp"
#include "LdltArithmetic.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/supernodal/UpdateLayout.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace oos::linalg
{
namespace
{

constexpr std::size_t NOT_RUN = std::numeric_limits<std::size_t>::max();

std::size_t roundedToGranule(std::size_t values) noexcept
{
    return (values + KERNEL_TILE_GRANULE - 1) / KERNEL_TILE_GRANULE * KERNEL_TILE_GRANULE;
}

// The panel blocks: ⌈c / KERNEL_INNER_BLOCK⌉ blocks of equal widths (the last no wider), so that each block's terms are
// one inner block of the micro-kernel and no block is a short remainder (a pass over every tile for a few terms).
std::vector<std::size_t> panelBlockFirsts(std::size_t columns)
{
    std::vector<std::size_t> firsts{0};
    if (columns == 0)
        return firsts;
    const std::size_t blocks = (columns + KERNEL_INNER_BLOCK - 1) / KERNEL_INNER_BLOCK;
    const std::size_t width = (columns + blocks - 1) / blocks;
    for (std::size_t first = width; first < columns; first += width)
        firsts.push_back(first);
    firsts.push_back(columns);
    return firsts;
}

} // namespace

std::vector<std::size_t> TaskedFront::trailingBlockFirsts(std::size_t rowsBelow, std::int64_t blockColumns)
{
    const UpdateLayout layout{static_cast<std::int64_t>(rowsBelow), blockColumns};
    const std::size_t width =
        layout.isBlocked() ? static_cast<std::size_t>(blockColumns) : TaskedFront::SQUARE_BLOCK_COLUMNS;
    std::vector<std::size_t> firsts;
    for (std::size_t first = 0; first < rowsBelow; first += width)
        firsts.push_back(first);
    firsts.push_back(rowsBelow);
    return firsts;
}

// The front's grid, its packed panels and what its tasks record.
struct TaskedFront::State
{
    const KernelFamily* family = nullptr;
    DenseKernels serial; // the diagonal blocks' recursion, on the task's thread
    DenseMatrixView panel;
    DenseMatrixView contribution;
    UpdateLayout layout;
    std::span<double> packing;
    std::vector<std::size_t> panelFirst;    // K + 1
    std::vector<std::size_t> trailingFirst; // J + 1
    std::vector<std::size_t> rowFirst;      // per row block (panel blocks, then trailing blocks), and n
    std::size_t termWidth = 0;              // the widest panel block
    std::vector<std::size_t> packOffset;    // per row block: Â at it, Ŵ after Â, within a slot
    std::size_t slotValues = 0;
    std::vector<double> diagonalProducts[2]; // per slot: L·D of a diagonal block (w(j, q), q < j)
    std::vector<std::size_t> eliminated;     // per panel block, NOT_RUN until F(k) ran
    // Per panel block: a zero or non-finite pivot in it or before it (written by F(k), read by the tasks after it).
    std::vector<std::uint8_t> hasFailed;

    [[nodiscard]] bool hasFailedBy(std::size_t k) const noexcept { return hasFailed[k] != 0; }

    State(KernelKind kind, DenseKernelOptions options) : serial(kind, options) {}

    [[nodiscard]] std::size_t panelBlocks() const noexcept { return panelFirst.size() - 1; }
    [[nodiscard]] std::size_t trailingBlocks() const noexcept { return trailingFirst.size() - 1; }
    [[nodiscard]] std::size_t rowsOf(std::size_t m) const noexcept { return rowFirst[m + 1] - rowFirst[m]; }
    [[nodiscard]] std::size_t widthOf(std::size_t k) const noexcept { return panelFirst[k + 1] - panelFirst[k]; }
    [[nodiscard]] double* packedRows(std::size_t k, std::size_t m) const noexcept
    {
        return packing.data() + (k % 2) * slotValues + packOffset[m];
    }
    [[nodiscard]] double* packedColumns(std::size_t k, std::size_t m) const noexcept
    {
        return packedRows(k, m) + roundedToGranule(rowsOf(m)) * termWidth;
    }
    [[nodiscard]] BlockDiagonalView pivots(std::size_t k) const noexcept
    {
        const std::size_t first = panelFirst[k];
        return {&panel(first, first), panel.leadingDimension + 1, nullptr, widthOf(k)};
    }

    // F(k): the diagonal block eliminated, then its L·D for the panel solves. At a zero or non-finite pivot (here or
    // in an earlier block) the front has failed: F(k) records it, and every task after it does nothing (each reads the
    // flag of its own block, written by an F it depends on: the same whatever the threads), so that the flow goes on
    // with the tasks that do not depend on the front.
    void factorDiagonal(std::size_t k)
    {
        if (k > 0 && hasFailedBy(k - 1))
        {
            hasFailed[k] = 1;
            return;
        }
        const std::size_t first = panelFirst[k];
        const std::size_t width = widthOf(k);
        const DenseMatrixView block = panel.block(first, first, width, width);
        const std::size_t done = BlockedLdlt(serial, *family).factorWithoutPivoting(block, width);
        eliminated[k] = done;
        if (done < width)
        {
            hasFailed[k] = 1;
            return;
        }
        // W = L·D of the block, row j and columns q < j (DenseKernels::solvePanel's expressions).
        std::vector<double>& w = diagonalProducts[k % 2];
        w.assign(width * width, 0.0);
        std::vector<double> row(width);
        const BlockDiagonalView d = pivots(k);
        for (std::size_t j = 1; j < width; ++j)
        {
            computeProductRow(block, j, 0, d, std::span<double>(row.data(), j));
            for (std::size_t q = 0; q < j; ++q)
                w[j + q * width] = row[q];
        }
    }

    void packRows(std::size_t k, std::size_t m) const noexcept
    {
        ProductTileOperands operands;
        operands.a = &panel(rowFirst[m], panelFirst[k]);
        operands.lda = panel.leadingDimension;
        operands.rowCount = rowsOf(m);
        operands.innerCount = widthOf(k);
        family->packProductRows(operands, 0, widthOf(k), packedRows(k, m));
    }

    void packColumns(std::size_t k, std::size_t m) const noexcept
    {
        ProductTileOperands operands;
        operands.b = &panel(rowFirst[m], panelFirst[k]);
        operands.ldb = panel.leadingDimension;
        operands.columnCount = rowsOf(m);
        operands.d = pivots(k);
        operands.innerCount = widthOf(k);
        family->packProductColumns(operands, 0, widthOf(k), packedColumns(k, m));
    }

    // S(k, m): the rows of row block m in panel block k's columns solved against its diagonal block; then packed for
    // the panel's later blocks (Â of every row block, Ŵ of the panel's row blocks).
    void solveRows(std::size_t k, std::size_t m) const noexcept
    {
        if (hasFailedBy(k))
            return;
        PanelSolveOperands operands;
        operands.p = &panel(rowFirst[m], panelFirst[k]);
        operands.ldp = panel.leadingDimension;
        operands.rowCount = rowsOf(m);
        operands.columnCount = widthOf(k);
        operands.w = diagonalProducts[k % 2].data();
        operands.ldw = widthOf(k);
        operands.d = pivots(k);
        family->solvePanel(operands);
        if (k + 1 < panelBlocks())
        {
            packRows(k, m);
            if (m < panelBlocks())
                packColumns(k, m);
        }
    }

    // U(k, m, j) and V(k, m, j): the tile of row block m and column block j (a panel block, or a trailing block when
    // isTrailing) receives block k's terms.
    void updateTile(std::size_t k, std::size_t m, std::size_t j, bool isTrailing) const noexcept
    {
        // A trailing tile runs after the whole panel: the last block's flag says whether the front failed.
        if (hasFailedBy(isTrailing ? panelBlocks() - 1 : k))
            return;
        ProductTileOperands operands;
        if (!isTrailing)
        {
            operands.c = &panel(rowFirst[m], panelFirst[j]);
            operands.ldc = panel.leadingDimension;
            operands.columnCount = widthOf(j);
        }
        else
        {
            const std::size_t trailing = panelBlocks();
            const std::size_t column = trailingFirst[j];
            const std::size_t row = rowFirst[m] - rowFirst[trailing];
            operands.columnCount = trailingFirst[j + 1] - column;
            if (layout.isBlocked())
            {
                const auto block = static_cast<std::int64_t>(j);
                operands.c = contribution.data + layout.blockOffset(block) + (row - column);
                operands.ldc = static_cast<std::size_t>(layout.order) - column;
            }
            else
            {
                operands.c = &contribution(row, column);
                operands.ldc = contribution.leadingDimension;
            }
        }
        operands.rowCount = rowsOf(m);
        operands.isLowerOnly = isTrailing ? m == j + panelBlocks() : m == j;
        operands.diagonalOffset = 0;
        const std::size_t columnBlock = isTrailing ? j + panelBlocks() : j;
        family->subtractPrepackedProduct(operands, widthOf(k), packedRows(k, m), packedColumns(k, columnBlock));
    }
};

std::size_t TaskedFront::packingValues(std::size_t columns, std::size_t rowsBelow,
                                       std::int64_t contributionBlockColumns) noexcept
{
    const std::vector<std::size_t> panel = panelBlockFirsts(columns);
    const std::vector<std::size_t> trailing = trailingBlockFirsts(rowsBelow, contributionBlockColumns);
    std::size_t width = 0;
    std::size_t perSlot = 0;
    for (std::size_t k = 0; k + 1 < panel.size(); ++k)
    {
        width = std::max(width, panel[k + 1] - panel[k]);
        perSlot += 2 * roundedToGranule(panel[k + 1] - panel[k]);
    }
    for (std::size_t j = 0; j + 1 < trailing.size(); ++j)
        perSlot += 2 * roundedToGranule(trailing[j + 1] - trailing[j]);
    const std::size_t slots = panel.size() > 2 ? 2 : 1;
    return slots * perSlot * width;
}

double TaskedFront::multiplyAdds(std::size_t columns, std::size_t rowsBelow) noexcept
{
    const auto c = static_cast<double>(columns);
    const auto r = static_cast<double>(rowsBelow);
    // The panel's LDLᵀ (c³/6 + c²·r/2) and the trailing block's update (c·r²/2).
    return c * c * c / 6.0 + c * c * r / 2.0 + c * r * r / 2.0;
}

bool TaskedFront::isWorthTasks(std::size_t columns, std::size_t rowsBelow, double minimumMultiplyAdds) noexcept
{
    return columns > 0 && multiplyAdds(columns, rowsBelow) >= minimumMultiplyAdds &&
           (columns <= KERNEL_INNER_BLOCK || columns + rowsBelow <= MAXIMUM_MULTIBLOCK_ORDER);
}

TaskedFront::TaskedFront(KernelKind kind, DenseMatrixView panel, DenseMatrixView contribution,
                         std::int64_t contributionBlockColumns, std::span<double> packing, KernelWorkspace* workspace)
{
    DenseKernelOptions serial;
    serial.workspace = workspace;
    m_state = std::make_unique<State>(kind, serial);
    State& state = *m_state;
    state.family = &kernelFamilyOf(kind);
    OOS_ASSERT(state.family->isBlocked && state.family->packProductRows != nullptr);
    state.panel = panel;
    state.contribution = contribution;
    const std::size_t rowsBelow = panel.rows - panel.columns;
    state.layout = UpdateLayout{static_cast<std::int64_t>(rowsBelow), contributionBlockColumns};
    state.panelFirst = panelBlockFirsts(panel.columns);
    state.trailingFirst = trailingBlockFirsts(rowsBelow, contributionBlockColumns);
    m_trailingFirst = state.trailingFirst;
    for (std::size_t k = 0; k < state.panelBlocks(); ++k)
    {
        state.rowFirst.push_back(state.panelFirst[k]);
        state.termWidth = std::max(state.termWidth, state.widthOf(k));
    }
    for (std::size_t j = 0; j < state.trailingBlocks(); ++j)
        state.rowFirst.push_back(panel.columns + state.trailingFirst[j]);
    state.rowFirst.push_back(panel.rows);
    std::size_t offset = 0;
    for (std::size_t m = 0; m + 1 < state.rowFirst.size(); ++m)
    {
        state.packOffset.push_back(offset);
        offset += 2 * roundedToGranule(state.rowsOf(m)) * state.termWidth;
    }
    state.slotValues = offset;
    OOS_ASSERT(packing.size() >= packingValues(panel.columns, rowsBelow, contributionBlockColumns));
    state.packing = packing;
    state.eliminated.assign(state.panelBlocks(), NOT_RUN);
    state.hasFailed.assign(state.panelBlocks(), 0);
}

TaskedFront::~TaskedFront() = default;

void TaskedFront::addTasks(base::TaskFlow& flow, std::span<const TaskId> panelAfter,
                           std::span<const TaskId> trailingAfter, std::int64_t priorityBase)
{
    State& state = *m_state;
    const std::size_t blocks = state.panelBlocks();
    const std::size_t trailing = state.trailingBlocks();
    const std::size_t rowBlocks = blocks + trailing;
    OOS_ASSERT(blocks > 0 && (trailingAfter.empty() || trailingAfter.size() == trailing));
    const auto panelPriority = [&](std::size_t k, std::int64_t kind)
    { return priorityBase + PANEL_PRIORITY - static_cast<std::int64_t>(k) * 8 - kind; };
    const auto always = [] { return true; };

    // The panel phase. U[k] holds the tiles of block k by (m, j): m in [j, rowBlocks), j in (k, blocks).
    std::vector<TaskId> factor(blocks);
    std::vector<std::vector<TaskId>> solve(blocks, std::vector<TaskId>(rowBlocks, 0));
    std::vector<std::vector<TaskId>> tiles(blocks, std::vector<TaskId>(rowBlocks * blocks, 0));
    std::vector<TaskId> tilesDone(blocks, 0);
    const auto tileOf = [&](std::size_t k, std::size_t m, std::size_t j) -> TaskId&
    { return tiles[k][m * blocks + j]; };
    for (std::size_t k = 0; k < blocks; ++k)
    {
        std::vector<TaskId> before;
        if (k == 0)
            before.assign(panelAfter.begin(), panelAfter.end());
        else
            before.push_back(tileOf(k - 1, k, k));
        if (k >= 2)
            before.push_back(tilesDone[k - 2]); // the slot of block k − 2 (its packs and its L·D) free
        factor[k] = flow.add(
            [&state, k]
            {
                state.factorDiagonal(k);
                return true;
            },
            before, panelPriority(k, 0));
        for (std::size_t m = k + 1; m < rowBlocks; ++m)
        {
            std::vector<TaskId> rowsBefore{factor[k]};
            if (k > 0)
                rowsBefore.push_back(tileOf(k - 1, m, k));
            solve[k][m] = flow.add(
                [&state, k, m]
                {
                    state.solveRows(k, m);
                    return true;
                },
                rowsBefore, panelPriority(k, 1));
        }
        if (k + 1 == blocks)
            break;
        std::vector<TaskId> all;
        for (std::size_t j = k + 1; j < blocks; ++j)
            for (std::size_t m = j; m < rowBlocks; ++m)
            {
                std::vector<TaskId> tileBefore{solve[k][m]};
                if (m != j)
                    tileBefore.push_back(solve[k][j]);
                if (k > 0)
                    tileBefore.push_back(tileOf(k - 1, m, j));
                tileOf(k, m, j) = flow.add(
                    [&state, k, m, j]
                    {
                        state.updateTile(k, m, j, false);
                        return true;
                    },
                    tileBefore, panelPriority(k, j == k + 1 ? 2 : 3));
                all.push_back(tileOf(k, m, j));
            }
        tilesDone[k] = flow.add(always, all, panelPriority(k, 4));
    }
    const std::size_t last = blocks - 1;
    std::vector<TaskId> panelParts{factor[last]};
    for (std::size_t m = blocks; m < rowBlocks; ++m)
        panelParts.push_back(solve[last][m]);
    m_panelDone = flow.add(always, panelParts, panelPriority(last, 5));

    // The trailing phase: block after block of terms, once the panel is eliminated (one block: each row block's packs
    // as soon as its rows are solved).
    m_trailingDone.assign(trailing, 0);
    std::vector<TaskId> previous(trailing * trailing, 0); // V(k − 1, m, j) by (m, j)
    std::vector<TaskId> previousDone;                     // the joins of the blocks of terms, by k
    for (std::size_t k = 0; k < blocks && trailing > 0; ++k)
    {
        std::vector<TaskId> pack(trailing);
        for (std::size_t m = 0; m < trailing; ++m)
        {
            std::vector<TaskId> packBefore{blocks == 1 ? solve[0][blocks + m] : m_panelDone};
            if (k >= 2)
                packBefore.push_back(previousDone[k - 2]);
            pack[m] = flow.add(
                [&state, k, m, blocks]
                {
                    if (!state.hasFailedBy(blocks - 1))
                    {
                        state.packRows(k, blocks + m);
                        state.packColumns(k, blocks + m);
                    }
                    return true;
                },
                packBefore, panelPriority(k, 6));
        }
        // A task: the tiles of row block m in a group of TRAILING_GROUP_BLOCKS column blocks (its packed rows read once
        // from memory for the group, then from the core's cache).
        std::vector<TaskId> all;
        for (std::size_t g = 0; g * TRAILING_GROUP_BLOCKS < trailing; ++g)
        {
            const std::size_t firstBlock = g * TRAILING_GROUP_BLOCKS;
            for (std::size_t m = firstBlock; m < trailing; ++m)
            {
                const std::size_t endBlock = std::min(firstBlock + TRAILING_GROUP_BLOCKS, m + 1);
                std::vector<TaskId> tileBefore{pack[m]};
                for (std::size_t j = firstBlock; j < endBlock; ++j)
                {
                    if (j != m)
                        tileBefore.push_back(pack[j]);
                    if (k == 0 && !trailingAfter.empty())
                        tileBefore.push_back(trailingAfter[j]);
                }
                if (k > 0)
                    tileBefore.push_back(previous[m * trailing + g]);
                const std::int64_t priority = priorityBase -
                                              static_cast<std::int64_t>(k * (trailing + 1) + firstBlock) * 2 -
                                              (m < firstBlock + TRAILING_GROUP_BLOCKS ? 0 : 1);
                previous[m * trailing + g] = flow.add(
                    [&state, k, m, firstBlock, endBlock, blocks]
                    {
                        for (std::size_t j = firstBlock; j < endBlock; ++j)
                            state.updateTile(k, blocks + m, j, true);
                        return true;
                    },
                    tileBefore, priority);
                all.push_back(previous[m * trailing + g]);
            }
        }
        previousDone.push_back(flow.add(always, all, panelPriority(k, 7)));
    }
    std::vector<TaskId> doneParts{m_panelDone};
    for (std::size_t j = 0; j < trailing; ++j)
    {
        std::vector<TaskId> column;
        const std::size_t g = j / TRAILING_GROUP_BLOCKS;
        for (std::size_t m = j; m < trailing; ++m)
            column.push_back(previous[m * trailing + g]);
        m_trailingDone[j] = flow.add(always, column, priorityBase - static_cast<std::int64_t>(j));
        doneParts.push_back(m_trailingDone[j]);
    }
    m_done = flow.add(always, doneParts, priorityBase);
}

std::size_t TaskedFront::eliminatedCount() const noexcept
{
    const State& state = *m_state;
    std::size_t count = 0;
    for (std::size_t k = 0; k < state.panelBlocks(); ++k)
    {
        const std::size_t done = state.eliminated[k];
        if (done == NOT_RUN)
            return count;
        count += done;
        if (done < state.widthOf(k))
            return count;
    }
    return count;
}

std::size_t TaskedFront::run(base::ThreadPool* pool, const base::CancellationToken* cancellation)
{
    base::TaskFlow flow;
    addTasks(flow, {}, {}, 0);
    flow.run(pool, cancellation);
    return eliminatedCount();
}

} // namespace oos::linalg
