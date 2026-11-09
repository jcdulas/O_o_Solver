#include "SolveSweeps.hpp"

#include "../dense/LdltArithmetic.hpp"

#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <bit>
#include <span>
#include <vector>

namespace oos::linalg::sweeps
{

void checkCancellation(const base::CancellationToken* cancellation)
{
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

// The rows of a unit read from its accumulator: the last ones (the rows beyond a subtree are its largest equations,
// and they come last in a unit's block), from the returned index on.
std::size_t firstAccumulatorRow(std::span<const std::int32_t> accumulatorRows)
{
    std::size_t first = accumulatorRows.size();
    while (first > 0 && accumulatorRows[first - 1] != SolvePlan::IN_RIGHT_HAND_SIDES)
        --first;
    return first;
}

// Rows [first, end) of a unit's block from the right-hand sides (and from the accumulator from `accumulatorFrom` on),
// the padding columns zero.
void loadRows(const Tile& tile, std::span<const std::int32_t> equations, std::span<const std::int32_t> accumulatorRows,
              const double* accumulator, std::size_t accumulatorFrom, double* block, std::size_t first, std::size_t end)
{
    const std::size_t inB = std::min(end, accumulatorFrom);
    for (std::size_t k = std::max(first, accumulatorFrom); k < end; ++k)
        std::copy_n(accumulator + static_cast<std::size_t>(accumulatorRows[k]) * tile.ldx, tile.ldx,
                    block + k * tile.ldx);
    if (tile.ldx > tile.width)
        for (std::size_t k = first; k < inB; ++k)
            std::fill(block + k * tile.ldx + tile.width, block + (k + 1) * tile.ldx, 0.0);
    std::size_t j = 0;
    for (; j + COPY_COLUMNS <= tile.width; j += COPY_COLUMNS)
    {
        const double* c0 = columnOf(tile, j);
        const double* c1 = columnOf(tile, j + 1);
        const double* c2 = columnOf(tile, j + 2);
        const double* c3 = columnOf(tile, j + 3);
        for (std::size_t k = first; k < inB; ++k)
        {
            const auto e = static_cast<std::size_t>(equations[k]);
            double* row = block + k * tile.ldx + j;
            row[0] = c0[e];
            row[1] = c1[e];
            row[2] = c2[e];
            row[3] = c3[e];
        }
    }
    for (; j < tile.width; ++j)
    {
        const double* column = columnOf(tile, j);
        for (std::size_t k = first; k < inB; ++k)
            block[k * tile.ldx + j] = column[static_cast<std::size_t>(equations[k])];
    }
}

// Rows [first, end) of a unit's block back to the right-hand sides and the accumulator, without −0.
void storeRows(const Tile& tile, std::span<const std::int32_t> equations, std::span<const std::int32_t> accumulatorRows,
               double* accumulator, std::size_t accumulatorFrom, const double* block, std::size_t first,
               std::size_t end)
{
    const std::size_t inB = std::min(end, accumulatorFrom);
    for (std::size_t k = std::max(first, accumulatorFrom); k < end; ++k)
    {
        const double* row = block + k * tile.ldx;
        double* target = accumulator + static_cast<std::size_t>(accumulatorRows[k]) * tile.ldx;
        for (std::size_t c = 0; c < tile.ldx; ++c)
            target[c] = canonical(row[c]);
    }
    std::size_t j = 0;
    for (; j + COPY_COLUMNS <= tile.width; j += COPY_COLUMNS)
    {
        double* c0 = columnOf(tile, j);
        double* c1 = columnOf(tile, j + 1);
        double* c2 = columnOf(tile, j + 2);
        double* c3 = columnOf(tile, j + 3);
        for (std::size_t k = first; k < inB; ++k)
        {
            const auto e = static_cast<std::size_t>(equations[k]);
            const double* row = block + k * tile.ldx + j;
            c0[e] = canonical(row[0]);
            c1[e] = canonical(row[1]);
            c2[e] = canonical(row[2]);
            c3[e] = canonical(row[3]);
        }
    }
    for (; j < tile.width; ++j)
    {
        double* column = columnOf(tile, j);
        for (std::size_t k = first; k < inB; ++k)
            column[static_cast<std::size_t>(equations[k])] = canonical(block[k * tile.ldx + j]);
    }
}

// y ← D⁻¹·y on the eliminated rows of a panel in a block, columns [firstColumn, endColumn): the expressions of
// DenseKernels::solveBlockDiagonal (each operation rounded; linalg is compiled with -ffp-contract=off).
void applyDiagonal(const FactorPanel& panel, std::span<const std::int32_t> localRows, double* block, std::size_t ldx,
                   std::size_t firstColumn, std::size_t endColumn)
{
    applyDiagonal(panel, localRows, block, ldx, firstColumn, endColumn, 0, panel.pivots().size);
}

void applyDiagonal(const FactorPanel& panel, std::span<const std::int32_t> localRows, double* block, std::size_t ldx,
                   std::size_t firstColumn, std::size_t endColumn, std::size_t firstPivot, std::size_t endPivot)
{
    const BlockDiagonalView d = panel.pivots();
    std::size_t p = firstPivot;
    while (p < endPivot)
    {
        double* first = block + static_cast<std::size_t>(localRows[p]) * ldx;
        if (startsTwoByTwoBlock(d, p))
        {
            const TwoByTwoInverse inverse = invertTwoByTwo(d.diagonal[p * d.diagonalStride], d.subdiagonal[p],
                                                           d.diagonal[(p + 1) * d.diagonalStride]);
            double* second = block + static_cast<std::size_t>(localRows[p + 1]) * ldx;
            for (std::size_t c = firstColumn; c < endColumn; ++c)
            {
                const double r0 = first[c];
                const double r1 = second[c];
                first[c] = inverse.s * (inverse.d11 * r0 - r1);
                second[c] = inverse.s * (inverse.d22 * r1 - r0);
            }
            p += 2;
            continue;
        }
        const double pivot = d.diagonal[p * d.diagonalStride];
        for (std::size_t c = firstColumn; c < endColumn; ++c)
            first[c] = first[c] / pivot;
        p += 1;
    }
}

namespace
{

// The operands of the products of a supernode on its unit's block.
RowProductOperands rowProducts(const Tile& tile, double* block)
{
    RowProductOperands operands;
    operands.x = block;
    operands.ldx = tile.ldx;
    operands.width = tile.ldx;
    return operands;
}

RowTriangleOperands rowTriangle(const Tile& tile, double* block, const std::int32_t* rows, std::size_t size,
                                const FactorPanel& panel, std::size_t first, bool isTransposed)
{
    RowTriangleOperands operands;
    operands.x = block;
    operands.ldx = tile.ldx;
    operands.width = tile.ldx;
    operands.rows = rows;
    operands.size = size;
    operands.l = panel.values.pointer(first, first);
    operands.ldl = panel.values.leadingDimension;
    operands.isTransposed = isTransposed;
    return operands;
}

// x(R) ← x(R) − L_RF·x(F) for the targets [first, end) of the rows below the eliminated ones.
void subtractBelow(const SweepContext& context, const Tile& tile, double* block, const FactorPanel& panel,
                   std::span<const std::int32_t> local, std::size_t first, std::size_t end)
{
    const std::size_t k = panel.eliminatedCount();
    RowProductOperands operands = rowProducts(tile, block);
    operands.targets = local.data() + k + first;
    operands.targetCount = end - first;
    operands.sources = local.data();
    operands.termCount = k;
    operands.l = panel.values.pointer(k + first, 0);
    operands.targetStride = 1;
    operands.termStride = static_cast<std::ptrdiff_t>(panel.values.leadingDimension);
    context.kernels.subtractRowProducts(operands);
}

// x(F) ← x(F) − L_RFᵀ·x(R) for the eliminated targets [first, end), the terms in panel order.
void subtractFromBelow(const SweepContext& context, const Tile& tile, double* block, const FactorPanel& panel,
                       std::span<const std::int32_t> local, std::size_t first, std::size_t end)
{
    const std::size_t k = panel.eliminatedCount();
    RowProductOperands operands = rowProducts(tile, block);
    operands.targets = local.data() + first;
    operands.targetCount = end - first;
    operands.sources = local.data() + k;
    operands.termCount = panel.rows.size() - k;
    operands.l = panel.values.pointer(k, first);
    operands.targetStride = static_cast<std::ptrdiff_t>(panel.values.leadingDimension);
    operands.termStride = 1;
    context.kernels.subtractRowProducts(operands);
}

// The backward step of supernode s on a block whose row local[p] holds the panel's row p.
void backwardSupernodeOn(const SweepContext& context, const Tile& tile, double* block, const FactorPanel& panel,
                         std::span<const std::int32_t> local)
{
    const std::size_t k = panel.eliminatedCount();
    if (panel.rows.size() > k)
        subtractFromBelow(context, tile, block, panel, local, 0, k);
    context.kernels.solveRowTriangle(rowTriangle(tile, block, local.data(), k, panel, 0, true));
}

// The backward step of a large supernode on the threads: `equations` the rows of its block (their right-hand sides
// gathered and the eliminated ones written back), local[p] the block row of the panel's row p.
void backwardLargeOn(const SweepContext& context, const Tile& tile, const FactorPanel& panel,
                     std::span<const std::int32_t> equations, std::span<const std::int32_t> local, double* block)
{
    const std::size_t k = panel.eliminatedCount();
    base::parallelFor(
        context.pool, equations.size(), PIECE_COPY_ROWS, [&](std::size_t first, std::size_t end)
        { loadRows(tile, equations, {}, nullptr, equations.size(), block, first, end); }, context.cancellation);
    if (panel.rows.size() > k)
        base::parallelFor(
            context.pool, k, PIECE_TARGET_ROWS, [&](std::size_t first, std::size_t end)
            { subtractFromBelow(context, tile, block, panel, local, first, end); }, context.cancellation);
    // The transposed triangle by diagonal blocks from the last: each block receives the terms of the rows below it in
    // decreasing order (pieces of rows), then solves its own triangle (pieces of columns).
    std::size_t end = k;
    while (end > 0)
    {
        const std::size_t first = (end - 1) / TRIANGLE_BLOCK_ROWS * TRIANGLE_BLOCK_ROWS;
        const std::size_t size = end - first;
        if (end < k)
            base::parallelFor(
                context.pool, size, PIECE_TARGET_ROWS,
                [&](std::size_t begin, std::size_t finish)
                {
                    RowProductOperands operands = rowProducts(tile, block);
                    operands.targets = local.data() + first + begin;
                    operands.targetCount = finish - begin;
                    // Term p is the row k − 1 − p, with l(i, p) = L(k − 1 − p, first + begin + i).
                    operands.sources = local.data() + (k - 1);
                    operands.sourceStep = -1;
                    operands.termCount = k - end;
                    operands.l = panel.values.pointer(k - 1, first + begin);
                    operands.targetStride = static_cast<std::ptrdiff_t>(panel.values.leadingDimension);
                    operands.termStride = -1;
                    context.kernels.subtractRowProducts(operands);
                },
                context.cancellation);
        base::parallelFor(
            context.pool, tile.ldx, PIECE_COLUMNS,
            [&](std::size_t begin, std::size_t finish)
            {
                RowTriangleOperands operands =
                    rowTriangle(tile, block + begin, local.data() + first, size, panel, first, true);
                operands.width = finish - begin;
                context.kernels.solveRowTriangle(operands);
            },
            context.cancellation);
        end = first;
    }
    base::parallelFor(
        context.pool, k, PIECE_COPY_ROWS, [&](std::size_t first, std::size_t finish)
        { storeRows(tile, equations, {}, nullptr, equations.size(), block, first, finish); }, context.cancellation);
}

} // namespace

// The forward step of a supernode on its unit's block (the class comment of FactorSolve).
void forwardSupernode(const SweepContext& context, const Tile& tile, double* block, std::int32_t s)
{
    const FactorPanel panel = context.factor.panel(s);
    const std::size_t k = panel.eliminatedCount();
    if (k == 0)
        return;
    const std::span<const std::int32_t> local = context.plan.localRows(s);
    context.kernels.solveRowTriangle(rowTriangle(tile, block, local.data(), k, panel, 0, false));
    if (panel.rows.size() > k)
        subtractBelow(context, tile, block, panel, local, 0, panel.rows.size() - k);
    if (context.isDiagonalFused)
        applyDiagonal(panel, local, block, tile.ldx, 0, tile.width);
}

// The backward step of a supernode on its unit's block.
void backwardSupernode(const SweepContext& context, const Tile& tile, double* block, std::int32_t s)
{
    const FactorPanel panel = context.factor.panel(s);
    const std::size_t k = panel.eliminatedCount();
    if (k == 0)
        return;
    backwardSupernodeOn(context, tile, block, panel, context.plan.localRows(s));
}

bool isLargeSingleton(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit)
{
    if (context.pool == nullptr || unit.firstSupernode != unit.lastSupernode)
        return false;
    const FactorPanel panel = context.factor.panel(unit.firstSupernode);
    const auto k = static_cast<std::uint64_t>(panel.eliminatedCount());
    const auto r = static_cast<std::uint64_t>(panel.rows.size());
    return (k * (k + 1) / 2 + (r - k) * k) * tile.ldx >= LARGE_SUPERNODE_MULTIPLY_ADDS;
}

// The forward sweep of a unit on one thread: its rows gathered, its supernodes that are active swept in order, its rows
// written back. Nothing when none of its supernodes is active.
void forwardUnit(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* accumulator,
                 std::span<const std::uint8_t> isActive, double* block)
{
    bool hasActive = false;
    for (std::int32_t s = unit.firstSupernode; s <= unit.lastSupernode && !hasActive; ++s)
        hasActive = isActive[static_cast<std::size_t>(s)] != 0;
    if (!hasActive)
        return;
    const std::span<const std::int32_t> equations = context.plan.rows(unit);
    const std::span<const std::int32_t> accumulatorRows = context.plan.accumulatorRows(unit);
    const std::size_t accumulatorFrom =
        accumulator != nullptr ? firstAccumulatorRow(accumulatorRows) : equations.size();
    loadRows(tile, equations, accumulatorRows, accumulator, accumulatorFrom, block, 0, equations.size());
    for (std::int32_t s = unit.firstSupernode; s <= unit.lastSupernode; ++s)
    {
        if (isActive[static_cast<std::size_t>(s)] == 0)
            continue;
        checkCancellation(context.cancellation);
        forwardSupernode(context, tile, block, s);
    }
    storeRows(tile, equations, accumulatorRows, accumulator, accumulatorFrom, block, 0, equations.size());
}

// The backward sweep of a unit on one thread: the rows it eliminates are written back.
void backwardUnit(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* block)
{
    const std::span<const std::int32_t> equations = context.plan.rows(unit);
    loadRows(tile, equations, {}, nullptr, equations.size(), block, 0, equations.size());
    for (std::int32_t s = unit.lastSupernode; s >= unit.firstSupernode; --s)
    {
        checkCancellation(context.cancellation);
        backwardSupernode(context, tile, block, s);
    }
    storeRows(tile, equations, {}, nullptr, equations.size(), block, 0,
              static_cast<std::size_t>(unit.eliminatedRowCount));
}

// The forward sweep of a large supernode alone, its rows shared between the threads.
void forwardLarge(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* block)
{
    const std::int32_t s = unit.firstSupernode;
    const FactorPanel panel = context.factor.panel(s);
    const std::size_t k = panel.eliminatedCount();
    if (k == 0)
        return;
    const std::span<const std::int32_t> local = context.plan.localRows(s);
    const std::span<const std::int32_t> equations = context.plan.rows(unit);
    base::parallelFor(
        context.pool, equations.size(), PIECE_COPY_ROWS, [&](std::size_t first, std::size_t end)
        { loadRows(tile, equations, {}, nullptr, equations.size(), block, first, end); }, context.cancellation);
    // The triangle by diagonal blocks: each block receives the terms of the rows above it (pieces of rows), then
    // solves its own triangle (pieces of columns); every value keeps its chain.
    for (std::size_t first = 0; first < k; first += TRIANGLE_BLOCK_ROWS)
    {
        const std::size_t size = std::min(TRIANGLE_BLOCK_ROWS, k - first);
        if (first > 0)
            base::parallelFor(
                context.pool, size, PIECE_TARGET_ROWS,
                [&](std::size_t begin, std::size_t end)
                {
                    RowProductOperands operands = rowProducts(tile, block);
                    operands.targets = local.data() + first + begin;
                    operands.targetCount = end - begin;
                    operands.sources = local.data();
                    operands.termCount = first;
                    operands.l = panel.values.pointer(first + begin, 0);
                    operands.targetStride = 1;
                    operands.termStride = static_cast<std::ptrdiff_t>(panel.values.leadingDimension);
                    context.kernels.subtractRowProducts(operands);
                },
                context.cancellation);
        base::parallelFor(
            context.pool, tile.ldx, PIECE_COLUMNS,
            [&](std::size_t begin, std::size_t end)
            {
                RowTriangleOperands operands =
                    rowTriangle(tile, block + begin, local.data() + first, size, panel, first, false);
                operands.width = end - begin;
                context.kernels.solveRowTriangle(operands);
            },
            context.cancellation);
    }
    if (panel.rows.size() > k)
        base::parallelFor(
            context.pool, panel.rows.size() - k, PIECE_TARGET_ROWS, [&](std::size_t first, std::size_t end)
            { subtractBelow(context, tile, block, panel, local, first, end); }, context.cancellation);
    if (context.isDiagonalFused)
        base::parallelFor(
            context.pool, tile.width, PIECE_COLUMNS, [&](std::size_t first, std::size_t end)
            { applyDiagonal(panel, local, block, tile.ldx, first, end); }, context.cancellation);
    base::parallelFor(
        context.pool, equations.size(), PIECE_COPY_ROWS, [&](std::size_t first, std::size_t end)
        { storeRows(tile, equations, {}, nullptr, equations.size(), block, first, end); }, context.cancellation);
}

// The backward sweep of a large supernode alone, its rows shared between the threads.
void backwardLarge(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* block)
{
    const std::int32_t s = unit.firstSupernode;
    const FactorPanel panel = context.factor.panel(s);
    if (panel.eliminatedCount() == 0)
        return;
    backwardLargeOn(context, tile, panel, context.plan.rows(unit), context.plan.localRows(s), block);
}

// The supernodes the forward sweep of a tile visits: those whose eliminated rows hold a value other than +0 in a
// column of the tile, and their ancestors (the closure along the parents: the rows a supernode updates are rows of its
// ancestors). The test is on the bits, so that a −0 counts as a value: a skipped supernode then only ever sees +0.
std::vector<std::uint8_t> activeSupernodes(const SweepContext& context, const Tile& tile)
{
    const auto count = static_cast<std::size_t>(context.factor.supernodeCount());
    std::vector<std::uint8_t> isActive(count, 0); // one per supernode
    base::parallelFor(
        context.pool, count, PIECE_SUPERNODES,
        [&](std::size_t first, std::size_t end)
        {
            for (std::size_t s = first; s < end; ++s)
            {
                const FactorPanel panel = context.factor.panel(static_cast<std::int32_t>(s));
                const std::span<const std::int32_t> rows = panel.rows.first(panel.eliminatedCount());
                for (std::size_t j = 0; j < tile.width && isActive[s] == 0; ++j)
                {
                    const double* column = columnOf(tile, j);
                    std::uint64_t bits = 0;
                    for (const std::int32_t row : rows)
                        bits |= std::bit_cast<std::uint64_t>(column[static_cast<std::size_t>(row)]);
                    isActive[s] = bits != 0 ? 1 : 0;
                }
            }
        },
        context.cancellation);
    for (std::size_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = context.plan.parent(static_cast<std::int32_t>(s));
        if (isActive[s] != 0 && parent != SolvePlan::NO_PARENT)
            isActive[static_cast<std::size_t>(parent)] = 1;
    }
    return isActive;
}

void canonicalizeTrailingRows(const SweepContext& context, const Tile& tile)
{
    const auto dimension = static_cast<std::size_t>(context.factor.dimension());
    for (std::size_t j = 0; j < tile.width; ++j)
    {
        double* column = columnOf(tile, j);
        for (auto e = static_cast<std::size_t>(context.factor.eliminatedCount()); e < dimension; ++e)
            column[e] = canonical(column[e]);
    }
}

void addAccumulators(const SweepContext& context, const Tile& tile, const double* accumulators,
                     std::span<const std::uint8_t> isActive)
{
    const SolvePlan& plan = context.plan;
    const std::span<const SolvePlan::Subtree> subtrees = plan.subtrees();
    const std::span<const SupernodeRange> cut = context.factor.subtrees();
    base::parallelFor(
        context.pool, tile.width, PIECE_ADD_COLUMNS,
        [&](std::size_t first, std::size_t end)
        {
            for (std::size_t t = 0; t < subtrees.size(); ++t)
            {
                if (isActive[static_cast<std::size_t>(cut[t].last)] == 0)
                    continue;
                const std::span<const std::int32_t> rows = plan.outsideRows(subtrees[t]);
                const double* accumulator =
                    accumulators + static_cast<std::size_t>(subtrees[t].firstOutsideRow) * tile.ldx;
                for (std::size_t j = first; j < end; ++j)
                {
                    double* column = columnOf(tile, j);
                    for (std::size_t q = 0; q < rows.size(); ++q)
                        column[static_cast<std::size_t>(rows[q])] += accumulator[q * tile.ldx + j];
                }
            }
        },
        context.cancellation);
}

void backwardAlone(const SweepContext& context, const Tile& tile, std::int32_t s, double* block)
{
    const FactorPanel panel = context.factor.panel(s);
    const std::size_t k = panel.eliminatedCount();
    if (k == 0)
        return;
    // Block row p is panel row p: the panel's rows are its eliminated equations, then its rows below in increasing
    // order (a unit of this supernode alone has them in that order too).
    const std::size_t rowCount = panel.rows.size();
    std::vector<std::int32_t> local(rowCount); // bounded by a front
    for (std::size_t p = 0; p < rowCount; ++p)
        local[p] = static_cast<std::int32_t>(p);
    const auto r = static_cast<std::uint64_t>(rowCount);
    const auto work = (static_cast<std::uint64_t>(k) * (k + 1) / 2 + (r - k) * k) * tile.ldx;
    if (context.pool != nullptr && work >= LARGE_SUPERNODE_MULTIPLY_ADDS)
    {
        backwardLargeOn(context, tile, panel, panel.rows, local, block);
        return;
    }
    loadRows(tile, panel.rows, {}, nullptr, rowCount, block, 0, rowCount);
    backwardSupernodeOn(context, tile, block, panel, local);
    storeRows(tile, panel.rows, {}, nullptr, rowCount, block, 0, k);
}

} // namespace oos::linalg::sweeps
