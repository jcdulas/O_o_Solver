#include "RegionSweep.hpp"

#include "ForwardStream.hpp"
#include "SolveSweeps.hpp"
#include "WorkspacePool.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::linalg::regions
{

RegionSweepCost sweep(StreamedFactorisation& streamed, DenseMatrixView loadsToSolution,
                      ConstDenseMatrixView trailingValues, const DenseKernels& kernels, const FactorOptions& options,
                      base::MemoryBudget& budget, std::int64_t regionValues, DenseMatrix* schurComplement)
{
    const auto dimension = static_cast<std::size_t>(streamed.factor().dimension());
    const auto interior = static_cast<std::size_t>(streamed.factor().eliminatedCount());
    const std::size_t trailing = dimension - interior;
    const std::size_t columnCount = loadsToSolution.columns;
    const std::int32_t count = streamed.supernodeCount();
    base::BudgetedBuffer<std::uint8_t> isKept(budget, static_cast<std::size_t>(count), "regions: panels kept");
    const auto valuesOf = [&](std::int32_t s) { return WorkspacePool::classSize(streamed.panelValueCount(s)); };
    // The flops of the subtrees factorised again: prefix sums over the postorder (a subtree is a range of it).
    base::BudgetedBuffer<double> flopsBefore(budget, static_cast<std::size_t>(count) + 1, "regions: flops");
    for (std::int32_t s = 0; s < count; ++s)
        flopsBefore[static_cast<std::size_t>(s) + 1] =
            flopsBefore[static_cast<std::size_t>(s)] + streamed.supernodeFlops(s);

    // The forward sweep, on the columns with a nonzero eliminated row (by their bits, a −0 counting, as FactorSolve's
    // active supernodes test them): no supernode visits the others, which FactorSolve leaves as they are but their
    // trailing rows (written without −0, then replaced by u_b). The panels of the largest suffix of the supernodes (the
    // top of the tree) that fits regionValues are kept. When every column is swept, in place (no copy).
    std::vector<std::size_t> swept; // bounded by the columns
    for (std::size_t j = 0; j < columnCount; ++j)
    {
        std::uint64_t bits = 0;
        const double* column = &loadsToSolution(0, j);
        for (std::size_t e = 0; e < interior && bits == 0; ++e)
            bits |= std::bit_cast<std::uint64_t>(column[e]);
        if (bits != 0)
            swept.push_back(j);
    }
    RegionSweepCost cost;
    cost.regionValues = regionValues;
    const std::int32_t top = topFirst(count, regionValues, valuesOf);
    for (std::int32_t s = top; s < count; ++s)
    {
        isKept[static_cast<std::size_t>(s)] = 1;
        cost.topValues += valuesOf(s);
    }
    cost.keptValues = cost.topValues;
    if (swept.size() == columnCount)
    {
        ForwardStream forward(streamed, loadsToSolution, kernels, budget, options.cancellation);
        streamed.run(&forward, isKept.span(), schurComplement);
    }
    else
    {
        DenseMatrix work(budget, dimension, swept.size(), "regions: swept columns");
        for (std::size_t k = 0; k < swept.size(); ++k)
            std::copy_n(&loadsToSolution(0, swept[k]), dimension, &work(0, k));
        {
            ForwardStream forward(streamed, work.view(), kernels, budget, options.cancellation);
            streamed.run(&forward, isKept.span(), schurComplement);
        }
        for (std::size_t k = 0; k < swept.size(); ++k)
            std::copy_n(&work(0, k), dimension, &loadsToSolution(0, swept[k]));
    }
    // u_b in the trailing rows, then the diagonal (FactorSolve::solveDiagonal: x ← x / d without −0) and the backward
    // sweep.
    for (std::size_t j = 0; j < columnCount && trailing > 0; ++j)
        std::copy_n(trailingValues.pointer(0, j), trailing, &loadsToSolution(interior, j));
    const std::span<const double> pivots = streamed.pivots();
    for (std::size_t j = 0; j < columnCount; ++j)
    {
        double* column = &loadsToSolution(0, j);
        for (std::size_t e = 0; e < interior; ++e)
            column[e] = sweeps::canonical(column[e] / pivots[e]);
    }
    if (columnCount == 0 || count == 0)
    {
        streamed.releaseKept();
        return cost;
    }
    const SolvePlan& plan = streamed.plan();
    const sweeps::SweepContext context{streamed.factor(),    plan, kernels, kernels.options().pool,
                                       options.cancellation, false};
    std::vector<sweeps::Tile> tiles; // bounded by the columns over a tile's width
    for (std::size_t first = 0; first < columnCount; first += FactorSolve::RHS_TILE_COLUMNS)
    {
        const std::size_t columns = std::min(FactorSolve::RHS_TILE_COLUMNS, columnCount - first);
        tiles.push_back({loadsToSolution.block(0, first, dimension, columns), columns,
                         sweeps::roundUp(columns, kernels.rowLanes())});
    }
    sweeps::BlockPool blocks(static_cast<std::size_t>(plan.largestUnitRowCount()) * tiles.front().ldx);
    // The backward steps of supernodes [first, last] from the last: every ancestor of each is done before it.
    const auto substitute = [&](std::int32_t first, std::int32_t last)
    {
        for (const sweeps::Tile& tile : tiles)
        {
            sweeps::BlockLease block(blocks);
            for (std::int32_t s = last; s >= first; --s)
            {
                sweeps::checkCancellation(options.cancellation);
                sweeps::backwardAlone(context, tile, s, block.data());
            }
        }
    };
    substitute(top, count - 1);
    streamed.releaseKept();
    for (std::int32_t s = top; s < count; ++s)
        isKept[static_cast<std::size_t>(s)] = 0;

    // The regions of the rest (a forest: the supernodes before a suffix are complete subtrees), top-down: each subtree
    // factorised again with the largest suffix of its supernodes that fits regionValues (at least its root) kept,
    // swept, released.
    forEach(
        top, regionValues, valuesOf, [&](std::int32_t s) { return streamed.subtreeFirst(s); },
        [&](std::int32_t kept, std::int32_t root, std::int64_t values)
        {
            for (std::int32_t s = kept; s <= root; ++s)
                isKept[static_cast<std::size_t>(s)] = 1;
            streamed.runSubtreeOf(root, isKept.span());
            substitute(kept, root);
            streamed.releaseKept();
            for (std::int32_t s = kept; s <= root; ++s)
                isKept[static_cast<std::size_t>(s)] = 0;
            ++cost.regionRunCount;
            cost.keptValues = std::max(cost.keptValues, values);
            const std::int32_t first = streamed.subtreeFirst(root);
            cost.recomputedFlops +=
                flopsBefore[static_cast<std::size_t>(root) + 1] - flopsBefore[static_cast<std::size_t>(first)];
        });
    return cost;
}

} // namespace oos::linalg::regions
