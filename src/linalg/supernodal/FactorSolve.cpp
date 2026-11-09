#include "oos/linalg/supernodal/FactorSolve.hpp"

#include "../dense/LdltArithmetic.hpp"
#include "SolveFlow.hpp"
#include "SolvePlan.hpp"
#include "SolveSweeps.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <algorithm>
#include <bit>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

using namespace sweeps;

// The forward sweep of a tile (with D⁻¹ when fused).
void forwardSweep(const SweepContext& context, const Tile& tile, BlockPool& blocks, std::vector<double>& accumulators)
{
    // The trailing rows of a partial factor are only ever added to: written without −0 first, as every row a unit
    // writes back, so that skipping a supernode changes no bit (the class comment).
    canonicalizeTrailingRows(context, tile);
    const std::vector<std::uint8_t> isActive = activeSupernodes(context, tile);
    const SolvePlan& plan = context.plan;
    const std::span<const SolvePlan::Unit> units = plan.units();
    if (!plan.isByTree())
    {
        BlockLease block(blocks);
        for (const SolvePlan::Unit& unit : units)
        {
            if (isLargeSingleton(context, tile, unit) && isActive[static_cast<std::size_t>(unit.firstSupernode)] != 0)
                forwardLarge(context, tile, unit, block.data());
            else
                forwardUnit(context, tile, unit, nullptr, isActive, block.data());
        }
        return;
    }
    // The subtrees in parallel, each into its accumulator; then the accumulators added in subtree order; then the top.
    const std::span<const SolvePlan::Subtree> subtrees = plan.subtrees();
    const std::span<const SupernodeRange> cut = context.factor.subtrees();
    const auto isSubtreeActive = [&](std::size_t t) { return isActive[static_cast<std::size_t>(cut[t].last)] != 0; };
    base::parallelFor(
        context.pool, subtrees.size(), 1,
        [&](std::size_t first, std::size_t end)
        {
            BlockLease block(blocks);
            for (std::size_t t = first; t < end; ++t)
            {
                if (!isSubtreeActive(t))
                    continue;
                const SolvePlan::Subtree& subtree = subtrees[t];
                double* accumulator =
                    accumulators.data() + static_cast<std::size_t>(subtree.firstOutsideRow) * tile.ldx;
                std::fill_n(accumulator, static_cast<std::size_t>(subtree.outsideRowCount) * tile.ldx, 0.0);
                for (std::int32_t u = subtree.firstUnit; u < subtree.endUnit; ++u)
                    forwardUnit(context, tile, units[static_cast<std::size_t>(u)], accumulator, isActive, block.data());
            }
        },
        context.cancellation);
    addAccumulators(context, tile, accumulators.data(), isActive);
    BlockLease block(blocks);
    for (const SolvePlan::Unit& unit : units)
    {
        if (unit.subtree != SolvePlan::NO_SUBTREE || isActive[static_cast<std::size_t>(unit.firstSupernode)] == 0)
            continue;
        checkCancellation(context.cancellation);
        if (isLargeSingleton(context, tile, unit))
            forwardLarge(context, tile, unit, block.data());
        else
            forwardUnit(context, tile, unit, nullptr, isActive, block.data());
    }
}

// The backward sweep of a tile.
void backwardSweep(const SweepContext& context, const Tile& tile, BlockPool& blocks)
{
    const SolvePlan& plan = context.plan;
    const std::span<const SolvePlan::Unit> units = plan.units();
    {
        BlockLease block(blocks);
        for (auto unit = units.rbegin(); unit != units.rend(); ++unit)
        {
            if (plan.isByTree() && unit->subtree != SolvePlan::NO_SUBTREE)
                continue;
            if (isLargeSingleton(context, tile, *unit))
                backwardLarge(context, tile, *unit, block.data());
            else
                backwardUnit(context, tile, *unit, block.data());
        }
    }
    if (!plan.isByTree())
        return;
    // The subtrees in parallel: each reads the rows of its ancestors, final by then, and writes its own.
    const std::span<const SolvePlan::Subtree> subtrees = plan.subtrees();
    base::parallelFor(
        context.pool, subtrees.size(), 1,
        [&](std::size_t first, std::size_t end)
        {
            BlockLease block(blocks);
            for (std::size_t t = first; t < end; ++t)
                for (std::int32_t u = subtrees[t].endUnit; u-- > subtrees[t].firstUnit;)
                    backwardUnit(context, tile, units[static_cast<std::size_t>(u)], block.data());
        },
        context.cancellation);
}

// D⁻¹ on the right-hand sides themselves (solveDiagonal alone), supernode by supernode.
void diagonalSweep(const SweepContext& context, const Tile& tile)
{
    base::parallelFor(
        context.pool, static_cast<std::size_t>(context.factor.supernodeCount()), PIECE_SUPERNODES,
        [&](std::size_t first, std::size_t end)
        {
            for (std::size_t s = first; s < end; ++s)
            {
                const FactorPanel panel = context.factor.panel(static_cast<std::int32_t>(s));
                const BlockDiagonalView d = panel.pivots();
                for (std::size_t j = 0; j < tile.width; ++j)
                {
                    double* column = columnOf(tile, j);
                    const auto at = [&](std::size_t p) -> double&
                    { return column[static_cast<std::size_t>(panel.rows[p])]; };
                    std::size_t p = 0;
                    while (p < d.size)
                    {
                        if (startsTwoByTwoBlock(d, p))
                        {
                            const TwoByTwoInverse inverse =
                                invertTwoByTwo(d.diagonal[p * d.diagonalStride], d.subdiagonal[p],
                                               d.diagonal[(p + 1) * d.diagonalStride]);
                            const double r0 = at(p);
                            const double r1 = at(p + 1);
                            at(p) = canonical(inverse.s * (inverse.d11 * r0 - r1));
                            at(p + 1) = canonical(inverse.s * (inverse.d22 * r1 - r0));
                            p += 2;
                            continue;
                        }
                        at(p) = canonical(at(p) / d.diagonal[p * d.diagonalStride]);
                        p += 1;
                    }
                }
            }
        },
        context.cancellation);
}

} // namespace

void FactorSolve::requireRows(const char* operation, const DenseMatrixView& rightHandSides) const
{
    if (rightHandSides.rows != static_cast<std::size_t>(m_factor.dimension()))
        throw std::invalid_argument(base::concatenate(
            {"FactorSolve::", operation, ": the right-hand sides have ", std::to_string(rightHandSides.rows),
             " rows, the factor ", std::to_string(m_factor.dimension()),
             " equations; pass them in the numbering of the factorised matrix"}));
    if (rightHandSides.leadingDimension < std::max<std::size_t>(rightHandSides.rows, 1))
        throw std::invalid_argument(base::concatenate({"FactorSolve::", operation, ": the leading dimension ",
                                                       std::to_string(rightHandSides.leadingDimension),
                                                       " is below the row count ", std::to_string(rightHandSides.rows),
                                                       "; give a leading dimension of at least the row count"}));
}

void FactorSolve::solve(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation) const
{
    requireRows("solve", rightHandSides);
    if (!m_factor.isComplete())
        throw std::invalid_argument(
            base::concatenate({"FactorSolve::solve: the factor eliminates ", std::to_string(m_factor.eliminatedCount()),
                               " of its ", std::to_string(m_factor.dimension()),
                               " equations (a partial factorisation); use PartialFactor to condense and recover"}));
    if (m_factor.isSingular())
        throw base::SingularityError(base::concatenate(
            {"FactorSolve::solve: the factor has ", std::to_string(m_factor.inertia().zero),
             " zero pivots: the matrix is singular (a shift at an eigenvalue, or a mechanism); move the shift or "
             "constrain the mechanism"}));
    runSweeps(rightHandSides, {Sweep::Forward, Sweep::Diagonal, Sweep::Backward}, cancellation);
}

void FactorSolve::eliminateForward(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation) const
{
    requireRows("eliminateForward", rightHandSides);
    runSweeps(rightHandSides, {Sweep::Forward}, cancellation);
}

void FactorSolve::solveDiagonal(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation) const
{
    requireRows("solveDiagonal", rightHandSides);
    runSweeps(rightHandSides, {Sweep::Diagonal}, cancellation);
}

void FactorSolve::substituteBackward(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation) const
{
    requireRows("substituteBackward", rightHandSides);
    runSweeps(rightHandSides, {Sweep::Backward}, cancellation);
}

namespace
{

// Whether the sweeps run as one task flow: by subtrees, on a pool of more than one thread.
bool isFlowFor(const SolvePlan& plan, const base::ThreadPool* pool)
{
    return plan.isByTree() && pool != nullptr && pool->threadCount() > 1;
}

// The rows of the scratch a tile in flight of the flow holds per column: its subtrees' accumulators and its top block
// (the top supernodes' eliminated rows and at most the trailing rows of a partial factor).
std::uint64_t flowRowsPerTile(const SupernodalFactor& factor, const SolvePlan& plan)
{
    auto topRows = static_cast<std::uint64_t>(factor.dimension() - factor.eliminatedCount());
    for (const SolvePlan::Unit& unit : plan.units())
        if (unit.subtree == SolvePlan::NO_SUBTREE)
            topRows += static_cast<std::uint64_t>(unit.eliminatedRowCount);
    return static_cast<std::uint64_t>(plan.outsideRowCount()) + topRows;
}

} // namespace

std::size_t FactorSolve::tileColumns(std::size_t columnCount) const noexcept
{
    const SolvePlan& plan = m_factor.solvePlan();
    const std::size_t lanes = m_kernels.rowLanes();
    const base::ThreadPool* pool = m_kernels.options().pool;
    // The tile width never changes a bit: it may follow the thread count. The scratch of the tiles in flight: their
    // accumulators (sweeps by subtrees; with the flow, their top blocks too) and a block per running task.
    const std::uint64_t blocksInFlight = pool != nullptr ? pool->threadCount() : 1;
    const std::uint64_t tilesInFlight = plan.isByTree() ? TILES_IN_FLIGHT : 1;
    const std::uint64_t tileRows =
        isFlowFor(plan, pool) ? flowRowsPerTile(m_factor, plan) : static_cast<std::uint64_t>(plan.outsideRowCount());
    const std::uint64_t rowsPerColumn =
        tilesInFlight * tileRows + blocksInFlight * static_cast<std::uint64_t>(plan.largestUnitRowCount());
    std::size_t width = std::min(columnCount, RHS_TILE_COLUMNS);
    while (width > lanes && rowsPerColumn * roundUp(width, lanes) * sizeof(double) > SOLVE_WORKSPACE_BYTES)
        width = roundUp(width / 2, lanes);
    return std::max<std::size_t>(width, 1);
}

void FactorSolve::runSweeps(DenseMatrixView rightHandSides, std::initializer_list<Sweep> sweeps,
                            const base::CancellationToken* cancellation) const
{
    const std::size_t columnCount = rightHandSides.columns;
    if (columnCount == 0 || m_factor.dimension() == 0)
        return;
    const auto has = [&](Sweep sweep) { return std::find(sweeps.begin(), sweeps.end(), sweep) != sweeps.end(); };
    const bool isForward = has(Sweep::Forward);
    const bool isDiagonal = has(Sweep::Diagonal);
    const bool isBackward = has(Sweep::Backward);
    const SolvePlan& plan = m_factor.solvePlan();
    const std::size_t width = tileColumns(columnCount);
    const std::size_t ldx = roundUp(width, m_kernels.rowLanes());
    const std::size_t tileCount = base::pieceCount(columnCount, width);
    base::ThreadPool* pool = m_kernels.options().pool;

    const auto runTile =
        [&](std::size_t t, base::ThreadPool* tilePool, BlockPool& blocks, std::vector<double>& accumulators)
    {
        const std::size_t first = t * width;
        const std::size_t columns = std::min(width, columnCount - first);
        const Tile tile{rightHandSides.block(0, first, rightHandSides.rows, columns), columns,
                        roundUp(columns, m_kernels.rowLanes())};
        const SweepContext context{m_factor, plan, m_kernels, tilePool, cancellation, isForward && isDiagonal};
        if (isForward)
            forwardSweep(context, tile, blocks, accumulators);
        if (isDiagonal && !isForward)
            diagonalSweep(context, tile);
        if (isBackward)
            backwardSweep(context, tile, blocks);
    };
    if (isFlowFor(plan, pool) && (isForward || isBackward) && (isForward || !isDiagonal))
    {
        // The tiles in flight within the workspace (each its accumulators and top block, a block per thread), at most
        // one per thread.
        const std::uint64_t threads = pool->threadCount();
        const std::uint64_t blockBytes =
            threads * static_cast<std::uint64_t>(plan.largestUnitRowCount()) * ldx * sizeof(double);
        const std::uint64_t tileBytes =
            std::max<std::uint64_t>(flowRowsPerTile(m_factor, plan) * ldx * sizeof(double), 1);
        const std::uint64_t room = SOLVE_WORKSPACE_BYTES > blockBytes ? SOLVE_WORKSPACE_BYTES - blockBytes : 0;
        const auto inFlight =
            static_cast<std::size_t>(std::clamp<std::uint64_t>(room / tileBytes, TILES_IN_FLIGHT, threads));
        sweeps::SolveFlow flow(m_factor, plan, m_kernels, *pool, cancellation);
        flow.run(rightHandSides, width, inFlight, isForward, isBackward, isForward && isDiagonal);
        return;
    }
    const std::size_t blockValues = static_cast<std::size_t>(plan.largestUnitRowCount()) * ldx;
    BlockPool blocks(blockValues);
    if (plan.isByTree())
    {
        // TILES_IN_FLIGHT tiles at once, each on every thread (its subtrees, the rows of its large supernodes).
        const std::size_t inFlight = std::min(tileCount, TILES_IN_FLIGHT);
        std::vector<std::vector<double>> accumulators(
            inFlight, std::vector<double>(static_cast<std::size_t>(plan.outsideRowCount()) * ldx));
        for (std::size_t first = 0; first < tileCount; first += inFlight)
            base::parallelFor(
                pool, std::min(inFlight, tileCount - first), 1,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t q = begin; q < end; ++q)
                        runTile(first + q, pool, blocks, accumulators[q]);
                },
                cancellation);
        return;
    }
    std::vector<double> none;
    if (tileCount == 1)
    {
        // One tile: the threads share the rows of its large supernodes.
        runTile(0, pool, blocks, none);
        return;
    }
    // Sweeps in supernode order (delayed pivots, or one subtree): the tiles are the parallel pieces, each on one
    // thread.
    base::parallelFor(
        pool, tileCount, 1,
        [&](std::size_t first, std::size_t end)
        {
            for (std::size_t t = first; t < end; ++t)
                runTile(t, nullptr, blocks, none);
        },
        cancellation);
}

} // namespace oos::linalg
