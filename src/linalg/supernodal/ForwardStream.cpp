#include "ForwardStream.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>

namespace oos::linalg
{

std::uint64_t ForwardStream::accumulatorBytes(const SolvePlan& plan, std::size_t columns, std::size_t rowLanes) noexcept
{
    if (!plan.isByTree() || columns == 0)
        return 0;
    // Each tile pads its columns to the kernels' vector length.
    const std::size_t width = FactorSolve::RHS_TILE_COLUMNS;
    const std::size_t fullTiles = columns / width;
    const std::size_t rest = columns - fullTiles * width;
    const std::size_t padded =
        fullTiles * sweeps::roundUp(width, rowLanes) + (rest > 0 ? sweeps::roundUp(rest, rowLanes) : 0);
    return static_cast<std::uint64_t>(plan.outsideRowCount()) * padded * sizeof(double);
}

ForwardStream::ForwardStream(const StreamedFactorisation& factorisation, DenseMatrixView rightHandSides,
                             const DenseKernels& kernels, base::MemoryBudget& budget,
                             const base::CancellationToken* cancellation)
    : m_factorisation(factorisation),
      m_kernels(kernels),
      m_cancellation(cancellation),
      m_blocks(static_cast<std::size_t>(factorisation.plan().largestUnitRowCount()) *
               sweeps::roundUp(std::min(rightHandSides.columns, FactorSolve::RHS_TILE_COLUMNS), kernels.rowLanes()))
{
    OOS_ASSERT(rightHandSides.rows == static_cast<std::size_t>(factorisation.factor().dimension()));
    const SolvePlan& plan = factorisation.plan();
    const std::size_t columnCount = rightHandSides.columns;
    for (std::size_t first = 0; first < columnCount; first += FactorSolve::RHS_TILE_COLUMNS)
    {
        const std::size_t columns = std::min(FactorSolve::RHS_TILE_COLUMNS, columnCount - first);
        TileState state;
        state.tile = sweeps::Tile{rightHandSides.block(0, first, rightHandSides.rows, columns), columns,
                                  sweeps::roundUp(columns, kernels.rowLanes())};
        const sweeps::SweepContext context = contextOf();
        // FactorSolve's forward sweep starts so (forwardSweep): the trailing rows without −0, then the supernodes it
        // visits, from the right-hand sides as given.
        sweeps::canonicalizeTrailingRows(context, state.tile);
        state.isActive = sweeps::activeSupernodes(context, state.tile);
        if (plan.isByTree())
            state.accumulators =
                base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(plan.outsideRowCount()) * state.tile.ldx,
                                             "streamed forward sweep: accumulators of the subtrees");
        m_tiles.push_back(std::move(state));
    }
}

sweeps::SweepContext ForwardStream::contextOf() const noexcept
{
    return sweeps::SweepContext{m_factorisation.factor(), m_factorisation.plan(), m_kernels,
                                m_kernels.options().pool, m_cancellation,         false};
}

void ForwardStream::consumeUnit(std::size_t index)
{
    const SolvePlan& plan = m_factorisation.plan();
    const SolvePlan::Unit& unit = plan.units()[index];
    const sweeps::SweepContext context = contextOf();
    const std::span<const SupernodeRange> cut = m_factorisation.factor().subtrees();
    for (TileState& state : m_tiles)
    {
        const std::span<const std::uint8_t> isActive = state.isActive;
        if (!plan.isByTree())
        {
            sweeps::BlockLease block(m_blocks);
            if (sweeps::isLargeSingleton(context, state.tile, unit) &&
                isActive[static_cast<std::size_t>(unit.firstSupernode)] != 0)
                sweeps::forwardLarge(context, state.tile, unit, block.data());
            else
                sweeps::forwardUnit(context, state.tile, unit, nullptr, isActive, block.data());
            continue;
        }
        if (unit.subtree != SolvePlan::NO_SUBTREE)
        {
            // A unit of a subtree into the subtree's accumulator (zero from its allocation); an inactive subtree is
            // skipped whole, as FactorSolve skips it.
            const auto t = static_cast<std::size_t>(unit.subtree);
            if (isActive[static_cast<std::size_t>(cut[t].last)] == 0)
                continue;
            const SolvePlan::Subtree& subtree = plan.subtrees()[t];
            double* accumulator =
                state.accumulators.data() + static_cast<std::size_t>(subtree.firstOutsideRow) * state.tile.ldx;
            sweeps::BlockLease block(m_blocks);
            sweeps::forwardUnit(context, state.tile, unit, accumulator, isActive, block.data());
            continue;
        }
        if (isActive[static_cast<std::size_t>(unit.firstSupernode)] == 0)
            continue;
        sweeps::checkCancellation(m_cancellation);
        sweeps::BlockLease block(m_blocks);
        if (sweeps::isLargeSingleton(context, state.tile, unit))
            sweeps::forwardLarge(context, state.tile, unit, block.data());
        else
            sweeps::forwardUnit(context, state.tile, unit, nullptr, isActive, block.data());
    }
}

void ForwardStream::afterSubtrees()
{
    const sweeps::SweepContext context = contextOf();
    for (TileState& state : m_tiles)
        sweeps::addAccumulators(context, state.tile, state.accumulators.data(), state.isActive);
}

} // namespace oos::linalg
