#include "oos/linalg/supernodal/OutOfCoreFactor.hpp"

#include "MultifrontalDriver.hpp"
#include "SolvePlan.hpp"
#include "SolveSweeps.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{

using namespace sweeps;

namespace
{

FactorOptions indefiniteOf(const FactorOptions& options)
{
    FactorOptions indefinite = options;
    indefinite.pivoting = PivotingMode::Indefinite;
    indefinite.storage = nullptr;
    indefinite.keptSolvePlan = nullptr;
    indefinite.keepsSolvePlan = false;
    return indefinite;
}

} // namespace

OutOfCoreFactor OutOfCoreFactor::factorise(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                           const DenseKernels& kernels, const FactorOptions& options,
                                           const ScratchSpool::Options& spoolOptions, base::MemoryBudget& budget)
{
    OutOfCoreFactor result;
    result.m_budget = &budget;
    result.m_spool = std::make_unique<ScratchSpool>(spoolOptions, budget);
    result.m_records.resize(static_cast<std::size_t>(symbolic.supernodeCount()));
    const FactorOptions indefinite = indefiniteOf(options);
    MultifrontalDriver driver(symbolic, matrix, kernels, indefinite, false, budget);
    driver.writePanelsTo(*result.m_spool, result.m_records);
    driver.run(result.m_factor, nullptr);
    result.m_traffic.writtenBytes = result.m_spool->writtenBytes();
    return result;
}

OutOfCoreFactor OutOfCoreFactor::factorise(const SymbolicFactor& symbolic, FrontSource& source,
                                           const DenseKernels& kernels, const FactorOptions& options,
                                           const ScratchSpool::Options& spoolOptions, base::MemoryBudget& budget)
{
    OutOfCoreFactor result;
    result.m_budget = &budget;
    result.m_spool = std::make_unique<ScratchSpool>(spoolOptions, budget);
    result.m_records.resize(static_cast<std::size_t>(symbolic.supernodeCount()));
    const FactorOptions indefinite = indefiniteOf(options);
    MultifrontalDriver driver(symbolic, source, kernels, indefinite, budget);
    driver.writePanelsTo(*result.m_spool, result.m_records);
    driver.run(result.m_factor, nullptr);
    result.m_traffic.writtenBytes = result.m_spool->writtenBytes();
    return result;
}

OutOfCoreTraffic OutOfCoreFactor::traffic() const noexcept
{
    OutOfCoreTraffic traffic = m_traffic;
    if (m_spool)
        traffic.writtenBytes = m_spool->writtenBytes();
    return traffic;
}

void OutOfCoreFactor::load(std::int32_t first, std::int32_t last)
{
    base::Stopwatch watch;
    std::uint64_t total = 0;
    for (std::int32_t s = first; s <= last; ++s)
        total += m_records[static_cast<std::size_t>(s)].bytes;
    if (m_window.size() < total)
        m_window = SpoolBuffer(*m_budget, static_cast<std::size_t>(total), "factor out of core: resident panels");
    // The records of consecutive supernodes are mostly consecutive on the spool (one writer, its blocks): one read
    // for each run of them.
    std::uint64_t at = 0;
    std::int32_t s = first;
    while (s <= last)
    {
        const SpoolRecord& start = m_records[static_cast<std::size_t>(s)];
        std::uint64_t bytes = start.bytes;
        std::int32_t end = s + 1;
        while (end <= last && m_records[static_cast<std::size_t>(end)].offset == start.offset + bytes)
            bytes += m_records[static_cast<std::size_t>(end++)].bytes;
        if (bytes > 0)
        {
            m_spool->readSpan(start.offset, bytes, m_window.data() + at);
            ++m_traffic.readRequests;
        }
        for (std::int32_t k = s; k < end; ++k)
        {
            m_factor.m_panelPointer[static_cast<std::size_t>(k)] = reinterpret_cast<double*>(m_window.data() + at);
            at += m_records[static_cast<std::size_t>(k)].bytes;
        }
        s = end;
    }
    m_traffic.readBytes += total;
    m_traffic.readSeconds += watch.lap();
}

void OutOfCoreFactor::release(std::int32_t first, std::int32_t last) noexcept
{
    for (std::int32_t s = first; s <= last; ++s)
        m_factor.m_panelPointer[static_cast<std::size_t>(s)] = nullptr;
}

void OutOfCoreFactor::solve(DenseMatrixView rightHandSides, const DenseKernels& kernels,
                            const base::CancellationToken* cancellation)
{
    if (rightHandSides.rows != static_cast<std::size_t>(m_factor.dimension()))
        throw std::invalid_argument(base::concatenate(
            {"OutOfCoreFactor::solve: expected right-hand sides of ", std::to_string(m_factor.dimension()),
             " rows, found ", std::to_string(rightHandSides.rows), "; give a block of the factor's dimension"}));
    if (m_factor.isSingular())
        throw base::SingularityError(
            "the out-of-core factor is singular (a zero pivot): it cannot be solved with "
            "(no remedy here: the eigenvalue solver moves its shift off the root)");
    const std::size_t columnCount = rightHandSides.columns;
    if (columnCount == 0 || m_factor.dimension() == 0)
        return;
    const SolvePlan& plan = m_factor.solvePlan();
    // The tiles: any width gives the same bits (FactorSolve's tile comment); a tile's panels are read once a sweep.
    const std::size_t width = std::min(columnCount, FactorSolve::RHS_TILE_COLUMNS);
    base::ThreadPool* pool = kernels.options().pool;
    const std::span<const SolvePlan::Unit> units = plan.units();
    const std::size_t blockValues =
        static_cast<std::size_t>(plan.largestUnitRowCount()) * roundUp(width, kernels.rowLanes());
    BlockPool blocks(blockValues);
    std::vector<double> accumulators; // the subtrees' accumulators of a tile (bounded by the plan's outside rows)
    const auto loadUnit = [&](const SolvePlan::Unit& unit) { load(unit.firstSupernode, unit.lastSupernode); };
    const auto releaseUnit = [&](const SolvePlan::Unit& unit) { release(unit.firstSupernode, unit.lastSupernode); };
    for (std::size_t first = 0; first < columnCount; first += width)
    {
        const std::size_t columns = std::min(width, columnCount - first);
        const Tile tile{rightHandSides.block(0, first, rightHandSides.rows, columns), columns,
                        roundUp(columns, kernels.rowLanes())};
        const SweepContext context{m_factor, plan, kernels, pool, cancellation, true};
        BlockLease block(blocks);
        // Forward, D⁻¹ fused (FactorSolve's forwardSweep, its subtrees one after the other: each writes only its rows
        // and its accumulator, so their order changes no bit).
        canonicalizeTrailingRows(context, tile);
        const std::vector<std::uint8_t> isActive = activeSupernodes(context, tile);
        if (!plan.isByTree())
        {
            for (const SolvePlan::Unit& unit : units)
            {
                checkCancellation(cancellation);
                loadUnit(unit);
                if (isLargeSingleton(context, tile, unit) &&
                    isActive[static_cast<std::size_t>(unit.firstSupernode)] != 0)
                    forwardLarge(context, tile, unit, block.data());
                else
                    forwardUnit(context, tile, unit, nullptr, isActive, block.data());
                releaseUnit(unit);
            }
        }
        else
        {
            const std::span<const SolvePlan::Subtree> subtrees = plan.subtrees();
            const std::span<const SupernodeRange> cut = m_factor.subtrees();
            accumulators.assign(static_cast<std::size_t>(plan.outsideRowCount()) * tile.ldx, 0.0);
            for (std::size_t t = 0; t < subtrees.size(); ++t)
            {
                if (isActive[static_cast<std::size_t>(cut[t].last)] == 0)
                    continue;
                const SolvePlan::Subtree& subtree = subtrees[t];
                double* accumulator =
                    accumulators.data() + static_cast<std::size_t>(subtree.firstOutsideRow) * tile.ldx;
                std::fill_n(accumulator, static_cast<std::size_t>(subtree.outsideRowCount) * tile.ldx, 0.0);
                for (std::int32_t u = subtree.firstUnit; u < subtree.endUnit; ++u)
                {
                    checkCancellation(cancellation);
                    const SolvePlan::Unit& unit = units[static_cast<std::size_t>(u)];
                    loadUnit(unit);
                    forwardUnit(context, tile, unit, accumulator, isActive, block.data());
                    releaseUnit(unit);
                }
            }
            addAccumulators(context, tile, accumulators.data(), isActive);
            for (const SolvePlan::Unit& unit : units)
            {
                if (unit.subtree != SolvePlan::NO_SUBTREE ||
                    isActive[static_cast<std::size_t>(unit.firstSupernode)] == 0)
                    continue;
                checkCancellation(cancellation);
                loadUnit(unit);
                if (isLargeSingleton(context, tile, unit))
                    forwardLarge(context, tile, unit, block.data());
                else
                    forwardUnit(context, tile, unit, nullptr, isActive, block.data());
                releaseUnit(unit);
            }
        }
        // Backward (FactorSolve's backwardSweep: the top in reverse, then each subtree's units in reverse).
        for (auto unit = units.rbegin(); unit != units.rend(); ++unit)
        {
            if (plan.isByTree() && unit->subtree != SolvePlan::NO_SUBTREE)
                continue;
            checkCancellation(cancellation);
            loadUnit(*unit);
            if (isLargeSingleton(context, tile, *unit))
                backwardLarge(context, tile, *unit, block.data());
            else
                backwardUnit(context, tile, *unit, block.data());
            releaseUnit(*unit);
        }
        if (plan.isByTree())
        {
            const std::span<const SolvePlan::Subtree> subtrees = plan.subtrees();
            for (std::size_t t = 0; t < subtrees.size(); ++t)
                for (std::int32_t u = subtrees[t].endUnit; u-- > subtrees[t].firstUnit;)
                {
                    checkCancellation(cancellation);
                    const SolvePlan::Unit& unit = units[static_cast<std::size_t>(u)];
                    loadUnit(unit);
                    backwardUnit(context, tile, unit, block.data());
                    releaseUnit(unit);
                }
        }
    }
}

} // namespace oos::linalg
