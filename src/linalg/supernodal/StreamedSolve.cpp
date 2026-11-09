#include "oos/linalg/supernodal/StreamedSolve.hpp"

#include "FactorSchedule.hpp"
#include "FrontStream.hpp"
#include "RegionSweep.hpp"
#include "StreamedFactorisation.hpp"
#include "WorkspacePool.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

// Per supernode, what a solve holds beside the streamed factorisation's skeleton: its kept flag and the prefix sum of
// the flops of the regions (RegionSweep.cpp).
constexpr std::uint64_t REGION_SUPERNODE_BYTES = sizeof(std::uint8_t) + sizeof(double);

// The supernodes a solve factorises (the trailing one of a partial symbolic factor is left out, as
// StreamedFactorisation leaves it).
std::int32_t factorisedCount(const SymbolicFactor& symbolic) noexcept
{
    return symbolic.trailingDofCount() > 0 ? symbolic.supernodeCount() - 1 : symbolic.supernodeCount();
}

} // namespace

std::uint64_t StreamedSolveEstimate::bytes(std::size_t columnCount, std::int64_t regionValues) const noexcept
{
    const std::int64_t kept = std::min(std::max(regionValues, run.largestPanelValues), panelValues);
    return run.bytes(columnCount, kept) + static_cast<std::uint64_t>(supernodeCount + 1) * REGION_SUPERNODE_BYTES;
}

StreamedSolveResult StreamedSolve::solve(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                         DenseMatrixView rightHandSides, const DenseKernels& kernels,
                                         const FactorOptions& options, base::MemoryBudget& budget,
                                         std::int64_t regionValues)
{
    if (options.pivoting != PivotingMode::PositiveDefinite)
        throw std::invalid_argument(
            "StreamedSolve::solve: only a positive definite matrix is solved without its factor (no pivot is "
            "delayed); pass PivotingMode::PositiveDefinite");
    if (symbolic.trailingDofCount() > 0)
        throw std::invalid_argument(base::concatenate(
            {"StreamedSolve::solve: the symbolic factor leaves ", std::to_string(symbolic.trailingDofCount()),
             " trailing equations unfactorised; analyse the complete matrix (a partial "
             "factorisation is PartialFactor's)"}));
    if (rightHandSides.rows != static_cast<std::size_t>(symbolic.dofCount()))
        throw std::invalid_argument(
            base::concatenate({"StreamedSolve::solve: the right-hand sides have ", std::to_string(rightHandSides.rows),
                               " rows, the symbolic factor ", std::to_string(symbolic.dofCount()),
                               " equations; give one row per equation"}));
    StreamedSolveResult result;
    result.regions.regionValues = regionValues;
    if (symbolic.dofCount() == 0)
        return result;
    StreamedFactorisation streamed(symbolic, matrix, kernels, options, false, budget);
    result.regions = regions::sweep(streamed, rightHandSides, {}, kernels, options, budget, regionValues, nullptr);
    result.diagnostics = streamed.diagnostics();
    result.workspace = streamed.workspaceUsage();
    return result;
}

StreamedSolveEstimate StreamedSolve::estimate(const SymbolicFactor& symbolic, std::size_t threadCount)
{
    StreamedSolveEstimate estimate;
    estimate.run = PartialFactor::estimateWithoutFactor(symbolic, threadCount);
    const std::int32_t count = factorisedCount(symbolic);
    estimate.supernodeCount = std::max(count, 0);
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int64_t c = symbolic.columnCount(s);
        const std::int64_t r = symbolic.rowCount(s);
        estimate.panelValues += WorkspacePool::classSize(c * (c + r));
        estimate.flopCount += FactorSchedule::supernodeFlops(c, r);
    }
    return estimate;
}

StreamedSolveResult StreamedSolve::solve(const SymbolicFactor& symbolic, FrontSource& source, SolutionSink& sink,
                                         std::size_t columnCount, const DenseKernels& kernels,
                                         const FactorOptions& options, base::MemoryBudget& budget,
                                         std::int64_t regionValues, FrontStreamOrder order,
                                         std::int32_t updateBlockColumns, std::int64_t retainedValues)
{
    FrontStream stream(symbolic, source, columnCount, kernels, options, budget, order, updateBlockColumns);
    StreamedSolveResult result;
    result.regions = stream.solve(sink, regionValues, retainedValues);
    result.diagnostics = stream.diagnostics();
    result.workspace = stream.workspaceUsage();
    return result;
}

FrontStreamEstimate StreamedSolve::estimate(const SymbolicFactor& symbolic, std::size_t threadCount,
                                            std::size_t columnCount, std::int64_t regionValues, FrontStreamOrder order,
                                            std::int32_t updateBlockColumns, std::int64_t retainedValues)
{
    return FrontStream::estimate(symbolic, threadCount, columnCount, regionValues, order, updateBlockColumns,
                                 retainedValues);
}

RegionSweepCost StreamedSolve::regionCost(const SymbolicFactor& symbolic, std::int64_t regionValues)
{
    const std::int32_t count = factorisedCount(symbolic);
    RegionSweepCost cost;
    cost.regionValues = regionValues;
    if (count <= 0)
        return cost;
    base::MemoryBudget scratch(base::MemoryBudget::UNLIMITED);
    const auto total = static_cast<std::size_t>(count);
    base::BudgetedBuffer<std::int64_t> values(scratch, total, "region cost: panel values");
    base::BudgetedBuffer<double> flopsBefore(scratch, total + 1, "region cost: flops");
    base::BudgetedBuffer<std::int32_t> subtreeFirst(scratch, total, "region cost: subtree firsts");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        const std::int64_t c = symbolic.columnCount(s);
        const std::int64_t r = symbolic.rowCount(s);
        values[at] = WorkspacePool::classSize(c * (c + r));
        flopsBefore[at + 1] = flopsBefore[at] + FactorSchedule::supernodeFlops(c, r);
        subtreeFirst[at] = s;
    }
    // Postorder: a child before its parent (StreamedFactorisation's subtree firsts).
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent != Supernodes::ROOT && parent < count)
            subtreeFirst[static_cast<std::size_t>(parent)] =
                std::min(subtreeFirst[static_cast<std::size_t>(parent)], subtreeFirst[static_cast<std::size_t>(s)]);
    }
    const auto valuesOf = [&](std::int32_t s) { return values[static_cast<std::size_t>(s)]; };
    const std::int32_t top = regions::topFirst(count, regionValues, valuesOf);
    for (std::int32_t s = top; s < count; ++s)
        cost.topValues += valuesOf(s);
    cost.keptValues = cost.topValues;
    regions::forEach(
        top, regionValues, valuesOf, [&](std::int32_t s) { return subtreeFirst[static_cast<std::size_t>(s)]; },
        [&](std::int32_t, std::int32_t root, std::int64_t kept)
        {
            ++cost.regionRunCount;
            cost.keptValues = std::max(cost.keptValues, kept);
            const auto first = static_cast<std::size_t>(subtreeFirst[static_cast<std::size_t>(root)]);
            cost.recomputedFlops += flopsBefore[static_cast<std::size_t>(root) + 1] - flopsBefore[first];
        });
    return cost;
}

} // namespace oos::linalg
