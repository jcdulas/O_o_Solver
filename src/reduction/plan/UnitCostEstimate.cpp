#include "oos/reduction/plan/UnitCostEstimate.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::reduction
{

double panelFlopCount(std::int64_t columns, std::int64_t rowsBelow) noexcept
{
    double flops = 0.0;
    for (std::int64_t column = 0; column < columns; ++column)
    {
        const auto below = static_cast<double>(columns - 1 - column + rowsBelow);
        flops += below * (below + 1.0) + 2.0 * below;
    }
    return flops;
}

UnitCostEstimate UnitCostEstimate::fromSymbolic(const linalg::SymbolicFactor& symbolic)
{
    UnitCostEstimate cost;
    cost.nodeCount = symbolic.nodeCount();
    cost.boundaryDofCount = symbolic.trailingDofCount();
    cost.interiorDofCount = symbolic.dofCount() - cost.boundaryDofCount;
    cost.interiorFactorValueCount = symbolic.panelValueCount();
    cost.flopCount = symbolic.flopCount();
    if (cost.boundaryDofCount > 0)
    {
        // The trailing supernode is K* itself: PartialFactor leaves it as the assembled front, it is not factorised.
        const std::int32_t trailing = symbolic.supernodeCount() - 1;
        const std::int64_t columns = symbolic.columnCount(trailing);
        const std::int64_t rows = symbolic.rowCount(trailing);
        cost.interiorFactorValueCount -= columns * (columns + rows);
        cost.flopCount -= panelFlopCount(columns, rows);
    }
    cost.schurValueCount = cost.boundaryDofCount * cost.boundaryDofCount;
    cost.memoryPeakBytes = symbolic.memoryPeakBytes();
    cost.estimatedSeconds = cost.flopCount / ESTIMATE_FLOP_RATE;
    cost.isKeptFullRecommended = cost.schurValueCount > cost.interiorFactorValueCount;
    const std::int32_t interiorSupernodes = symbolic.supernodeCount() - (cost.boundaryDofCount > 0 ? 1 : 0);
    cost.supernodeCount = interiorSupernodes;
    for (std::int32_t s = 0; s < interiorSupernodes; ++s)
        cost.factorRowCount += symbolic.columnCount(s) + symbolic.rowCount(s);
    // A unit's partial factorisation is positive definite (PartialFactor).
    cost.factorisationBytes =
        linalg::SupernodalFactor::estimateMemoryBytes(symbolic, 1, linalg::PivotingMode::PositiveDefinite);
    cost.withoutFactor = linalg::PartialFactor::estimateWithoutFactor(symbolic, 1);
    return cost;
}

UnitCostEstimate UnitCostEstimate::fromGraph(const linalg::SymbolicFactor& symbolic, const linalg::NodeGraph& graph,
                                             std::span<const std::int32_t> weights)
{
    UnitCostEstimate cost = fromSymbolic(symbolic);
    cost.blockCount = graph.nodeCount() + graph.adjacencySize() / 2;
    for (std::int32_t node = 0; node < graph.nodeCount(); ++node)
    {
        const std::int64_t w = weights[static_cast<std::size_t>(node)];
        cost.matrixValueCount += w * (w + 1) / 2;
        for (const std::int32_t neighbour : graph.neighbours(node))
            if (neighbour < node)
                cost.matrixValueCount += w * weights[static_cast<std::size_t>(neighbour)];
    }
    return cost;
}

UnitCostEstimate UnitCostEstimate::estimate(const PlanGraph& graph, std::span<const base::EntityId> nodes,
                                            std::span<const base::EntityId> boundary,
                                            std::span<const RelationKey> interfaceRelations, base::MemoryBudget& budget,
                                            base::ThreadPool* pool)
{
    // The unit's points with free DOFs, by increasing node index (the induced subgraph numbers them in this order).
    base::BudgetedBuffer<std::int32_t> candidates(budget, nodes.size(), "unit points");
    std::size_t keptCount = 0;
    for (const base::EntityId nodeId : nodes)
    {
        const std::optional<std::int32_t> node = graph.nodeOf(nodeId);
        if (!node)
            throw std::invalid_argument(
                base::concatenate({"UnitCostEstimate: point ", std::to_string(nodeId),
                                   " is not a point of the model; the units and the plan graph must come from one "
                                   "snapshot"}));
        if (graph.unitDofCount(*node, interfaceRelations) > 0)
            candidates[keptCount++] = *node;
    }
    if (keptCount == 0)
        return {};
    const std::span<std::int32_t> kept = candidates.span().first(keptCount);
    std::sort(kept.begin(), kept.end());

    base::BudgetedBuffer<std::int32_t> weights(budget, keptCount, "unit point weights");
    base::BudgetedBuffer<base::Vec3> positions(budget, keptCount, "unit point positions");
    base::BudgetedBuffer<std::int32_t> trailing(budget, std::min(keptCount, boundary.size()), "unit boundary points");
    std::size_t trailingCount = 0;
    for (std::size_t k = 0; k < keptCount; ++k)
    {
        const auto node = static_cast<std::size_t>(kept[k]);
        weights[k] = graph.unitDofCount(kept[k], interfaceRelations);
        positions[k] = graph.positions()[node];
        if (std::binary_search(boundary.begin(), boundary.end(), graph.nodeIds()[node]))
            trailing[trailingCount++] = static_cast<std::int32_t>(k);
    }
    const linalg::NodeGraph subgraph = graph.graph().inducedSubgraph(kept, budget);
    const linalg::NodePermutation order = linalg::ConstrainedOrdering().order(
        subgraph, weights.span(), positions.span(), trailing.span().first(trailingCount),
        linalg::OrderingMethod::Automatic, budget, pool);
    const linalg::SymbolicFactor symbolic =
        linalg::SymbolicFactor::analyse(subgraph, weights.span(), order, static_cast<std::int32_t>(trailingCount),
                                        linalg::AmalgamationThresholds{}, budget);
    return fromGraph(symbolic, subgraph, weights.span());
}

} // namespace oos::reduction
