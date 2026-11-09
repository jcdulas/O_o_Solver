#include "oos/linalg/ordering/ConstrainedOrdering.hpp"

#include "TrailingSet.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{

OrderingMethod ConstrainedOrdering::chooseMethod(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                                 std::span<const base::Vec3> coordinates)
{
    if (coordinates.empty() || graph.nodeCount() == 0)
        return OrderingMethod::MinimumDegree;
    std::int64_t weight = 0;
    for (std::int32_t node = 0; node < graph.nodeCount(); ++node)
        weight += nodeWeights.empty() ? 1 : nodeWeights[static_cast<std::size_t>(node)];
    const double meanDegree = static_cast<double>(graph.adjacencySize()) / static_cast<double>(graph.nodeCount());
    const std::int64_t threshold =
        meanDegree > VOLUME_MEAN_DEGREE ? DISSECTION_MIN_WEIGHT_VOLUME : DISSECTION_MIN_WEIGHT_SURFACE;
    return weight >= threshold ? OrderingMethod::NestedDissection : OrderingMethod::MinimumDegree;
}

NodePermutation ConstrainedOrdering::order(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                           std::span<const base::Vec3> coordinates,
                                           std::span<const std::int32_t> trailingNodes, OrderingMethod method,
                                           base::MemoryBudget& budget, base::ThreadPool* pool,
                                           const base::OperationControl& control) const
{
    if (method == OrderingMethod::Automatic)
        method = chooseMethod(graph, nodeWeights, coordinates);
    switch (method)
    {
    case OrderingMethod::NestedDissection:
        if (coordinates.empty())
            throw std::invalid_argument(
                "ConstrainedOrdering: nested dissection needs the node coordinates; give them or choose "
                "OrderingMethod::MinimumDegree");
        return GeometricDissection(m_dissection, m_minimumDegree)
            .order(graph, nodeWeights, coordinates, trailingNodes, budget, pool, control);
    case OrderingMethod::Natural:
    {
        const TrailingSet trailing(graph.nodeCount(), trailingNodes, budget);
        base::BudgetedBuffer<std::int32_t> nodes(budget, static_cast<std::size_t>(graph.nodeCount()),
                                                 "node permutation");
        std::size_t next = 0;
        for (std::int32_t node = 0; node < graph.nodeCount(); ++node)
        {
            if (!trailing.contains(node))
                nodes[next++] = node;
        }
        for (const std::int32_t node : trailing.nodes())
            nodes[next++] = node;
        control.advance(static_cast<std::uint64_t>(graph.nodeCount()));
        return NodePermutation(std::move(nodes), budget);
    }
    case OrderingMethod::MinimumDegree:
    case OrderingMethod::Automatic:
        break;
    }
    return ApproximateMinimumDegree(m_minimumDegree).order(graph, nodeWeights, trailingNodes, budget, control);
}

} // namespace oos::linalg
