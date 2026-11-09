#include "oos/reduction/plan/BoundaryPromotion.hpp"

#include <algorithm>
#include <map>
#include <optional>

namespace oos::reduction
{

std::vector<BoundaryNode> BoundaryPromotion::apply(const PlanGraph& graph, std::span<const base::EntityId> unitNodes,
                                                   std::span<const BoundaryNode> boundary,
                                                   std::span<const RelationKey> interfaceRelations,
                                                   std::span<const base::EntityId> relationPoints)
{
    std::map<base::EntityId, std::uint8_t> reasons; // bounded by the unit's boundary and the points it promotes
    for (const BoundaryNode& node : boundary)
    {
        const std::optional<std::int32_t> modelNode = graph.nodeOf(node.nodeId);
        if (!modelNode)
            continue;
        if (graph.unitDofCount(*modelNode, interfaceRelations) > 0 || node.has(BoundaryReason::SharedPoint) ||
            std::binary_search(relationPoints.begin(), relationPoints.end(), node.nodeId))
            reasons[node.nodeId] |= node.reasons;
        for (const DependentPoint& entry : graph.dependentsOf(*modelNode))
        {
            if (std::binary_search(interfaceRelations.begin(), interfaceRelations.end(), entry.relation))
                continue;
            for (const std::int32_t independent : entry.independentNodes)
            {
                const base::EntityId nodeId = graph.nodeIds()[static_cast<std::size_t>(independent)];
                if (graph.unitDofCount(independent, interfaceRelations) > 0 &&
                    std::binary_search(unitNodes.begin(), unitNodes.end(), nodeId))
                    reasons[nodeId] |= node.reasons;
            }
        }
    }
    std::vector<BoundaryNode> result;
    result.reserve(reasons.size());
    for (const auto& [nodeId, bits] : reasons)
        result.push_back(BoundaryNode{nodeId, bits});
    return result;
}

std::vector<base::EntityId> BoundaryPromotion::interfaceRelationPoints(const PlanGraph& graph,
                                                                       std::span<const RelationKey> interfaceRelations)
{
    std::vector<base::EntityId> points;
    for (const DependentPoint& entry : graph.dependents())
    {
        if (!std::binary_search(interfaceRelations.begin(), interfaceRelations.end(), entry.relation))
            continue;
        points.push_back(graph.nodeIds()[static_cast<std::size_t>(entry.node)]);
        for (const std::int32_t independent : entry.independentNodes)
            points.push_back(graph.nodeIds()[static_cast<std::size_t>(independent)]);
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

} // namespace oos::reduction
