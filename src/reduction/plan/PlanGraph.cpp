#include "oos/reduction/plan/PlanGraph.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

PlanGraph::PlanGraph(const linalg::NodeGraph& graph, std::span<const base::EntityId> nodeIds,
                     std::span<const std::int32_t> unconstrainedDofCounts, std::span<const std::int32_t> freeDofCounts,
                     std::span<const base::Vec3> positions, std::vector<DependentPoint> dependents)
    : m_graph(&graph),
      m_nodeIds(nodeIds),
      m_unconstrainedDofCounts(unconstrainedDofCounts),
      m_freeDofCounts(freeDofCounts),
      m_positions(positions),
      m_dependents(std::move(dependents))
{
    const auto count = static_cast<std::size_t>(graph.nodeCount());
    if (nodeIds.size() != count || unconstrainedDofCounts.size() != count || freeDofCounts.size() != count ||
        positions.size() != count)
        throw std::invalid_argument(base::concatenate(
            {"PlanGraph: expected ", std::to_string(count), " IDs, DOF counts and positions (one per node), found ",
             std::to_string(nodeIds.size()), ", ", std::to_string(unconstrainedDofCounts.size()), ", ",
             std::to_string(freeDofCounts.size()), " and ", std::to_string(positions.size())}));
    m_index.reserve(count);
    for (std::size_t node = 0; node < count; ++node)
    {
        if (unconstrainedDofCounts[node] < 0 || unconstrainedDofCounts[node] > 6 || freeDofCounts[node] < 0 ||
            freeDofCounts[node] > unconstrainedDofCounts[node])
            throw std::invalid_argument(base::concatenate({"PlanGraph: node ", std::to_string(nodeIds[node]), " has ",
                                                           std::to_string(unconstrainedDofCounts[node]),
                                                           " unconstrained and ", std::to_string(freeDofCounts[node]),
                                                           " free DOFs; expected 0 <= free <= unconstrained <= 6"}));
        m_index.emplace_back(nodeIds[node], static_cast<std::int32_t>(node));
    }
    std::sort(m_index.begin(), m_index.end());
    for (std::size_t k = 1; k < m_index.size(); ++k)
        if (m_index[k].first == m_index[k - 1].first)
            throw std::invalid_argument(base::concatenate(
                {"PlanGraph: point ", std::to_string(m_index[k].first), " is given twice; IDs must be distinct"}));
    const auto isBefore = [](const DependentPoint& a, const DependentPoint& b)
    { return a.node != b.node ? a.node < b.node : a.relation < b.relation; };
    if (!std::is_sorted(m_dependents.begin(), m_dependents.end(), isBefore))
        throw std::invalid_argument("PlanGraph: the dependent points must be sorted by node, then relation");
}

std::span<const DependentPoint> PlanGraph::dependentsOf(std::int32_t node) const noexcept
{
    const auto first = std::lower_bound(m_dependents.begin(), m_dependents.end(), node,
                                        [](const DependentPoint& entry, std::int32_t n) { return entry.node < n; });
    auto last = first;
    while (last != m_dependents.end() && last->node == node)
        ++last;
    return {first, last};
}

std::int32_t PlanGraph::unitDofCount(std::int32_t node, std::span<const RelationKey> interfaceRelations) const noexcept
{
    std::int32_t count = m_unconstrainedDofCounts[static_cast<std::size_t>(node)];
    for (const DependentPoint& entry : dependentsOf(node))
        if (!std::binary_search(interfaceRelations.begin(), interfaceRelations.end(), entry.relation))
            count -= entry.dofCount;
    return std::max(count, 0);
}

std::optional<std::int32_t> PlanGraph::nodeOf(base::EntityId nodeId) const noexcept
{
    const auto found = std::lower_bound(m_index.begin(), m_index.end(), std::make_pair(nodeId, std::int32_t{0}));
    if (found == m_index.end() || found->first != nodeId)
        return std::nullopt;
    return found->second;
}

} // namespace oos::reduction
