#include "oos/linalg/ordering/QuotientGraph.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::linalg
{

QuotientGraph::QuotientGraph(const NodeGraph& graph, std::span<const std::int32_t> unitOfNode, std::int32_t unitCount,
                             std::span<const std::int32_t> nodeWeights, std::span<const base::Vec3> coordinates,
                             base::MemoryBudget& budget)
{
    const auto n = static_cast<std::size_t>(graph.nodeCount());
    if (unitOfNode.size() != n || (!nodeWeights.empty() && nodeWeights.size() != n) ||
        (!coordinates.empty() && coordinates.size() != n) || unitCount < 0)
        throw std::invalid_argument(
            base::concatenate({"QuotientGraph: expected one unit (and weight and coordinate, when given) per node (",
                               std::to_string(n), ") and a unit count >= 0"}));
    const auto units = static_cast<std::size_t>(unitCount);
    m_unitWeights = base::BudgetedBuffer<std::int32_t>(budget, units, "quotient graph: unit weights");
    m_centroids = base::BudgetedBuffer<base::Vec3>(budget, units, "quotient graph: centroids");
    m_unitNodeStart = base::BudgetedBuffer<std::int64_t>(budget, units + 1, "quotient graph: unit starts");
    m_unitNodes = base::BudgetedBuffer<std::int32_t>(budget, n, "quotient graph: unit nodes");
    for (std::size_t node = 0; node < n; ++node)
    {
        const std::int32_t unit = unitOfNode[node];
        if (unit < 0 || unit >= unitCount)
            throw std::invalid_argument(
                base::concatenate({"QuotientGraph: expected units in [0, ", std::to_string(unitCount), "), found ",
                                   std::to_string(unit), " at node ", std::to_string(node)}));
        const std::int32_t w = nodeWeights.empty() ? 1 : nodeWeights[node];
        m_unitWeights[static_cast<std::size_t>(unit)] += w;
        if (!coordinates.empty())
            m_centroids[static_cast<std::size_t>(unit)] =
                m_centroids[static_cast<std::size_t>(unit)] + static_cast<double>(w) * coordinates[node];
        ++m_unitNodeStart[static_cast<std::size_t>(unit) + 1];
    }
    for (std::size_t u = 0; u < units; ++u)
    {
        m_unitNodeStart[u + 1] += m_unitNodeStart[u];
        if (m_unitWeights[u] > 0)
            m_centroids[u] = m_centroids[u] / static_cast<double>(m_unitWeights[u]);
    }
    {
        base::BudgetedBuffer<std::int64_t> cursor(budget, units, "quotient graph: cursor");
        std::copy_n(m_unitNodeStart.data(), units, cursor.data());
        for (std::size_t node = 0; node < n; ++node)
            m_unitNodes[static_cast<std::size_t>(cursor[static_cast<std::size_t>(unitOfNode[node])]++)] =
                static_cast<std::int32_t>(node);
    }
    // The units adjacent to each unit: the units of the neighbours of its nodes, sorted and distinct. Two passes
    // (count, fill) with a scratch list per unit, bounded by the unit's boundary.
    base::BudgetedBuffer<std::int64_t> offsets(budget, units + 1, "node graph: offsets");
    std::vector<std::int32_t> scratch;
    const auto gather = [&](std::size_t unit)
    {
        scratch.clear();
        for (auto k = m_unitNodeStart[unit]; k < m_unitNodeStart[unit + 1]; ++k)
        {
            for (const std::int32_t neighbour : graph.neighbours(m_unitNodes[static_cast<std::size_t>(k)]))
            {
                const std::int32_t other = unitOfNode[static_cast<std::size_t>(neighbour)];
                if (other != static_cast<std::int32_t>(unit))
                    scratch.push_back(other);
            }
        }
        std::sort(scratch.begin(), scratch.end());
        scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
    };
    for (std::size_t u = 0; u < units; ++u)
    {
        gather(u);
        offsets[u + 1] = offsets[u] + static_cast<std::int64_t>(scratch.size());
    }
    base::BudgetedBuffer<std::int32_t> adjacency(budget, static_cast<std::size_t>(offsets[units]),
                                                 "node graph: adjacency");
    for (std::size_t u = 0; u < units; ++u)
    {
        gather(u);
        std::copy(scratch.begin(), scratch.end(), adjacency.data() + offsets[u]);
    }
    m_graph = NodeGraph::fromAdjacency(unitCount, offsets.span(), adjacency.span(), budget);
}

NodePermutation QuotientGraph::expandUnitOrder(const NodePermutation& unitOrder, base::MemoryBudget& budget) const
{
    if (unitOrder.size() != m_graph.nodeCount())
        throw std::invalid_argument(base::concatenate(
            {"QuotientGraph::expandUnitOrder: the order has ", std::to_string(unitOrder.size()), " units, the graph ",
             std::to_string(m_graph.nodeCount()), "; give one position per unit of the graph"}));
    base::BudgetedBuffer<std::int32_t> nodes(budget, m_unitNodes.size(), "node permutation");
    std::size_t next = 0;
    for (std::int32_t k = 0; k < unitOrder.size(); ++k)
    {
        const auto unit = static_cast<std::size_t>(unitOrder.nodeAt(k));
        for (auto i = m_unitNodeStart[unit]; i < m_unitNodeStart[unit + 1]; ++i)
            nodes[next++] = m_unitNodes[static_cast<std::size_t>(i)];
    }
    return NodePermutation(std::move(nodes), budget);
}

} // namespace oos::linalg
