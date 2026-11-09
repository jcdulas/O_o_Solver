#include "oos/solutions/common/FsetOrdering.hpp"

#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/Stopwatch.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace oos::solutions
{

FsetOrdering::FsetOrdering(const linalg::NodeGraph& graph, const assembly::DofMap& dofMap,
                           const assembly::DofSets& sets, assembly::DofSet keptSet, linalg::OrderingMethod ordering,
                           base::MemoryBudget& budget, base::ThreadPool* pool,
                           const base::CancellationToken* cancellation, std::span<const std::int32_t> lastNodes,
                           const linalg::AmalgamationThresholds& thresholds, linalg::ComponentCuts cuts,
                           std::span<const std::int32_t> componentOfNode)
{
    base::Stopwatch watch;
    const base::BudgetedBuffer<std::uint8_t> dofCounts = dofMap.nodeDofCounts(budget);
    const base::BudgetedBuffer<std::uint8_t> eliminated = sets.eliminationFlags(keptSet, budget);

    // The nodes with a kept DOF, their kept DOF counts (the weights of the ordering and of the symbolic analysis) and
    // positions. Nodes without one are dropped from the ordered graph (DofSelection: an absent node keeps no DOF).
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    base::BudgetedBuffer<std::int32_t> keptNodes(budget, nodeCount, "f-set nodes");
    base::BudgetedBuffer<std::int32_t> weights(budget, nodeCount, "f-set node weights");
    base::BudgetedBuffer<base::Vec3> coordinates(budget, nodeCount, "f-set node positions");
    std::size_t kept = 0;
    std::size_t firstDof = 0;
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        std::int32_t free = 0;
        for (std::size_t c = 0; c < dofCounts[node]; ++c)
            free += eliminated[firstDof + c] == 0 ? 1 : 0;
        firstDof += dofCounts[node];
        if (free == 0)
            continue;
        keptNodes[kept] = static_cast<std::int32_t>(node);
        weights[kept] = free;
        // A scalar point has no position: it takes that of a grid point it is coupled to (the dissection only needs
        // it near its neighbours), the origin when it has none.
        std::int32_t located = static_cast<std::int32_t>(node);
        if (located >= dofMap.gridCount())
        {
            located = -1;
            for (const std::int32_t neighbour : graph.neighbours(static_cast<std::int32_t>(node)))
                if (neighbour < dofMap.gridCount())
                {
                    located = neighbour;
                    break;
                }
        }
        coordinates[kept] = located >= 0 ? dofMap.positionInBasic(located) : base::Vec3{};
        ++kept;
    }
    if (kept == 0)
    {
        m_selection = linalg::DofSelection(dofCounts.span(), eliminated.span(), {}, budget);
        return;
    }

    const base::OperationControl control{cancellation, nullptr};
    const std::span<const std::int32_t> nodes = keptNodes.span().first(kept);
    const std::span<const std::int32_t> nodeWeights = weights.span().first(kept);
    const linalg::NodeGraph subgraph = graph.inducedSubgraph(nodes, budget);
    // The nodes ordered last, as positions in the kept nodes (sorted: a binary search).
    std::vector<std::int32_t> trailing; // bounded by lastNodes
    for (const std::int32_t node : lastNodes)
    {
        const auto found = std::lower_bound(nodes.begin(), nodes.end(), node);
        if (found != nodes.end() && *found == node)
            trailing.push_back(static_cast<std::int32_t>(found - nodes.begin()));
    }
    // The components of the kept nodes, for the dissection's cuts.
    base::BudgetedBuffer<std::int32_t> components(budget, componentOfNode.empty() ? 0 : kept, "f-set node components");
    for (std::size_t k = 0; k < components.size(); ++k)
        components[k] = componentOfNode[static_cast<std::size_t>(nodes[k])];
    linalg::DissectionOptions dissection;
    dissection.componentCuts = cuts;
    dissection.componentOfNode = components.span();
    const linalg::NodePermutation order =
        linalg::ConstrainedOrdering(dissection)
            .order(subgraph, nodeWeights, coordinates.span().first(kept), trailing, ordering, budget, pool, control);
    m_timings.ordering = watch.lap();
    m_symbolic = linalg::SymbolicFactor::analyse(subgraph, nodeWeights, order, 0, thresholds, budget, control);

    // The factor's order of the kept nodes, mapped back to the model's nodes, numbers the kept DOFs.
    base::BudgetedBuffer<std::int32_t> nodeOrder(budget, kept, "f-set node order");
    const std::span<const std::int32_t> symbolicOrder = m_symbolic.nodeOrder().nodes();
    for (std::size_t k = 0; k < kept; ++k)
        nodeOrder[k] = nodes[static_cast<std::size_t>(symbolicOrder[k])];
    m_selection = linalg::DofSelection(dofCounts.span(), eliminated.span(), nodeOrder.span(), budget);
    m_timings.symbolic = watch.lap();
}

} // namespace oos::solutions
