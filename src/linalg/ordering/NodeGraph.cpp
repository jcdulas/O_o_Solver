#include "oos/linalg/ordering/NodeGraph.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

// Nodes per piece of the parallel build: enough work per piece to hide the scheduling, few enough pieces to balance.
constexpr std::size_t NODES_PER_PIECE = 2048;

void requireValidList(const NodeGroupList& list, std::size_t listIndex, std::int32_t nodeCount)
{
    const std::string where = std::string("NodeGraph: group list ").append(std::to_string(listIndex));
    if (list.offsets.empty())
    {
        if (!list.nodes.empty())
            throw std::invalid_argument(base::concatenate({where, ": expected groupCount + 1 offsets, found none for ",
                                                           std::to_string(list.nodes.size()),
                                                           " nodes; give the offsets {0, ...} of the groups"}));
        return;
    }
    if (list.offsets.front() != 0 || list.offsets.back() != static_cast<std::int64_t>(list.nodes.size()))
        throw std::invalid_argument(
            base::concatenate({where, ": expected offsets from 0 to ", std::to_string(list.nodes.size()), ", found ",
                               std::to_string(list.offsets.front()), " to ", std::to_string(list.offsets.back())}));
    for (std::size_t g = 0; g + 1 < list.offsets.size(); ++g)
    {
        if (list.offsets[g + 1] < list.offsets[g])
            throw std::invalid_argument(
                base::concatenate({where, ": expected nondecreasing offsets, found ", std::to_string(list.offsets[g]),
                                   " then ", std::to_string(list.offsets[g + 1]), " at group ", std::to_string(g)}));
    }
    for (std::size_t k = 0; k < list.nodes.size(); ++k)
    {
        if (list.nodes[k] < 0 || list.nodes[k] >= nodeCount)
            throw std::invalid_argument(
                base::concatenate({where, ": expected node indices in [0, ", std::to_string(nodeCount), "), found ",
                                   std::to_string(list.nodes[k]), " at entry ", std::to_string(k)}));
    }
}

// The groups of every list, numbered globally (list 0 first), and for every node the groups that hold it.
class GroupIncidence
{
public:
    GroupIncidence(std::int32_t nodeCount, std::span<const NodeGroupList> lists, base::MemoryBudget& budget)
        : m_lists(lists), m_listStart(lists.size() + 1, 0)
    {
        std::int64_t totalEntries = 0;
        for (std::size_t l = 0; l < lists.size(); ++l)
        {
            m_listStart[l + 1] = m_listStart[l] + static_cast<std::int64_t>(lists[l].groupCount());
            totalEntries += static_cast<std::int64_t>(lists[l].nodes.size());
        }
        m_offsets = base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(nodeCount) + 1,
                                                       "node graph build: incidence offsets");
        m_groups = base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(totalEntries),
                                                      "node graph build: incidence");
        for (const NodeGroupList& list : lists)
        {
            for (const std::int32_t node : list.nodes)
                ++m_offsets[static_cast<std::size_t>(node) + 1];
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(nodeCount); ++i)
            m_offsets[i + 1] += m_offsets[i];
        // Filled in group order, so the groups of a node are in increasing global order.
        base::BudgetedBuffer<std::int64_t> next(budget, static_cast<std::size_t>(nodeCount),
                                                "node graph build: incidence cursor");
        std::copy_n(m_offsets.data(), static_cast<std::size_t>(nodeCount), next.data());
        for (std::size_t l = 0; l < lists.size(); ++l)
        {
            for (std::size_t g = 0; g < lists[l].groupCount(); ++g)
            {
                const std::int64_t globalGroup = m_listStart[l] + static_cast<std::int64_t>(g);
                for (const std::int32_t node : lists[l].group(g))
                    m_groups[static_cast<std::size_t>(next[static_cast<std::size_t>(node)]++)] = globalGroup;
            }
        }
    }

    // Writes into `scratch` the sorted distinct neighbours of `node` through its groups.
    void gatherNeighbours(std::int32_t node, std::vector<std::int32_t>& scratch) const
    {
        scratch.clear();
        const auto begin = static_cast<std::size_t>(m_offsets[static_cast<std::size_t>(node)]);
        const auto end = static_cast<std::size_t>(m_offsets[static_cast<std::size_t>(node) + 1]);
        std::int64_t previousGroup = -1;
        for (std::size_t k = begin; k < end; ++k)
        {
            const std::int64_t globalGroup = m_groups[k];
            if (globalGroup == previousGroup)
                continue; // the node appears twice in this group
            previousGroup = globalGroup;
            const auto list = static_cast<std::size_t>(
                std::upper_bound(m_listStart.begin(), m_listStart.end(), globalGroup) - m_listStart.begin() - 1);
            for (const std::int32_t other :
                 m_lists[list].group(static_cast<std::size_t>(globalGroup - m_listStart[list])))
            {
                if (other != node)
                    scratch.push_back(other);
            }
        }
        std::sort(scratch.begin(), scratch.end());
        scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
    }

private:
    std::span<const NodeGroupList> m_lists;
    std::vector<std::int64_t> m_listStart; // one entry per list, plus one: bounded by the caller's lists
    base::BudgetedBuffer<std::int64_t> m_offsets;
    base::BudgetedBuffer<std::int64_t> m_groups;
};

void requireNodeCount(std::int32_t nodeCount)
{
    if (nodeCount < 0)
        throw std::invalid_argument(
            base::concatenate({"NodeGraph: expected a node count >= 0, found ", std::to_string(nodeCount)}));
}

} // namespace

NodeGraph::NodeGraph(std::int32_t nodeCount, std::span<const NodeGroupList> groupLists, base::MemoryBudget& budget,
                     base::ThreadPool* pool, const base::OperationControl& control)
    : m_nodeCount(nodeCount)
{
    requireNodeCount(nodeCount);
    for (std::size_t l = 0; l < groupLists.size(); ++l)
        requireValidList(groupLists[l], l, nodeCount);
    const GroupIncidence incidence(nodeCount, groupLists, budget);
    const auto count = static_cast<std::size_t>(nodeCount);
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, count + 1, "node graph: offsets");

    // Two passes over the nodes, each node independent of the others: the degrees, then the neighbours at their
    // final place. Gathering twice costs less than keeping every gathered list.
    base::parallelFor(
        pool, count, NODES_PER_PIECE,
        [&](std::size_t begin, std::size_t end)
        {
            std::vector<std::int32_t> scratch;
            for (std::size_t i = begin; i < end; ++i)
            {
                incidence.gatherNeighbours(static_cast<std::int32_t>(i), scratch);
                m_offsets[i + 1] = static_cast<std::int64_t>(scratch.size());
            }
        },
        control.cancellation);
    for (std::size_t i = 0; i < count; ++i)
        m_offsets[i + 1] += m_offsets[i];
    m_neighbours =
        base::BudgetedBuffer<std::int32_t>(budget, static_cast<std::size_t>(m_offsets[count]), "node graph: adjacency");
    base::parallelFor(
        pool, count, NODES_PER_PIECE,
        [&](std::size_t begin, std::size_t end)
        {
            std::vector<std::int32_t> scratch;
            for (std::size_t i = begin; i < end; ++i)
            {
                incidence.gatherNeighbours(static_cast<std::int32_t>(i), scratch);
                std::copy(scratch.begin(), scratch.end(), m_neighbours.data() + static_cast<std::size_t>(m_offsets[i]));
            }
            control.advance(end - begin);
        },
        control.cancellation);
}

NodeGraph NodeGraph::fromAdjacency(std::int32_t nodeCount, std::span<const std::int64_t> offsets,
                                   std::span<const std::int32_t> neighbours, base::MemoryBudget& budget)
{
    requireNodeCount(nodeCount);
    const auto count = static_cast<std::size_t>(nodeCount);
    if (offsets.size() != count + 1 || offsets.front() != 0 ||
        offsets.back() != static_cast<std::int64_t>(neighbours.size()))
        throw std::invalid_argument(base::concatenate({"NodeGraph::fromAdjacency: expected ", std::to_string(count + 1),
                                                       " offsets from 0 to ", std::to_string(neighbours.size()),
                                                       ", found ", std::to_string(offsets.size()), " offsets"}));
    for (std::size_t i = 0; i < count; ++i)
    {
        if (offsets[i + 1] < offsets[i])
            throw std::invalid_argument(base::concatenate(
                {"NodeGraph::fromAdjacency: expected nondecreasing offsets, found a decrease at node ",
                 std::to_string(i)}));
        for (auto k = offsets[i]; k < offsets[i + 1]; ++k)
        {
            const std::int32_t other = neighbours[static_cast<std::size_t>(k)];
            const bool isSorted = k == offsets[i] || neighbours[static_cast<std::size_t>(k) - 1] < other;
            if (other < 0 || other >= nodeCount || other == static_cast<std::int32_t>(i) || !isSorted)
                throw std::invalid_argument(
                    base::concatenate({"NodeGraph::fromAdjacency: the neighbours of node ", std::to_string(i),
                                       " must be sorted, distinct, in range and without the node itself; found ",
                                       std::to_string(other)}));
            const auto otherBegin = neighbours.begin() + offsets[static_cast<std::size_t>(other)];
            const auto otherEnd = neighbours.begin() + offsets[static_cast<std::size_t>(other) + 1];
            if (!std::binary_search(otherBegin, otherEnd, static_cast<std::int32_t>(i)))
                throw std::invalid_argument(
                    base::concatenate({"NodeGraph::fromAdjacency: expected a symmetric graph; node ", std::to_string(i),
                                       " lists node ", std::to_string(other), " which does not list it"}));
        }
    }
    base::BudgetedBuffer<std::int64_t> ownOffsets(budget, offsets.size(), "node graph: offsets");
    base::BudgetedBuffer<std::int32_t> ownNeighbours(budget, neighbours.size(), "node graph: adjacency");
    std::copy(offsets.begin(), offsets.end(), ownOffsets.data());
    std::copy(neighbours.begin(), neighbours.end(), ownNeighbours.data());
    return NodeGraph(nodeCount, std::move(ownOffsets), std::move(ownNeighbours));
}

NodeGraph NodeGraph::inducedSubgraph(std::span<const std::int32_t> nodes, base::MemoryBudget& budget) const
{
    const auto count = static_cast<std::size_t>(m_nodeCount);
    base::BudgetedBuffer<std::int32_t> newIndex(budget, count, "induced subgraph: renumbering");
    std::fill_n(newIndex.data(), count, -1);
    for (std::size_t k = 0; k < nodes.size(); ++k)
    {
        const std::int32_t node = nodes[k];
        if (node < 0 || node >= m_nodeCount || newIndex[static_cast<std::size_t>(node)] >= 0)
            throw std::invalid_argument(base::concatenate(
                {"NodeGraph::inducedSubgraph: expected distinct nodes in [0, ", std::to_string(m_nodeCount),
                 "), found ", std::to_string(node), " at entry ", std::to_string(k)}));
        newIndex[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(k);
    }
    base::BudgetedBuffer<std::int64_t> offsets(budget, nodes.size() + 1, "node graph: offsets");
    for (std::size_t k = 0; k < nodes.size(); ++k)
    {
        std::int64_t kept = 0;
        for (const std::int32_t other : neighbours(nodes[k]))
            kept += newIndex[static_cast<std::size_t>(other)] >= 0 ? 1 : 0;
        offsets[k + 1] = offsets[k] + kept;
    }
    base::BudgetedBuffer<std::int32_t> adjacency(budget, static_cast<std::size_t>(offsets[nodes.size()]),
                                                 "node graph: adjacency");
    for (std::size_t k = 0; k < nodes.size(); ++k)
    {
        std::int32_t* out = adjacency.data() + offsets[k];
        std::int32_t* const first = out;
        for (const std::int32_t other : neighbours(nodes[k]))
        {
            const std::int32_t mapped = newIndex[static_cast<std::size_t>(other)];
            if (mapped >= 0)
                *out++ = mapped;
        }
        std::sort(first, out);
    }
    return NodeGraph(static_cast<std::int32_t>(nodes.size()), std::move(offsets), std::move(adjacency));
}

} // namespace oos::linalg
