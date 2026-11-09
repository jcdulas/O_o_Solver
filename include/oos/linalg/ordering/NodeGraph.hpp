#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace oos::linalg
{

// The adjacency graph of the nodes of a model: two nodes are adjacent when an element, or an extra
// coupling group (absorbed spider, internal constraint), holds both. It is the structure every sparse operation is
// built on: the block pattern of the matrices (SparsePattern), the orderings (each node a supervariable of its DOFs)
// and the symbolic analysis.
//
// Storage: compressed rows, 64-bit offsets (the number of edges may exceed 2³¹), 32-bit node indices. Invariants: the
// neighbours of each node are sorted ascending, distinct, without the node itself, and the graph is symmetric.
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: owns its arrays (budgeted); the budget
// must outlive the graph. Movable, not copyable.
class NodeGraph
{
public:
    NodeGraph() = default;

    // The graph of `nodeCount` nodes coupled by the groups of every list. The build runs on the pool (pieces of nodes
    // fixed by the work); the graph does not depend on the number of threads.
    // Throws std::invalid_argument when a list is malformed or names a node outside [0, nodeCount), MemoryBudgetError
    // when the graph does not fit the budget.
    // control: checked between pieces of nodes (OperationCancelledError); its counter is advanced by the nodes whose
    // neighbours are stored, the node count in all.
    NodeGraph(std::int32_t nodeCount, std::span<const NodeGroupList> groupLists, base::MemoryBudget& budget,
              base::ThreadPool* pool = nullptr, const base::OperationControl& control = {});

    // A graph given by its compressed rows, checked against the invariants above (std::invalid_argument otherwise).
    [[nodiscard]] static NodeGraph fromAdjacency(std::int32_t nodeCount, std::span<const std::int64_t> offsets,
                                                 std::span<const std::int32_t> neighbours, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t nodeCount() const noexcept { return m_nodeCount; }
    // The number of entries of the adjacency: twice the number of edges.
    [[nodiscard]] std::int64_t adjacencySize() const noexcept { return static_cast<std::int64_t>(m_neighbours.size()); }
    [[nodiscard]] std::int32_t degree(std::int32_t node) const noexcept
    {
        return static_cast<std::int32_t>(m_offsets[static_cast<std::size_t>(node) + 1] -
                                         m_offsets[static_cast<std::size_t>(node)]);
    }
    [[nodiscard]] std::span<const std::int32_t> neighbours(std::int32_t node) const noexcept
    {
        const auto begin = static_cast<std::size_t>(m_offsets[static_cast<std::size_t>(node)]);
        const auto end = static_cast<std::size_t>(m_offsets[static_cast<std::size_t>(node) + 1]);
        return m_neighbours.span().subspan(begin, end - begin);
    }
    [[nodiscard]] std::span<const std::int64_t> offsets() const noexcept { return m_offsets.span(); }
    [[nodiscard]] std::span<const std::int32_t> adjacency() const noexcept { return m_neighbours.span(); }

    // The graph induced by `nodes` (distinct, in range): node nodes[k] becomes node k, and only the edges between
    // listed nodes are kept. Used to drop the nodes whose DOFs are all eliminated before an ordering, and to order the
    // leaves of a dissection. Throws std::invalid_argument on a repeated or out-of-range node.
    [[nodiscard]] NodeGraph inducedSubgraph(std::span<const std::int32_t> nodes, base::MemoryBudget& budget) const;

private:
    NodeGraph(std::int32_t nodeCount, base::BudgetedBuffer<std::int64_t> offsets,
              base::BudgetedBuffer<std::int32_t> neighbours)
        : m_nodeCount(nodeCount), m_offsets(std::move(offsets)), m_neighbours(std::move(neighbours))
    {
    }

    std::int32_t m_nodeCount = 0;
    base::BudgetedBuffer<std::int64_t> m_offsets;    // nodeCount + 1 entries
    base::BudgetedBuffer<std::int32_t> m_neighbours; // m_offsets[nodeCount] entries
};

} // namespace oos::linalg
