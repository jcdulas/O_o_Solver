#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The quotient graph of a partition of the nodes into units (the interface problem is ordered by nested
// dissection on the graph of the units): one vertex per unit, two units adjacent when an edge of the node graph joins
// them. Each unit carries the weight of its nodes and the weighted centroid of their coordinates, so that the
// orderings run on it as on any node graph; an order of the units expands to an order of the nodes, unit by unit.
//
// Thread-safety: immutable once built. Ownership: owns its graph and arrays (budgeted). Movable, not copyable.
class QuotientGraph
{
public:
    // unitOfNode: the unit of each node, in [0, unitCount). nodeWeights: empty for 1 each. coordinates: empty when
    // none are known (the centroids are then zero).
    // Throws std::invalid_argument on a wrong size or a unit out of range.
    QuotientGraph(const NodeGraph& graph, std::span<const std::int32_t> unitOfNode, std::int32_t unitCount,
                  std::span<const std::int32_t> nodeWeights, std::span<const base::Vec3> coordinates,
                  base::MemoryBudget& budget);

    [[nodiscard]] const NodeGraph& graph() const noexcept { return m_graph; }
    [[nodiscard]] std::span<const std::int32_t> unitWeights() const noexcept { return m_unitWeights.span(); }
    [[nodiscard]] std::span<const base::Vec3> unitCentroids() const noexcept { return m_centroids.span(); }

    // The nodes unit by unit in the order of `unitOrder`, each unit's nodes in increasing index.
    [[nodiscard]] NodePermutation expandUnitOrder(const NodePermutation& unitOrder, base::MemoryBudget& budget) const;

private:
    NodeGraph m_graph;
    base::BudgetedBuffer<std::int32_t> m_unitWeights;
    base::BudgetedBuffer<base::Vec3> m_centroids;
    base::BudgetedBuffer<std::int64_t> m_unitNodeStart; // the nodes of unit u: m_unitNodes[start[u], start[u + 1])
    base::BudgetedBuffer<std::int32_t> m_unitNodes;
};

} // namespace oos::linalg
