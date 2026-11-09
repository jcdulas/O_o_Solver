#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The elimination tree of a symmetric matrix whose graph is `graph` and whose nodes are eliminated in `order`
// (J. W. H. Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix Anal. Appl. 11, 1990): the
// parent of column j is the first row below the diagonal of column j of the factor, parent(j) = min{i > j : L_ij ≠ 0}.
// Everything is in positions of the order (column k is the k-th eliminated node), so parent(k) > k.
//
// Built by Liu's algorithm with path compression on the ancestors, O(|E| log n) in the worst case, near-linear in
// practice; no factor structure is formed.
//
// Thread-safety: immutable once built. Ownership: owns its parents (budgeted). Movable, not copyable.
class EliminationTree
{
public:
    static constexpr std::int32_t ROOT = -1;

    // Throws std::invalid_argument when the order is not of the size of the graph.
    EliminationTree(const NodeGraph& graph, const NodePermutation& order, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t size() const noexcept { return static_cast<std::int32_t>(m_parent.size()); }
    // The parent position of position k, or ROOT.
    [[nodiscard]] std::int32_t parent(std::int32_t position) const noexcept
    {
        return m_parent[static_cast<std::size_t>(position)];
    }
    [[nodiscard]] std::span<const std::int32_t> parents() const noexcept { return m_parent.span(); }

    // A postorder of the tree: post[k] is the k-th position visited; children come before their parent and each
    // subtree is contiguous; roots, and the children of a node, are visited in increasing position (deterministic).
    [[nodiscard]] base::BudgetedBuffer<std::int32_t> postorder(base::MemoryBudget& budget) const;

private:
    base::BudgetedBuffer<std::int32_t> m_parent;
};

} // namespace oos::linalg
