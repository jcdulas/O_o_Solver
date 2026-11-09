#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/symbolic/EliminationTree.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The column counts of the Cholesky factor of a node graph, without forming its structure (J. R. Gilbert, E. G. Ng
// and B. W. Peyton, "An efficient algorithm to compute row and column counts for sparse Cholesky factorization", SIAM
// J. Matrix Anal. Appl. 15(4), 1994; in the form of T. A. Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006,
// §4.5, cs_counts).
//
// Column j of L holds row i > j exactly when j is in the row subtree T_i (the subtree of the elimination tree spanned
// by the columns k < i with A_ik ≠ 0, up to i). The count of column j is the number of row subtrees through j: each
// T_i adds w_i at each of its leaves (found from the postorder by the "first descendant" test), removes it at the least
// common ancestor of consecutive leaves, and once above i; the counts are the subtree sums of these differences.
// Weighted form: a node of weight w (its DOFs) counts w rows, so belowWeight is the number of DOF rows of the node's
// columns below the node's own diagonal block.
//
// Thread-safety: immutable once built. Ownership: owns its counts (budgeted). Movable, not copyable.
class ColumnCounts
{
public:
    // order and tree describe the same elimination; postorder is a postorder of the tree (EliminationTree::postorder);
    // nodeWeights is indexed by node (empty: 1 each).
    ColumnCounts(const NodeGraph& graph, const NodePermutation& order, const EliminationTree& tree,
                 std::span<const std::int32_t> postorder, std::span<const std::int32_t> nodeWeights,
                 base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t size() const noexcept { return static_cast<std::int32_t>(m_belowNodes.size()); }
    // The nodes in the structure of column k strictly below the diagonal.
    [[nodiscard]] std::int32_t belowNodeCount(std::int32_t position) const noexcept
    {
        return static_cast<std::int32_t>(m_belowNodes[static_cast<std::size_t>(position)]);
    }
    // Their total weight.
    [[nodiscard]] std::int64_t belowWeight(std::int32_t position) const noexcept
    {
        return m_belowWeight[static_cast<std::size_t>(position)];
    }

private:
    base::BudgetedBuffer<std::int64_t> m_belowNodes;
    base::BudgetedBuffer<std::int64_t> m_belowWeight;
};

} // namespace oos::linalg
