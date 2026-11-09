#include "oos/linalg/symbolic/ColumnCounts.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

constexpr std::int32_t NONE = -1;

// The leaf test of Davis (cs_leaf): is column j a leaf of the row subtree T_i, and if j is a later leaf, the least
// common ancestor q of j and the previous leaf of T_i. Returns 0 (not a leaf), 1 (first leaf, q = i) or 2.
struct LeafState
{
    std::int32_t* first;    // first[j]: postorder index of the first descendant of j
    std::int32_t* maxFirst; // maxFirst[i]: the largest first[j] of the leaves of T_i seen so far
    std::int32_t* prevLeaf; // prevLeaf[i]: the previous leaf of T_i
    std::int32_t* ancestor; // union-find forest of the columns already processed, path-compressed
};

int classifyLeaf(std::int32_t i, std::int32_t j, LeafState& state, std::int32_t& q)
{
    if (i <= j || state.first[j] <= state.maxFirst[i])
        return 0; // j is above i, or a descendant of an earlier leaf of T_i
    state.maxFirst[i] = state.first[j];
    const std::int32_t previous = state.prevLeaf[i];
    state.prevLeaf[i] = j;
    if (previous == NONE)
    {
        q = i;
        return 1;
    }
    for (q = previous; q != state.ancestor[q]; q = state.ancestor[q])
    {
    }
    for (std::int32_t s = previous; s != q;)
    {
        const std::int32_t parent = state.ancestor[s];
        state.ancestor[s] = q;
        s = parent;
    }
    return 2;
}

} // namespace

ColumnCounts::ColumnCounts(const NodeGraph& graph, const NodePermutation& order, const EliminationTree& tree,
                           std::span<const std::int32_t> postorder, std::span<const std::int32_t> nodeWeights,
                           base::MemoryBudget& budget)
    : m_belowNodes(budget, static_cast<std::size_t>(tree.size()), "column counts: nodes"),
      m_belowWeight(budget, static_cast<std::size_t>(tree.size()), "column counts: weights")
{
    const std::int32_t n = tree.size();
    if (order.size() != n || graph.nodeCount() != n || postorder.size() != static_cast<std::size_t>(n) ||
        (!nodeWeights.empty() && nodeWeights.size() != static_cast<std::size_t>(n)))
        throw std::invalid_argument(base::concatenate(
            {"ColumnCounts: the graph, order, tree, postorder and weights must have one entry per node (",
             std::to_string(n), ")"}));
    const auto count = static_cast<std::size_t>(n);
    base::BudgetedBuffer<std::int32_t> first(budget, count, "column counts: first descendants");
    base::BudgetedBuffer<std::int32_t> maxFirst(budget, count, "column counts: largest first");
    base::BudgetedBuffer<std::int32_t> prevLeaf(budget, count, "column counts: previous leaves");
    base::BudgetedBuffer<std::int32_t> ancestor(budget, count, "column counts: ancestors");
    std::fill_n(first.data(), count, NONE);
    std::fill_n(maxFirst.data(), count, NONE);
    std::fill_n(prevLeaf.data(), count, NONE);
    const auto weightAt = [&](std::int32_t position) -> std::int64_t
    { return nodeWeights.empty() ? 1 : nodeWeights[static_cast<std::size_t>(order.nodeAt(position))]; };

    // The differences, in m_belowNodes (counts) and m_belowWeight (weights) until the subtree sums.
    std::int64_t* const deltaCount = m_belowNodes.data();
    std::int64_t* const deltaWeight = m_belowWeight.data();
    for (std::int32_t k = 0; k < n; ++k)
    {
        std::int32_t j = postorder[static_cast<std::size_t>(k)];
        if (first[static_cast<std::size_t>(j)] == NONE)
        {
            // A leaf of the tree: its own diagonal (row j of T_j) is counted here.
            deltaCount[j] = 1;
            deltaWeight[j] = weightAt(j);
        }
        for (; j != EliminationTree::ROOT && first[static_cast<std::size_t>(j)] == NONE; j = tree.parent(j))
            first[static_cast<std::size_t>(j)] = k;
    }
    for (std::int32_t i = 0; i < n; ++i)
        ancestor[static_cast<std::size_t>(i)] = i;
    LeafState state{first.data(), maxFirst.data(), prevLeaf.data(), ancestor.data()};
    for (std::int32_t k = 0; k < n; ++k)
    {
        const std::int32_t j = postorder[static_cast<std::size_t>(k)];
        const std::int32_t parent = tree.parent(j);
        if (parent != EliminationTree::ROOT)
        {
            // T_j ends at j: its contribution stops above it.
            deltaCount[parent] -= 1;
            deltaWeight[parent] -= weightAt(j);
        }
        for (const std::int32_t neighbour : graph.neighbours(order.nodeAt(j)))
        {
            const std::int32_t i = order.positionOf(neighbour);
            std::int32_t q = NONE;
            const int leaf = classifyLeaf(i, j, state, q);
            if (leaf >= 1)
            {
                deltaCount[j] += 1;
                deltaWeight[j] += weightAt(i);
            }
            if (leaf == 2)
            {
                deltaCount[q] -= 1;
                deltaWeight[q] -= weightAt(i);
            }
        }
        if (parent != EliminationTree::ROOT)
            ancestor[static_cast<std::size_t>(j)] = parent;
    }
    // Subtree sums: positions increase towards the root, so a child is complete before it adds to its parent.
    for (std::int32_t j = 0; j < n; ++j)
    {
        const std::int32_t parent = tree.parent(j);
        if (parent != EliminationTree::ROOT)
        {
            deltaCount[parent] += deltaCount[j];
            deltaWeight[parent] += deltaWeight[j];
        }
    }
    // The sums include the diagonal (row j of T_j); keep the part below it.
    for (std::int32_t j = 0; j < n; ++j)
    {
        deltaCount[j] -= 1;
        deltaWeight[j] -= weightAt(j);
    }
}

} // namespace oos::linalg
