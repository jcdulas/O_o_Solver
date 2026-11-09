#include "oos/linalg/symbolic/EliminationTree.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::linalg
{

EliminationTree::EliminationTree(const NodeGraph& graph, const NodePermutation& order, base::MemoryBudget& budget)
    : m_parent(budget, static_cast<std::size_t>(graph.nodeCount()), "elimination tree")
{
    const std::int32_t n = graph.nodeCount();
    if (order.size() != n)
        throw std::invalid_argument(
            base::concatenate({"EliminationTree: the order has ", std::to_string(order.size()), " nodes, the graph ",
                               std::to_string(n), "; give one position per node of the graph"}));
    // Liu's algorithm: for each column k, every earlier column i adjacent to it climbs to the root of its current
    // subtree, which becomes a child of k; the ancestors are compressed to k on the way.
    base::BudgetedBuffer<std::int32_t> ancestor(budget, static_cast<std::size_t>(n), "elimination tree: ancestors");
    std::fill_n(m_parent.data(), m_parent.size(), ROOT);
    std::fill_n(ancestor.data(), ancestor.size(), ROOT);
    for (std::int32_t k = 0; k < n; ++k)
    {
        for (const std::int32_t neighbour : graph.neighbours(order.nodeAt(k)))
        {
            std::int32_t i = order.positionOf(neighbour);
            while (i != ROOT && i < k)
            {
                const std::int32_t next = ancestor[static_cast<std::size_t>(i)];
                ancestor[static_cast<std::size_t>(i)] = k;
                if (next == ROOT)
                    m_parent[static_cast<std::size_t>(i)] = k;
                i = next;
            }
        }
    }
}

base::BudgetedBuffer<std::int32_t> EliminationTree::postorder(base::MemoryBudget& budget) const
{
    const std::size_t n = m_parent.size();
    base::BudgetedBuffer<std::int32_t> head(budget, n, "postorder: first children");
    base::BudgetedBuffer<std::int32_t> next(budget, n, "postorder: next siblings");
    base::BudgetedBuffer<std::int32_t> stack(budget, n, "postorder: stack");
    base::BudgetedBuffer<std::int32_t> post(budget, n, "postorder");
    std::fill_n(head.data(), n, ROOT);
    // Children lists built from the last position down, so that each list is in increasing position.
    for (std::size_t j = n; j-- > 0;)
    {
        const std::int32_t parent = m_parent[j];
        if (parent == ROOT)
            continue;
        next[j] = head[static_cast<std::size_t>(parent)];
        head[static_cast<std::size_t>(parent)] = static_cast<std::int32_t>(j);
    }
    std::size_t k = 0;
    for (std::size_t root = 0; root < n; ++root)
    {
        if (m_parent[root] != ROOT)
            continue;
        std::size_t top = 0;
        stack[0] = static_cast<std::int32_t>(root);
        while (true)
        {
            const std::int32_t p = stack[top];
            const std::int32_t child = head[static_cast<std::size_t>(p)];
            if (child == ROOT)
            {
                post[k++] = p;
                if (top == 0)
                    break;
                --top;
            }
            else
            {
                head[static_cast<std::size_t>(p)] = next[static_cast<std::size_t>(child)];
                stack[++top] = child;
            }
        }
    }
    return post;
}

} // namespace oos::linalg
