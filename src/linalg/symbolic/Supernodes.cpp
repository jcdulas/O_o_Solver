#include "oos/linalg/symbolic/Supernodes.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{

Supernodes::Supernodes(base::BudgetedBuffer<std::int32_t> firstPosition, base::BudgetedBuffer<std::int32_t> parent)
    : m_firstPosition(std::move(firstPosition)), m_parent(std::move(parent))
{
    const std::size_t count = m_parent.size();
    bool isValid = m_firstPosition.size() == count + 1 && m_firstPosition[0] == 0;
    for (std::size_t s = 0; isValid && s < count; ++s)
    {
        isValid = m_firstPosition[s + 1] > m_firstPosition[s] &&
                  (m_parent[s] == ROOT ||
                   (m_parent[s] > static_cast<std::int32_t>(s) && m_parent[s] < static_cast<std::int32_t>(count)));
    }
    if (!isValid)
        throw std::invalid_argument(
            base::concatenate({"Supernodes: expected increasing first positions from 0 and parents after their "
                               "children, for ",
                               std::to_string(count), " supernodes"}));
}

Supernodes Supernodes::findFundamental(const EliminationTree& tree, const ColumnCounts& counts,
                                       std::int32_t trailingStart, base::MemoryBudget& budget)
{
    const std::int32_t n = tree.size();
    const auto size = static_cast<std::size_t>(n);
    // The number of children of each position: a position with two children starts a new supernode.
    base::BudgetedBuffer<std::int32_t> childCount(budget, size, "supernodes: child counts");
    for (std::int32_t j = 0; j < n; ++j)
    {
        if (tree.parent(j) != EliminationTree::ROOT)
            ++childCount[static_cast<std::size_t>(tree.parent(j))];
    }
    base::BudgetedBuffer<std::int32_t> supernodeOf(budget, size, "supernodes: supernode of each position");
    std::int32_t count = 0;
    for (std::int32_t j = 0; j < n; ++j)
    {
        const bool isTrailing = j >= trailingStart;
        bool continuesPrevious = false;
        if (j > 0 && isTrailing)
            continuesPrevious = j > trailingStart; // the trailing positions are one supernode
        else if (j > 0)
            continuesPrevious = tree.parent(j - 1) == j && childCount[static_cast<std::size_t>(j)] == 1 &&
                                counts.belowNodeCount(j - 1) == counts.belowNodeCount(j) + 1;
        if (!continuesPrevious)
            ++count;
        supernodeOf[static_cast<std::size_t>(j)] = count - 1;
    }
    base::BudgetedBuffer<std::int32_t> firstPosition(budget, static_cast<std::size_t>(count) + 1,
                                                     "supernodes: first positions");
    base::BudgetedBuffer<std::int32_t> parent(budget, static_cast<std::size_t>(count), "supernodes: parents");
    for (std::int32_t j = n - 1; j >= 0; --j)
        firstPosition[static_cast<std::size_t>(supernodeOf[static_cast<std::size_t>(j)])] = j;
    firstPosition[static_cast<std::size_t>(count)] = n;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t last = firstPosition[static_cast<std::size_t>(s) + 1] - 1;
        const std::int32_t parentPosition = tree.parent(last);
        parent[static_cast<std::size_t>(s)] =
            parentPosition == EliminationTree::ROOT ? ROOT : supernodeOf[static_cast<std::size_t>(parentPosition)];
    }
    return Supernodes(std::move(firstPosition), std::move(parent));
}

} // namespace oos::linalg
