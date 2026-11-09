#include "oos/linalg/ordering/NodePermutation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{

NodePermutation::NodePermutation(std::span<const std::int32_t> nodeAtPosition, base::MemoryBudget& budget)
    : NodePermutation(
          [&]
          {
              base::BudgetedBuffer<std::int32_t> copy(budget, nodeAtPosition.size(), "node permutation");
              std::copy(nodeAtPosition.begin(), nodeAtPosition.end(), copy.data());
              return copy;
          }(),
          budget)
{
}

NodePermutation::NodePermutation(base::BudgetedBuffer<std::int32_t> nodeAtPosition, base::MemoryBudget& budget)
    : m_nodeAt(std::move(nodeAtPosition)), m_positionOf(budget, m_nodeAt.size(), "node permutation: inverse")
{
    const std::size_t size = m_nodeAt.size();
    std::fill_n(m_positionOf.data(), size, -1);
    for (std::size_t k = 0; k < size; ++k)
    {
        const std::int32_t node = m_nodeAt[k];
        if (node < 0 || static_cast<std::size_t>(node) >= size || m_positionOf[static_cast<std::size_t>(node)] >= 0)
            throw std::invalid_argument(base::concatenate(
                {"NodePermutation: expected a permutation of [0, ", std::to_string(size), "), found node ",
                 std::to_string(node), " at position ", std::to_string(k), " (out of range or repeated)"}));
        m_positionOf[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(k);
    }
}

NodePermutation NodePermutation::identity(std::int32_t size, base::MemoryBudget& budget)
{
    base::BudgetedBuffer<std::int32_t> nodes(budget, static_cast<std::size_t>(std::max(size, 0)), "node permutation");
    for (std::size_t k = 0; k < nodes.size(); ++k)
        nodes[k] = static_cast<std::int32_t>(k);
    return NodePermutation(std::move(nodes), budget);
}

} // namespace oos::linalg
