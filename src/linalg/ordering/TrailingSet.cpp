#include "TrailingSet.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::linalg
{

TrailingSet::TrailingSet(std::int32_t nodeCount, std::span<const std::int32_t> trailingNodes,
                         base::MemoryBudget& budget)
    : m_nodes(trailingNodes),
      m_isTrailing(budget, static_cast<std::size_t>(nodeCount < 0 ? 0 : nodeCount), "ordering: trailing set")
{
    for (std::size_t k = 0; k < trailingNodes.size(); ++k)
    {
        const std::int32_t node = trailingNodes[k];
        if (node < 0 || node >= nodeCount || m_isTrailing[static_cast<std::size_t>(node)] != 0)
            throw std::invalid_argument(
                base::concatenate({"ordering: expected distinct trailing nodes in [0, ", std::to_string(nodeCount),
                                   "), found ", std::to_string(node), " at entry ", std::to_string(k)}));
        m_isTrailing[static_cast<std::size_t>(node)] = 1;
    }
}

} // namespace oos::linalg
