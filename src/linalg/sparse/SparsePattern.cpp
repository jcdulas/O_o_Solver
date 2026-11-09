#include "oos/linalg/sparse/SparsePattern.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/ids/EntityId.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::linalg
{

SparsePattern::SparsePattern(const NodeGraph& graph, std::span<const std::uint8_t> nodeDofCounts,
                             base::MemoryBudget& budget)
    : m_nodeCount(graph.nodeCount())
{
    const auto count = static_cast<std::size_t>(m_nodeCount);
    if (nodeDofCounts.size() != count)
        throw std::invalid_argument(
            base::concatenate({"SparsePattern: expected one DOF count per node (", std::to_string(count), "), found ",
                               std::to_string(nodeDofCounts.size())}));
    m_firstDof = base::BudgetedBuffer<std::int64_t>(budget, count + 1, "sparse pattern: first DOFs");
    for (std::size_t n = 0; n < count; ++n)
    {
        const int dofs = nodeDofCounts[n];
        if (dofs < 1 || dofs > MAX_NODE_DOFS)
            throw std::invalid_argument(base::concatenate({"SparsePattern: expected 1 to 6 DOFs per node, found ",
                                                           std::to_string(dofs), " at node ", std::to_string(n)}));
        m_firstDof[n + 1] = m_firstDof[n] + dofs;
    }
    if (m_firstDof[count] > base::MAX_DOF_COUNT)
        throw std::invalid_argument(base::concatenate(
            {"SparsePattern: the model has ", std::to_string(m_firstDof[count]), " DOFs, more than the ",
             std::to_string(base::MAX_DOF_COUNT), " a DofIndex holds; condense units to reduce it"}));

    // Column j: the diagonal block, then the neighbours above j (the graph lists them sorted).
    m_columnStart = base::BudgetedBuffer<std::int64_t>(budget, count + 1, "sparse pattern: column starts");
    for (std::size_t j = 0; j < count; ++j)
    {
        const auto neighbours = graph.neighbours(static_cast<std::int32_t>(j));
        const auto below =
            neighbours.end() - std::upper_bound(neighbours.begin(), neighbours.end(), static_cast<std::int32_t>(j));
        m_columnStart[j + 1] = m_columnStart[j] + 1 + below;
    }
    const auto blocks = static_cast<std::size_t>(m_columnStart[count]);
    m_blockRows = base::BudgetedBuffer<std::int32_t>(budget, blocks, "sparse pattern: block rows");
    m_blockValueOffsets = base::BudgetedBuffer<std::int64_t>(budget, blocks + 1, "sparse pattern: value offsets");
    for (std::size_t j = 0; j < count; ++j)
    {
        auto block = static_cast<std::size_t>(m_columnStart[j]);
        const auto columnDofs = static_cast<std::int64_t>(nodeDofCounts[j]);
        m_blockRows[block] = static_cast<std::int32_t>(j);
        m_blockValueOffsets[block + 1] = m_blockValueOffsets[block] + columnDofs * columnDofs;
        ++block;
        const auto neighbours = graph.neighbours(static_cast<std::int32_t>(j));
        for (auto it = std::upper_bound(neighbours.begin(), neighbours.end(), static_cast<std::int32_t>(j));
             it != neighbours.end(); ++it, ++block)
        {
            m_blockRows[block] = *it;
            m_blockValueOffsets[block + 1] =
                m_blockValueOffsets[block] + columnDofs * static_cast<std::int64_t>(nodeDofCounts[*it]);
        }
    }
}

std::int64_t SparsePattern::findBlock(std::int32_t row, std::int32_t column) const noexcept
{
    const std::int32_t* const begin = m_blockRows.data() + m_columnStart[static_cast<std::size_t>(column)];
    const std::int32_t* const end = m_blockRows.data() + m_columnStart[static_cast<std::size_t>(column) + 1];
    // The diagonal block comes first and is the smallest row, so the column is sorted.
    const std::int32_t* const found = std::lower_bound(begin, end, row);
    if (found == end || *found != row)
        return -1;
    return found - m_blockRows.data();
}

} // namespace oos::linalg
