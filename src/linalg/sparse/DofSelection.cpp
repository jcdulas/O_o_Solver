#include "oos/linalg/sparse/DofSelection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/ids/EntityId.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

// A node has 1 (scalar point) to 6 (grid point) DOFs, as SparsePattern requires.
constexpr int MAX_NODE_DOF_COUNT = 6;

// The node count as an int32 (nodes are numbered in 32 bits), refused beyond.
std::int32_t nodeCountOf(std::span<const std::uint8_t> nodeDofCounts)
{
    if (nodeDofCounts.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument(base::concatenate({"DofSelection: ", std::to_string(nodeDofCounts.size()),
                                                       " nodes, more than the 2^31 - 1 a node index "
                                                       "holds; condense the model into more units"}));
    return static_cast<std::int32_t>(nodeDofCounts.size());
}

} // namespace

DofSelection::DofSelection(std::span<const std::uint8_t> nodeDofCounts, std::span<const std::uint8_t> isEliminated,
                           std::span<const std::int32_t> nodeOrder, base::MemoryBudget& budget)
    : m_nodeCount(nodeCountOf(nodeDofCounts))
{
    const std::size_t count = nodeDofCounts.size();
    m_firstDof = base::BudgetedBuffer<std::int64_t>(budget, count + 1, "DOF selection: first DOFs");
    for (std::size_t n = 0; n < count; ++n)
    {
        if (nodeDofCounts[n] < 1 || nodeDofCounts[n] > MAX_NODE_DOF_COUNT)
            throw std::invalid_argument(base::concatenate(
                {"DofSelection: expected 1 to 6 DOFs per node, found ", std::to_string(nodeDofCounts[n]), " at node ",
                 std::to_string(n), "; give the DOF counts of the pattern (SparsePattern)"}));
        m_firstDof[n + 1] = m_firstDof[n] + nodeDofCounts[n];
    }
    const auto totalDofs = static_cast<std::size_t>(m_firstDof[count]);
    if (!isEliminated.empty() && isEliminated.size() != totalDofs)
        throw std::invalid_argument(
            base::concatenate({"DofSelection: expected one elimination flag per DOF (", std::to_string(totalDofs),
                               "), found ", std::to_string(isEliminated.size())}));
    if (!nodeOrder.empty() && nodeOrder.size() > count)
        throw std::invalid_argument(base::concatenate({"DofSelection: the node order lists ",
                                                       std::to_string(nodeOrder.size()), " nodes, more than the ",
                                                       std::to_string(count), " nodes; give each node at most once"}));
    const auto isKept = [&](std::size_t dof) { return isEliminated.empty() || isEliminated[dof] == 0; };
    std::size_t keptCount = 0;
    for (std::size_t dof = 0; dof < totalDofs; ++dof)
        keptCount += isKept(dof) ? 1 : 0;
    if (keptCount > static_cast<std::size_t>(base::MAX_DOF_COUNT))
        throw std::invalid_argument(
            base::concatenate({"DofSelection: ", std::to_string(keptCount),
                               " equations, more than a DofIndex holds; condense the model in units"}));

    m_equationOfDof = base::BudgetedBuffer<std::int32_t>(budget, totalDofs, "DOF selection: equations");
    m_keptDofs = base::BudgetedBuffer<std::int64_t>(budget, keptCount, "DOF selection: kept DOFs");
    std::fill_n(m_equationOfDof.data(), totalDofs, ELIMINATED);
    std::int32_t next = 0;
    const auto numberNode = [&](std::size_t node)
    {
        for (auto dof = static_cast<std::size_t>(m_firstDof[node]);
             dof < static_cast<std::size_t>(m_firstDof[node + 1]); ++dof)
        {
            if (!isKept(dof))
                continue;
            m_keptDofs[static_cast<std::size_t>(next)] = static_cast<std::int64_t>(dof);
            m_equationOfDof[dof] = next++;
        }
    };
    if (nodeOrder.empty())
    {
        for (std::size_t node = 0; node < count; ++node)
            numberNode(node);
        return;
    }
    base::BudgetedBuffer<std::uint8_t> isListed(budget, count, "DOF selection: listed nodes");
    for (std::size_t k = 0; k < nodeOrder.size(); ++k)
    {
        const std::int32_t node = nodeOrder[k];
        if (node < 0 || node >= m_nodeCount || isListed[static_cast<std::size_t>(node)] != 0)
            throw std::invalid_argument(base::concatenate({"DofSelection: expected distinct nodes in [0, ",
                                                           std::to_string(m_nodeCount), ") in the node order, found ",
                                                           std::to_string(node), " at entry ", std::to_string(k)}));
        isListed[static_cast<std::size_t>(node)] = 1;
        numberNode(static_cast<std::size_t>(node));
    }
    if (static_cast<std::size_t>(next) != keptCount)
        throw std::invalid_argument(base::concatenate({"DofSelection: the node order leaves out nodes that keep DOFs (",
                                                       std::to_string(keptCount - next),
                                                       " kept DOFs not numbered); list every node with a kept DOF"}));
}

} // namespace oos::linalg
