#include "oos/assembly/constraints/MsetCouplings.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace oos::assembly
{

MsetCouplings::MsetCouplings(linalg::NodeGroupList elementGroups, const RigidElimination& gm, const DofMap& dofMap,
                             base::MemoryBudget& budget)
{
    // The nodes of the Gm rows of each node with m DOFs, gathered once per node (bounded by the rows of that node).
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    base::BudgetedBuffer<std::uint8_t> hasDependent(budget, nodeCount, "m-set couplings");
    for (const base::DofIndex dof : gm.mDofs())
        hasDependent[static_cast<std::size_t>(dofMap.nodeDofOf(dof).node)] = 1;

    // Two passes: count, then fill (the sizes come from the groups and the rows).
    std::vector<std::int32_t> group; // bounded by an element and its Gm rows
    const auto extendedGroup = [&](std::span<const std::int32_t> nodes) -> bool
    {
        bool touches = false;
        for (const std::int32_t node : nodes)
            touches = touches || hasDependent[static_cast<std::size_t>(node)] != 0;
        if (!touches)
            return false;
        group.assign(nodes.begin(), nodes.end());
        for (const std::int32_t node : nodes)
        {
            if (hasDependent[static_cast<std::size_t>(node)] == 0)
                continue;
            const int components = node < dofMap.gridCount() ? 6 : 1;
            const base::DofIndex first =
                node < dofMap.gridCount() ? dofMap.gridDof(node, 1) : dofMap.scalarDof(node - dofMap.gridCount());
            for (int c = 0; c < components; ++c)
            {
                const std::optional<std::size_t> row = gm.findRow(first + c);
                if (!row)
                    continue;
                for (const ConstraintTerm& term : gm.row(*row))
                    group.push_back(dofMap.nodeDofOf(term.dof).node);
            }
        }
        std::sort(group.begin(), group.end());
        group.erase(std::unique(group.begin(), group.end()), group.end());
        return true;
    };

    // The groups: each element group touching a dependent node, then each dependent node alone (its own diagonal
    // block, K(d, e) of two DOFs of one node, couples the nodes of their rows even when no element group of two nodes
    // or more holds it: a grounded spring or a mass element on a dependent point).
    const std::size_t groupTotal = elementGroups.groupCount() + nodeCount;
    std::int32_t single = 0;
    const auto visit = [&](std::size_t g) -> bool
    {
        if (g < elementGroups.groupCount())
            return extendedGroup(elementGroups.group(g));
        single = static_cast<std::int32_t>(g - elementGroups.groupCount());
        if (hasDependent[static_cast<std::size_t>(single)] == 0)
            return false;
        return extendedGroup(std::span<const std::int32_t>(&single, 1));
    };

    std::size_t groupCount = 0;
    std::size_t nodeTotal = 0;
    for (std::size_t g = 0; g < groupTotal; ++g)
    {
        if (visit(g))
        {
            ++groupCount;
            nodeTotal += group.size();
        }
    }
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, groupCount + 1, "m-set couplings");
    m_nodes = base::BudgetedBuffer<std::int32_t>(budget, nodeTotal, "m-set couplings");
    std::size_t written = 0;
    std::size_t position = 0;
    for (std::size_t g = 0; g < groupTotal; ++g)
    {
        if (!visit(g))
            continue;
        std::copy(group.begin(), group.end(), m_nodes.data() + position);
        position += group.size();
        m_offsets[++written] = static_cast<std::int64_t>(position);
    }
}

} // namespace oos::assembly
