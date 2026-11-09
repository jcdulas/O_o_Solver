#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"

#include <cstdint>

namespace oos::assembly
{

// The node couplings that the elimination of the m set adds to the stiffness: for every element group holding a node
// with an m-set DOF, the group extended with the nodes of the Gm rows of those DOFs. With them in the node graph, the
// pattern of Kgg holds every term of Knn (MsetElimination::reduce), so both matrices share one SparsePattern.
//
// Thread-safety: immutable. Ownership: owns its groups (budgeted).
class MsetCouplings
{
public:
    MsetCouplings(linalg::NodeGroupList elementGroups, const RigidElimination& gm, const DofMap& dofMap,
                  base::MemoryBudget& budget);

    [[nodiscard]] linalg::NodeGroupList groups() const noexcept { return {m_offsets.span(), m_nodes.span()}; }

private:
    base::BudgetedBuffer<std::int64_t> m_offsets;
    base::BudgetedBuffer<std::int32_t> m_nodes;
};

} // namespace oos::assembly
