#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The map from the node-major DOFs of a pattern to the equations of a scalar system: the eliminated DOFs (constrained,
// dependent, or outside the set being solved) are removed and the others renumbered. The kept DOFs are
// numbered node by node in a given node order (the fill-reducing ordering of the nodes, expanded to their DOFs), and
// by increasing component within a node, so that the DOFs of a node stay consecutive: the supervariable the orderings
// and the symbolic analysis work on.
//
// Thread-safety: immutable once built. Ownership: owns its maps (budgeted); the budget must outlive it. Movable, not
// copyable.
class DofSelection
{
public:
    static constexpr std::int32_t ELIMINATED = -1;

    DofSelection() = default;
    // nodeDofCounts: the DOF count of each node (1 to 6). isEliminated: one flag per node-major DOF (nonzero when the
    // DOF is removed), or empty when every DOF is kept. nodeOrder: the nodes in the order of the equations, each at
    // most once; a node absent from it must have every DOF eliminated (it was dropped from the ordered graph). Empty
    // means the natural order 0, 1, 2, ...
    // Throws std::invalid_argument when the sizes disagree, a DOF count is outside 1 to 6, there are more than 2^31 - 1
    // nodes, a node is repeated or out of range, or an absent node keeps a DOF.
    DofSelection(std::span<const std::uint8_t> nodeDofCounts, std::span<const std::uint8_t> isEliminated,
                 std::span<const std::int32_t> nodeOrder, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t nodeCount() const noexcept { return m_nodeCount; }
    // The node-major DOFs of all nodes, kept or not.
    [[nodiscard]] std::int64_t totalDofCount() const noexcept
    {
        return m_firstDof[static_cast<std::size_t>(m_nodeCount)];
    }
    [[nodiscard]] std::int32_t keptDofCount() const noexcept { return static_cast<std::int32_t>(m_keptDofs.size()); }

    // The equation of a component (from 0) of a node, or ELIMINATED.
    [[nodiscard]] std::int32_t equation(std::int32_t node, int component) const noexcept
    {
        return m_equationOfDof[static_cast<std::size_t>(m_firstDof[static_cast<std::size_t>(node)] + component)];
    }
    // The equation of a node-major DOF, or ELIMINATED.
    [[nodiscard]] std::int32_t equationOfDof(std::int64_t dof) const noexcept
    {
        return m_equationOfDof[static_cast<std::size_t>(dof)];
    }
    // The node-major DOF of an equation.
    [[nodiscard]] std::int64_t dofOfEquation(std::int32_t equation) const noexcept
    {
        return m_keptDofs[static_cast<std::size_t>(equation)];
    }
    [[nodiscard]] std::int64_t firstDof(std::int32_t node) const noexcept
    {
        return m_firstDof[static_cast<std::size_t>(node)];
    }

private:
    std::int32_t m_nodeCount = 0;
    base::BudgetedBuffer<std::int64_t> m_firstDof;      // nodeCount + 1
    base::BudgetedBuffer<std::int32_t> m_equationOfDof; // one per node-major DOF
    base::BudgetedBuffer<std::int64_t> m_keptDofs;      // one per equation
};

} // namespace oos::linalg
