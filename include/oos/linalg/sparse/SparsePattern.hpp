#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The symbolic pattern of a symmetric matrix assembled at node granularity: the lower triangle of the
// node blocks, built once from the node graph (connectivity plus extra couplings) and shared by every matrix on it
// (stiffness, mass, differential stiffness). Block (i, j), i >= j, couples the DOFs of node i (rows) with those of node
// j (columns); it is dofCount(i) × dofCount(j), stored column-major at blockValueOffset(block) in the values of a
// BlockSparseMatrix. A node has 6 DOFs (grid point) or 1 (scalar point); any count from 1 to 6 is accepted.
//
// Layout: block columns, one per node; column j holds its diagonal block (j, j) first, then the blocks (i, j), i > j,
// rows ascending. Block indices and value offsets are 64-bit.
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted); the budget must outlive the pattern.
// Movable, not copyable.
class SparsePattern
{
public:
    static constexpr int MAX_NODE_DOFS = 6;

    SparsePattern() = default;
    // Throws std::invalid_argument when nodeDofCounts has not one entry in [1, 6] per node of the graph, or the model
    // has more DOFs than a DofIndex holds; MemoryBudgetError when the pattern does not fit.
    SparsePattern(const NodeGraph& graph, std::span<const std::uint8_t> nodeDofCounts, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t nodeCount() const noexcept { return m_nodeCount; }
    // The DOFs of all nodes, node-major: node n owns [firstDof(n), firstDof(n) + nodeDofCount(n)).
    [[nodiscard]] std::int64_t dofCount() const noexcept { return m_firstDof[static_cast<std::size_t>(m_nodeCount)]; }
    [[nodiscard]] int nodeDofCount(std::int32_t node) const noexcept
    {
        return static_cast<int>(m_firstDof[static_cast<std::size_t>(node) + 1] -
                                m_firstDof[static_cast<std::size_t>(node)]);
    }
    [[nodiscard]] std::int64_t firstDof(std::int32_t node) const noexcept
    {
        return m_firstDof[static_cast<std::size_t>(node)];
    }

    [[nodiscard]] std::int64_t blockCount() const noexcept
    {
        return m_columnStart[static_cast<std::size_t>(m_nodeCount)];
    }
    // The blocks of column j are [columnBegin(j), columnBegin(j + 1)); the first is the diagonal block.
    [[nodiscard]] std::int64_t columnBegin(std::int32_t column) const noexcept
    {
        return m_columnStart[static_cast<std::size_t>(column)];
    }
    [[nodiscard]] std::int32_t blockRow(std::int64_t block) const noexcept
    {
        return m_blockRows[static_cast<std::size_t>(block)];
    }
    [[nodiscard]] std::int64_t blockValueOffset(std::int64_t block) const noexcept
    {
        return m_blockValueOffsets[static_cast<std::size_t>(block)];
    }
    // The number of values of a BlockSparseMatrix on this pattern.
    [[nodiscard]] std::int64_t valueCount() const noexcept
    {
        return m_blockValueOffsets[static_cast<std::size_t>(blockCount())];
    }

    // The index of block (row, column), row >= column, or -1 when the nodes are not coupled. O(log degree).
    [[nodiscard]] std::int64_t findBlock(std::int32_t row, std::int32_t column) const noexcept;

private:
    std::int32_t m_nodeCount = 0;
    base::BudgetedBuffer<std::int64_t> m_firstDof;          // nodeCount + 1
    base::BudgetedBuffer<std::int64_t> m_columnStart;       // nodeCount + 1
    base::BudgetedBuffer<std::int32_t> m_blockRows;         // blockCount
    base::BudgetedBuffer<std::int64_t> m_blockValueOffsets; // blockCount + 1
};

} // namespace oos::linalg
