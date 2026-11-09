#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The symbolic analysis of a symmetric matrix on a node graph: the final elimination order, the
// supernodes and their row structures, and the exact size, work and memory of the numeric factorisation, all computed
// before any numeric allocation: they are the memory budget's estimate and the plan's cost report.
//
// Steps: the elimination tree of the given order; a postorder of its interior part (an equivalent order: same fill,
// subtrees contiguous) with the trailing nodes kept last in their order; the column counts (Gilbert, Ng and Peyton);
// the fundamental supernodes, the trailing nodes forming the last one; the relaxed amalgamation; the row structure of
// each supernode, R(s) = (adjacency of its nodes ∪ R(children)) beyond its last position, of exactly
// belowNodeCount(last) nodes (a check of the counts).
//
// DOF numbering: the nodes in the final order, the DOFs of a node consecutive (a node of weight w owns w columns). A
// DofSelection built with nodeOrder() (mapped to the caller's nodes) numbers the equations the same way, so the
// SymmetricCsc it produces is in the factor's order.
//
// Storage model of the numeric factor: supernode s is a dense panel of columnCount(s) columns over
// columnCount(s) + rowCount(s) rows, the rows being its own columns then rows(s), column-major. panelValueCount() is
// the exact number of values of these panels, factorEntryCount() the entries of their lower trapezoids (the factor
// with the explicit zeros of the amalgamation), exactEntryCount() the entries of L without them.
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted). Movable, not copyable.
class SymbolicFactor
{
public:
    SymbolicFactor() = default;

    // nodeWeights: the DOFs of each node (>= 1), indexed by node. ordering: the elimination order, whose last
    // trailingNodeCount positions are the trailing set (kept last, in that order). Throws std::invalid_argument on a
    // size mismatch, a weight below 1 or a trailing count out of [0, n]; MemoryBudgetError when it does not fit.
    // control: checked between the steps and at every supernode (OperationCancelledError); its counter is advanced by
    // the nodes whose row structure is built, the node count in all.
    [[nodiscard]] static SymbolicFactor analyse(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                                const NodePermutation& ordering, std::int32_t trailingNodeCount,
                                                const AmalgamationThresholds& thresholds, base::MemoryBudget& budget,
                                                const base::OperationControl& control = {});

    // The final elimination order of the nodes (the given order, postordered).
    [[nodiscard]] const NodePermutation& nodeOrder() const noexcept { return m_nodeOrder; }
    [[nodiscard]] std::int32_t nodeCount() const noexcept { return m_nodeOrder.size(); }
    [[nodiscard]] std::int32_t dofCount() const noexcept
    {
        return m_firstColumnOfPosition[static_cast<std::size_t>(nodeCount())];
    }
    [[nodiscard]] std::int32_t trailingDofCount() const noexcept { return m_trailingDofCount; }
    // The first DOF column of the node at a position.
    [[nodiscard]] std::int32_t firstColumnOfPosition(std::int32_t position) const noexcept
    {
        return m_firstColumnOfPosition[static_cast<std::size_t>(position)];
    }

    [[nodiscard]] std::int32_t supernodeCount() const noexcept { return m_supernodes.count(); }
    // The node positions of supernode s, [firstPosition(s), lastPosition(s)], and its parent (Supernodes::ROOT).
    [[nodiscard]] const Supernodes& supernodes() const noexcept { return m_supernodes; }
    [[nodiscard]] std::int32_t firstColumn(std::int32_t supernode) const noexcept
    {
        return firstColumnOfPosition(m_supernodes.firstPosition(supernode));
    }
    [[nodiscard]] std::int32_t columnCount(std::int32_t supernode) const noexcept
    {
        return firstColumnOfPosition(m_supernodes.lastPosition(supernode) + 1) - firstColumn(supernode);
    }
    // The DOF rows of supernode s below its own columns, ascending.
    [[nodiscard]] std::span<const std::int32_t> rows(std::int32_t supernode) const noexcept;
    [[nodiscard]] std::int32_t rowCount(std::int32_t supernode) const noexcept
    {
        return static_cast<std::int32_t>(rows(supernode).size());
    }
    // The node positions of supernode s below its own nodes, ascending.
    [[nodiscard]] std::span<const std::int32_t> rowPositions(std::int32_t supernode) const noexcept;
    // The trailing supernode (the last one) when trailingDofCount() > 0.
    [[nodiscard]] bool isTrailing(std::int32_t supernode) const noexcept
    {
        return m_trailingDofCount > 0 && supernode == supernodeCount() - 1;
    }

    // Values of the dense panels, Σ c·(c + r): what the numeric factor allocates.
    [[nodiscard]] std::int64_t panelValueCount() const noexcept { return m_panelValueCount; }
    // Entries of the lower trapezoids, Σ c(c + 1)/2 + c·r: the factor with the explicit zeros of the amalgamation.
    [[nodiscard]] std::int64_t factorEntryCount() const noexcept { return m_factorEntryCount; }
    // Entries of L without amalgamation (diagonal included), from the column counts.
    [[nodiscard]] std::int64_t exactEntryCount() const noexcept { return m_exactEntryCount; }
    // Floating-point operations of the factorisation of the panels: per column with m entries below the diagonal,
    // m(m + 1) for the rank-one update (m(m + 1)/2 multiply-adds) and 2m for the products W = L·D and the divisions.
    [[nodiscard]] double flopCount() const noexcept { return m_flopCount; }
    // The largest front, (c + r)², in values.
    [[nodiscard]] std::int64_t largestFrontValues() const noexcept { return m_largestFrontValues; }
    // The peak of the update matrices of a multifrontal factorisation in this postorder (Liu's stack: the front of s,
    // (c + r)², allocated while the r_c² updates of its children are on the stack), in values. It bounds the working
    // storage of a left-looking driver too (its largest update block is at most a front).
    [[nodiscard]] std::int64_t updateStackPeakValues() const noexcept { return m_updateStackPeakValues; }
    // The memory estimate of the numeric factorisation: the panels, the update stack and the row indices, in bytes.
    [[nodiscard]] std::uint64_t memoryPeakBytes() const noexcept;

private:
    NodePermutation m_nodeOrder;
    Supernodes m_supernodes;
    base::BudgetedBuffer<std::int32_t> m_firstColumnOfPosition; // nodeCount + 1
    base::BudgetedBuffer<std::int64_t> m_rowPositionStart;      // supernodeCount + 1
    base::BudgetedBuffer<std::int32_t> m_rowPositions;
    base::BudgetedBuffer<std::int64_t> m_rowStart; // supernodeCount + 1
    base::BudgetedBuffer<std::int32_t> m_rows;
    std::int32_t m_trailingDofCount = 0;
    std::int64_t m_panelValueCount = 0;
    std::int64_t m_factorEntryCount = 0;
    std::int64_t m_exactEntryCount = 0;
    double m_flopCount = 0.0;
    std::int64_t m_largestFrontValues = 0;
    std::int64_t m_updateStackPeakValues = 0;
};

} // namespace oos::linalg
