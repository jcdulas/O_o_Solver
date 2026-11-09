#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// An elimination order of the nodes of a graph: position k holds the node eliminated k-th (nodeAt), and positionOf
// is the inverse map. Every ordering returns one; the symbolic analysis works in positions.
//
// Invariant (checked at construction): nodeAt is a permutation of [0, size).
//
// Thread-safety: immutable once built. Ownership: owns both maps (budgeted); the budget must outlive it. Movable, not
// copyable.
class NodePermutation
{
public:
    NodePermutation() = default;
    // Throws std::invalid_argument when `nodeAtPosition` is not a permutation of [0, size).
    NodePermutation(std::span<const std::int32_t> nodeAtPosition, base::MemoryBudget& budget);
    // Takes the buffer over (no copy); throws as above.
    NodePermutation(base::BudgetedBuffer<std::int32_t> nodeAtPosition, base::MemoryBudget& budget);

    [[nodiscard]] static NodePermutation identity(std::int32_t size, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t size() const noexcept { return static_cast<std::int32_t>(m_nodeAt.size()); }
    [[nodiscard]] std::int32_t nodeAt(std::int32_t position) const noexcept
    {
        return m_nodeAt[static_cast<std::size_t>(position)];
    }
    [[nodiscard]] std::int32_t positionOf(std::int32_t node) const noexcept
    {
        return m_positionOf[static_cast<std::size_t>(node)];
    }
    [[nodiscard]] std::span<const std::int32_t> nodes() const noexcept { return m_nodeAt.span(); }
    [[nodiscard]] std::span<const std::int32_t> positions() const noexcept { return m_positionOf.span(); }

private:
    base::BudgetedBuffer<std::int32_t> m_nodeAt;
    base::BudgetedBuffer<std::int32_t> m_positionOf;
};

} // namespace oos::linalg
