#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The set of nodes an ordering must eliminate last, in the order given (the boundary of a unit, whose Schur
// complement a partial factorisation keeps). Internal to the orderings.
//
// Thread-safety: immutable once built. Ownership: owns its mask (budgeted); the list is a view of the caller's.
class TrailingSet
{
public:
    // Throws std::invalid_argument when a node is out of [0, nodeCount) or repeated.
    TrailingSet(std::int32_t nodeCount, std::span<const std::int32_t> trailingNodes, base::MemoryBudget& budget);

    [[nodiscard]] bool contains(std::int32_t node) const noexcept
    {
        return m_isTrailing[static_cast<std::size_t>(node)] != 0;
    }
    [[nodiscard]] std::span<const std::int32_t> nodes() const noexcept { return m_nodes; }
    [[nodiscard]] std::int32_t count() const noexcept { return static_cast<std::int32_t>(m_nodes.size()); }

private:
    std::span<const std::int32_t> m_nodes;
    base::BudgetedBuffer<std::uint8_t> m_isTrailing;
};

} // namespace oos::linalg
