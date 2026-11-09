#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/symbolic/ColumnCounts.hpp"
#include "oos/linalg/symbolic/EliminationTree.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// A partition of the positions of a postordered elimination into supernodes: consecutive positions eliminated as one
// dense block, supernode s covering [firstPosition(s), firstPosition(s + 1)), with the tree of the supernodes (the
// parent of s is the supernode of the parent position of its last position). Children come before their parent.
//
// A fundamental supernode (Liu, Ng and Peyton, "On finding supernodes for sparse matrix computations", SIAM J. Matrix
// Anal. Appl. 14, 1993) groups position j + 1 with j when j + 1 is the parent and only child-holder of j and the
// columns have the same structure below: belowNodes(j) = belowNodes(j + 1) + 1. Its columns then have no zero in the
// dense trapezoid. The trailing positions, when there are some, form one supernode of their own, last: the block a
// partial factorisation leaves as the Schur complement.
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted). Movable, not copyable.
class Supernodes
{
public:
    static constexpr std::int32_t ROOT = -1;

    Supernodes() = default;
    // firstPosition: count + 1 entries from 0 to the position count, increasing; parent: count entries, each ROOT or
    // a later supernode. Throws std::invalid_argument otherwise.
    Supernodes(base::BudgetedBuffer<std::int32_t> firstPosition, base::BudgetedBuffer<std::int32_t> parent);

    // The fundamental supernodes of a postordered tree (position k's parent is above k and each subtree is contiguous);
    // the positions from trailingStart on form one last supernode (trailingStart = size: none).
    [[nodiscard]] static Supernodes findFundamental(const EliminationTree& tree, const ColumnCounts& counts,
                                                    std::int32_t trailingStart, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t count() const noexcept { return static_cast<std::int32_t>(m_parent.size()); }
    [[nodiscard]] std::int32_t firstPosition(std::int32_t supernode) const noexcept
    {
        return m_firstPosition[static_cast<std::size_t>(supernode)];
    }
    [[nodiscard]] std::int32_t lastPosition(std::int32_t supernode) const noexcept
    {
        return m_firstPosition[static_cast<std::size_t>(supernode) + 1] - 1;
    }
    [[nodiscard]] std::int32_t parent(std::int32_t supernode) const noexcept
    {
        return m_parent[static_cast<std::size_t>(supernode)];
    }

private:
    base::BudgetedBuffer<std::int32_t> m_firstPosition;
    base::BudgetedBuffer<std::int32_t> m_parent;
};

} // namespace oos::linalg
