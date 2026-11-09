#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>

namespace oos::assembly
{

// The rows of a load table by load set (the `setId` column of loadsTable, pressureTable, lineLoadsTable): a deck of
// thousands of load cases has its rows spread over thousands of sets, and every set is read several times (its loads,
// its compact columns for the load basis), so the rows of a set are found by a binary search instead of a pass over
// the table. The rows of a set keep the table order (the order of every sum of the loads).
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted, one entry per table row).
class LoadSetIndex
{
public:
    LoadSetIndex() = default;
    LoadSetIndex(std::span<const std::int32_t> setIds, base::MemoryBudget& budget)
        : m_setIds(budget, setIds.size(), "load set index"), m_rows(budget, setIds.size(), "load set index rows")
    {
        std::iota(m_rows.data(), m_rows.data() + setIds.size(), std::int64_t{0});
        std::stable_sort(m_rows.data(), m_rows.data() + setIds.size(), [&](std::int64_t a, std::int64_t b)
                         { return setIds[static_cast<std::size_t>(a)] < setIds[static_cast<std::size_t>(b)]; });
        for (std::size_t k = 0; k < setIds.size(); ++k)
            m_setIds[k] = setIds[static_cast<std::size_t>(m_rows[k])];
    }

    // Calls visit(setId, rows) for each load set of the table, by increasing ID, its rows increasing.
    template <typename Visit>
    void forEachSet(Visit&& visit) const
    {
        std::size_t first = 0;
        while (first < m_setIds.size())
        {
            std::size_t last = first + 1;
            while (last < m_setIds.size() && m_setIds[last] == m_setIds[first])
                ++last;
            visit(m_setIds[first], m_rows.span().subspan(first, last - first));
            first = last;
        }
    }

    // The rows of load set `setId`, increasing.
    [[nodiscard]] std::span<const std::int64_t> rowsOf(std::int32_t setId) const
    {
        const std::int32_t* begin = m_setIds.data();
        const std::int32_t* end = begin + m_setIds.size();
        const std::int32_t* first = std::lower_bound(begin, end, setId);
        const std::int32_t* last = std::upper_bound(first, end, setId);
        return m_rows.span().subspan(static_cast<std::size_t>(first - begin), static_cast<std::size_t>(last - first));
    }

private:
    base::BudgetedBuffer<std::int32_t> m_setIds;
    base::BudgetedBuffer<std::int64_t> m_rows;
};

} // namespace oos::assembly
