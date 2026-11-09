#include "oos/linalg/symbolic/Amalgamation.hpp"

#include <utility>

namespace oos::linalg
{

bool Amalgamation::isMergeWorthIt(std::int64_t columns, std::int64_t zeros, std::int64_t rowsBelow) const noexcept
{
    if (columns <= m_thresholds.columnLimits[0])
        return true;
    // The entries of the merged dense trapezoid: the lower triangle of its diagonal block and the rows below.
    const double entries = static_cast<double>(columns) * static_cast<double>(columns + 1) / 2.0 +
                           static_cast<double>(columns) * static_cast<double>(rowsBelow);
    const double fraction = static_cast<double>(zeros) / entries;
    return (columns <= m_thresholds.columnLimits[1] && fraction < m_thresholds.zeroFractions[0]) ||
           (columns <= m_thresholds.columnLimits[2] && fraction < m_thresholds.zeroFractions[1]) ||
           fraction < m_thresholds.zeroFractions[2];
}

Supernodes Amalgamation::relax(const Supernodes& fundamental, std::span<const std::int32_t> positionWeight,
                               const ColumnCounts& counts, bool hasTrailingSupernode, base::MemoryBudget& budget) const
{
    const std::int32_t count = fundamental.count();
    const auto size = static_cast<std::size_t>(count);
    // Per group (identified by its top fundamental supernode): columns and zeros so far; rows below are those of the
    // top. groupTop[s]: the top of the group of s.
    base::BudgetedBuffer<std::int64_t> columns(budget, size, "amalgamation: columns");
    base::BudgetedBuffer<std::int64_t> rowsBelow(budget, size, "amalgamation: rows below");
    base::BudgetedBuffer<std::int64_t> zeros(budget, size, "amalgamation: zeros");
    base::BudgetedBuffer<std::int32_t> groupTop(budget, size, "amalgamation: groups");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto k = static_cast<std::size_t>(s);
        for (std::int32_t p = fundamental.firstPosition(s); p <= fundamental.lastPosition(s); ++p)
            columns[k] += positionWeight[static_cast<std::size_t>(p)];
        rowsBelow[k] = counts.belowWeight(fundamental.lastPosition(s));
        groupTop[k] = s;
    }
    const std::int32_t lastMergeable = hasTrailingSupernode ? count - 1 : count;
    // From the top down: when s is considered, the group of s + 1 starts at s + 1 (only lower supernodes join later).
    for (std::int32_t s = count - 2; s >= 0; --s)
    {
        const std::int32_t parent = fundamental.parent(s);
        if (parent != s + 1 || parent >= lastMergeable)
            continue;
        const auto child = static_cast<std::size_t>(s);
        const auto top = static_cast<std::size_t>(groupTop[static_cast<std::size_t>(parent)]);
        const std::int64_t mergedColumns = columns[child] + columns[top];
        const std::int64_t mergedZeros =
            zeros[child] + zeros[top] + columns[child] * (columns[top] + rowsBelow[top] - rowsBelow[child]);
        if (!isMergeWorthIt(mergedColumns, mergedZeros, rowsBelow[top]))
            continue;
        columns[top] = mergedColumns;
        zeros[top] = mergedZeros;
        groupTop[child] = static_cast<std::int32_t>(top);
    }
    // The groups are ranges of consecutive fundamental supernodes; number them in order.
    base::BudgetedBuffer<std::int32_t> groupIndex(budget, size, "amalgamation: group index");
    std::int32_t groups = 0;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const bool isNewGroup =
            s == 0 || groupTop[static_cast<std::size_t>(s)] != groupTop[static_cast<std::size_t>(s) - 1];
        if (isNewGroup)
            ++groups;
        groupIndex[static_cast<std::size_t>(s)] = groups - 1;
    }
    base::BudgetedBuffer<std::int32_t> firstPosition(budget, static_cast<std::size_t>(groups) + 1,
                                                     "supernodes: first positions");
    base::BudgetedBuffer<std::int32_t> parent(budget, static_cast<std::size_t>(groups), "supernodes: parents");
    for (std::int32_t s = count - 1; s >= 0; --s)
        firstPosition[static_cast<std::size_t>(groupIndex[static_cast<std::size_t>(s)])] = fundamental.firstPosition(s);
    firstPosition[static_cast<std::size_t>(groups)] = count == 0 ? 0 : fundamental.lastPosition(count - 1) + 1;
    for (std::int32_t s = 0; s < count; ++s)
    {
        if (groupTop[static_cast<std::size_t>(s)] != s)
            continue; // only the top of a group decides the group's parent
        const std::int32_t fundamentalParent = fundamental.parent(s);
        parent[static_cast<std::size_t>(groupIndex[static_cast<std::size_t>(s)])] =
            fundamentalParent == Supernodes::ROOT ? Supernodes::ROOT
                                                  : groupIndex[static_cast<std::size_t>(fundamentalParent)];
    }
    return Supernodes(std::move(firstPosition), std::move(parent));
}

} // namespace oos::linalg
