#include "SolvePlan.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include <algorithm>

namespace oos::linalg
{
namespace
{

// Below this many subtrees the sweeps run in supernode order (FactorSolve::MIN_PARALLEL_SUBTREES).
constexpr std::size_t MIN_SUBTREES = 2;
constexpr std::int32_t UNSET = -1;

std::size_t toSize(std::int64_t value)
{
    return static_cast<std::size_t>(value);
}

} // namespace

std::uint64_t SolvePlan::estimateMemoryBytes(std::int64_t supernodeCount, std::int64_t panelRowCount,
                                             std::int64_t dimension) noexcept
{
    const auto supernodes = static_cast<std::uint64_t>(supernodeCount);
    const auto rows = static_cast<std::uint64_t>(panelRowCount);
    const auto equations = static_cast<std::uint64_t>(dimension);
    // The rows of the panels in their units, the rows of the units and their accumulator rows (each at most the panel
    // rows: a unit's row is a row of one of its panels), the outside rows (at most the panel rows); per supernode at
    // most its offset, its parent, a unit, a subtree and a range of the construction; the two equation arrays of the
    // construction.
    constexpr std::uint64_t RANGE_BYTES = 3 * sizeof(std::int32_t);
    return 4 * rows * sizeof(std::int32_t) +
           supernodes * (sizeof(std::int64_t) + sizeof(std::int32_t) + sizeof(Unit) + sizeof(Subtree) + RANGE_BYTES) +
           2 * equations * sizeof(std::int32_t) + sizeof(std::int64_t);
}

bool SolvePlan::isFor(const SupernodalFactor& factor, std::span<const std::int32_t> parents) const noexcept
{
    const auto count = static_cast<std::size_t>(factor.supernodeCount());
    if (m_parent.size() != count || parents.size() != count || m_localRowOffset.size() != count + 1)
        return false;
    const bool isByTree = factor.subtrees().size() >= MIN_SUBTREES && factor.delayedPivotCount() == 0;
    if (isByTree != m_isByTree || (m_isByTree && m_subtrees.size() != factor.subtrees().size()))
        return false;
    for (std::size_t s = 0; s < count; ++s)
    {
        if (m_parent[s] != parents[s])
            return false;
        const FactorPanel panel = factor.panel(static_cast<std::int32_t>(s));
        if (m_localRowOffset[s + 1] - m_localRowOffset[s] != static_cast<std::int64_t>(panel.rows.size()))
            return false;
    }
    return true;
}

SolvePlan::SolvePlan(const SupernodalFactor& factor, std::span<const std::int32_t> parents, base::MemoryBudget& budget)
{
    const std::int32_t count = factor.supernodeCount();
    const auto total = static_cast<std::size_t>(count);
    const auto dimension = static_cast<std::size_t>(factor.dimension());
    OOS_ASSERT(parents.size() == total);
    m_parent = base::BudgetedBuffer<std::int32_t>(budget, total, "solve plan: parents");
    std::copy(parents.begin(), parents.end(), m_parent.data());
    m_localRowOffset = base::BudgetedBuffer<std::int64_t>(budget, total + 1, "solve plan: row offsets");
    for (std::int32_t s = 0; s < count; ++s)
        m_localRowOffset[static_cast<std::size_t>(s) + 1] =
            m_localRowOffset[static_cast<std::size_t>(s)] + static_cast<std::int64_t>(factor.panel(s).rows.size());
    m_localRows = base::BudgetedBuffer<std::int32_t>(budget, toSize(m_localRowOffset[total]),
                                                     "solve plan: panel rows in their units", base::FOR_OVERWRITE);

    // The ranges the units are cut in: each subtree, and every top supernode alone (sweeps by subtrees); else all the
    // supernodes.
    const std::span<const SupernodeRange> cut = factor.subtrees();
    m_isByTree = cut.size() >= MIN_SUBTREES && factor.delayedPivotCount() == 0;
    struct Range
    {
        std::int32_t first;
        std::int32_t last;
        std::int32_t subtree;
    };
    std::vector<Range> ranges; // bounded by the supernodes
    if (m_isByTree)
    {
        std::size_t next = 0;
        std::int32_t s = 0;
        while (s < count)
        {
            if (next < cut.size() && cut[next].first == s)
            {
                ranges.push_back({cut[next].first, cut[next].last, static_cast<std::int32_t>(next)});
                s = cut[next].last + 1;
                ++next;
                continue;
            }
            ranges.push_back({s, s, NO_SUBTREE});
            ++s;
        }
    }
    else if (count > 0)
    {
        ranges.push_back({0, count - 1, NO_SUBTREE});
    }

    // Pass 1: the units, greedily (the class comment); marker[e] is the unit that last counted equation e.
    base::BudgetedBuffer<std::int32_t> marker(budget, dimension, "solve plan: equation markers", base::FOR_OVERWRITE);
    std::fill_n(marker.data(), dimension, UNSET);
    std::vector<Unit> units; // at most one per supernode
    for (const Range& range : ranges)
    {
        const bool isTopSingleton = m_isByTree && range.subtree == NO_SUBTREE;
        std::int32_t s = range.first;
        while (s <= range.last)
        {
            const auto unitIndex = static_cast<std::int32_t>(units.size());
            Unit unit;
            unit.firstSupernode = s;
            unit.subtree = range.subtree;
            while (s <= range.last)
            {
                const FactorPanel panel = factor.panel(s);
                const bool isLarge =
                    static_cast<std::int64_t>(panel.eliminatedCount()) * static_cast<std::int64_t>(panel.rows.size()) >=
                    SINGLETON_PANEL_VALUES;
                std::int32_t added = 0;
                for (const std::int32_t row : panel.rows)
                    if (marker[static_cast<std::size_t>(row)] != unitIndex)
                        ++added;
                if (s > unit.firstSupernode && (isLarge || unit.rowCount + added > UNIT_ROWS))
                    break;
                for (const std::int32_t row : panel.rows)
                    marker[static_cast<std::size_t>(row)] = unitIndex;
                unit.rowCount += added;
                unit.eliminatedRowCount += static_cast<std::int32_t>(panel.eliminatedCount());
                ++s;
                if (isTopSingleton || isLarge || unit.rowCount > UNIT_ROWS)
                    break;
            }
            unit.lastSupernode = s - 1;
            units.push_back(unit);
        }
    }
    std::int64_t unitRowTotal = 0;
    for (Unit& unit : units)
    {
        unit.firstRow = unitRowTotal;
        unitRowTotal += unit.rowCount;
        m_largestUnitRowCount = std::max(m_largestUnitRowCount, unit.rowCount);
    }
    m_unitReservation = budget.reserve(units.size() * sizeof(Unit) + cut.size() * sizeof(Subtree), "solve plan: units");
    m_units = std::move(units);
    m_unitRows =
        base::BudgetedBuffer<std::int32_t>(budget, toSize(unitRowTotal), "solve plan: unit rows", base::FOR_OVERWRITE);
    m_accumulatorRows = base::BudgetedBuffer<std::int32_t>(budget, toSize(unitRowTotal), "solve plan: accumulator rows",
                                                           base::FOR_OVERWRITE);

    // The accumulators of the subtrees: the update rows of each root, and the first equation beyond the subtree.
    std::vector<std::int32_t> subtreeEnd; // one per subtree
    if (m_isByTree)
    {
        std::int64_t outsideTotal = 0;
        for (const SupernodeRange& range : cut)
        {
            const FactorPanel root = factor.panel(range.last);
            outsideTotal += static_cast<std::int64_t>(root.rows.size() - root.eliminatedCount());
        }
        m_outsideRows = base::BudgetedBuffer<std::int32_t>(budget, toSize(outsideTotal), "solve plan: outside rows",
                                                           base::FOR_OVERWRITE);
        std::int64_t next = 0;
        for (const SupernodeRange& range : cut)
        {
            const FactorPanel root = factor.panel(range.last);
            Subtree subtree;
            subtree.firstOutsideRow = next;
            subtree.outsideRowCount = static_cast<std::int32_t>(root.rows.size() - root.eliminatedCount());
            std::int32_t end = 0;
            for (const std::int32_t row : root.rows.first(root.eliminatedCount()))
                end = std::max(end, row + 1);
            for (const std::int32_t row : root.rows.subspan(root.eliminatedCount()))
                m_outsideRows[toSize(next++)] = row;
            OOS_ASSERT(
                std::is_sorted(m_outsideRows.data() + subtree.firstOutsideRow, m_outsideRows.data() + next));
            subtreeEnd.push_back(end);
            m_subtrees.push_back(subtree);
        }
        for (std::size_t u = 0; u < m_units.size(); ++u)
        {
            const std::int32_t t = m_units[u].subtree;
            if (t == NO_SUBTREE)
                continue;
            Subtree& subtree = m_subtrees[static_cast<std::size_t>(t)];
            if (subtree.endUnit == 0)
                subtree.firstUnit = static_cast<std::int32_t>(u);
            subtree.endUnit = static_cast<std::int32_t>(u) + 1;
        }
    }

    // Pass 2: the rows of each unit (eliminated first, then the others increasing) and the rows of its panels there;
    // position[e] is the row of equation e in the block of the unit being filled, UNSET outside it.
    base::BudgetedBuffer<std::int32_t> position(budget, dimension, "solve plan: equation positions",
                                                base::FOR_OVERWRITE);
    std::fill_n(position.data(), dimension, UNSET);
    std::vector<std::int32_t> others; // bounded by UNIT_ROWS or the largest panel
    for (const Unit& unit : m_units)
    {
        std::int32_t* unitRows = m_unitRows.data() + unit.firstRow;
        std::int32_t next = 0;
        for (std::int32_t s = unit.firstSupernode; s <= unit.lastSupernode; ++s)
        {
            const FactorPanel panel = factor.panel(s);
            for (const std::int32_t row : panel.rows.first(panel.eliminatedCount()))
            {
                OOS_ASSERT(position[static_cast<std::size_t>(row)] == UNSET);
                position[static_cast<std::size_t>(row)] = next;
                unitRows[next++] = row;
            }
        }
        OOS_ASSERT(next == unit.eliminatedRowCount);
        others.clear();
        constexpr std::int32_t COLLECTED = -2;
        for (std::int32_t s = unit.firstSupernode; s <= unit.lastSupernode; ++s)
            for (const std::int32_t row : factor.panel(s).rows)
                if (position[static_cast<std::size_t>(row)] == UNSET)
                {
                    position[static_cast<std::size_t>(row)] = COLLECTED;
                    others.push_back(row);
                }
        std::sort(others.begin(), others.end());
        for (const std::int32_t row : others)
        {
            position[static_cast<std::size_t>(row)] = next;
            unitRows[next++] = row;
        }
        OOS_ASSERT(next == unit.rowCount);
        for (std::int32_t s = unit.firstSupernode; s <= unit.lastSupernode; ++s)
        {
            std::int32_t* local = m_localRows.data() + m_localRowOffset[static_cast<std::size_t>(s)];
            for (const std::int32_t row : factor.panel(s).rows)
                *local++ = position[static_cast<std::size_t>(row)];
        }
        std::int32_t* accumulator = m_accumulatorRows.data() + unit.firstRow;
        for (std::int32_t k = 0; k < unit.rowCount; ++k)
        {
            accumulator[k] = IN_RIGHT_HAND_SIDES;
            if (unit.subtree == NO_SUBTREE || unitRows[k] < subtreeEnd[static_cast<std::size_t>(unit.subtree)])
                continue;
            // A row beyond the subtree is an update row of its root (FactorSolve's sweep by subtrees).
            const std::span<const std::int32_t> outside =
                outsideRows(m_subtrees[static_cast<std::size_t>(unit.subtree)]);
            const auto found = std::lower_bound(outside.begin(), outside.end(), unitRows[k]);
            OOS_ASSERT(found != outside.end() && *found == unitRows[k]);
            accumulator[k] = static_cast<std::int32_t>(found - outside.begin());
        }
        for (std::int32_t k = 0; k < unit.rowCount; ++k)
            position[static_cast<std::size_t>(unitRows[k])] = UNSET;
    }
}

} // namespace oos::linalg
