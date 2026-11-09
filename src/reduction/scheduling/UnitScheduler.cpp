#include "oos/reduction/scheduling/UnitScheduler.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace oos::reduction
{

UnitScheduler::UnitScheduler(std::span<const UnitFootprint> units, SchedulerOptions options)
    : m_units(units.begin(), units.end()), m_options(options), m_isScheduled(units.size(), 0)
{
    if (m_options.threadCount == 0)
        throw std::invalid_argument("UnitScheduler: expected at least one thread, found 0");
    m_order.resize(m_units.size());
    std::iota(m_order.begin(), m_order.end(), std::size_t{0});
    std::stable_sort(m_order.begin(), m_order.end(),
                     [&](std::size_t a, std::size_t b) { return m_units[a].flopCount > m_units[b].flopCount; });
    m_remaining = m_order;
    m_byCondensation = m_order;
    std::stable_sort(m_byCondensation.begin(), m_byCondensation.end(), [&](std::size_t a, std::size_t b)
                     { return m_units[a].condensationBytes() > m_units[b].condensationBytes(); });
    for (const UnitFootprint& unit : m_units)
    {
        m_keptRemaining += unit.keptBytes;
        m_flopsRemaining += unit.flopCount;
    }
}

bool UnitScheduler::isLarge(std::size_t unit) const noexcept
{
    const double work = m_units[unit].flopCount;
    return m_options.threadCount > 1 && work >= m_options.largeUnitMinimumFlops &&
           work * static_cast<double>(m_options.threadCount) >= m_flopsRemaining;
}

std::uint64_t UnitScheduler::largestCondensationExcept(std::span<const std::size_t> excluded) const
{
    for (std::size_t k = m_condensationStart; k < m_byCondensation.size(); ++k)
    {
        const std::size_t unit = m_byCondensation[k];
        if (m_isScheduled[unit] == 0 && std::find(excluded.begin(), excluded.end(), unit) == excluded.end())
            return m_units[unit].condensationBytes();
    }
    return 0;
}

std::uint64_t UnitScheduler::reserve() const
{
    return m_keptRemaining + std::max(largestCondensationExcept({}), m_options.laterPhaseBytes);
}

UnitWave UnitScheduler::nextWave(std::uint64_t availableBytes)
{
    if (m_remaining.empty())
        throw std::logic_error("UnitScheduler::nextWave: every unit is scheduled; ask isDone() first");
    UnitWave wave;
    const std::size_t first = m_remaining.front();
    wave.units.push_back(first);
    if (m_options.threadCount > 1 && !isLarge(first))
    {
        std::uint64_t peak = m_units[first].keptBytes + m_units[first].condensationBytes();
        std::uint64_t kept = m_units[first].keptBytes;
        const std::size_t largest = m_options.threadCount * std::max<std::size_t>(1, m_options.unitsPerThread);
        for (std::size_t k = 1; k < m_remaining.size() && wave.units.size() < largest; ++k)
        {
            const std::size_t unit = m_remaining[k];
            const UnitFootprint& footprint = m_units[unit];
            if (isLarge(unit) || peak + footprint.keptBytes + footprint.condensationBytes() > availableBytes)
                continue;
            // What the units after the wave need, once this one keeps its data too.
            wave.units.push_back(unit);
            const std::uint64_t restKept = m_keptRemaining - kept - footprint.keptBytes;
            const std::uint64_t rest =
                restKept + std::max(largestCondensationExcept(wave.units), m_options.laterPhaseBytes);
            if (kept + footprint.keptBytes + rest > availableBytes)
            {
                wave.units.pop_back();
                continue;
            }
            peak += footprint.keptBytes + footprint.condensationBytes();
            kept += footprint.keptBytes;
        }
    }
    wave.isShared = wave.units.size() == 1;
    for (const std::size_t unit : wave.units)
    {
        m_isScheduled[unit] = 1;
        m_keptRemaining -= m_units[unit].keptBytes;
        m_flopsRemaining -= m_units[unit].flopCount;
    }
    m_remaining.erase(std::remove_if(m_remaining.begin(), m_remaining.end(),
                                     [&](std::size_t unit) { return m_isScheduled[unit] != 0; }),
                      m_remaining.end());
    while (m_condensationStart < m_byCondensation.size() && m_isScheduled[m_byCondensation[m_condensationStart]] != 0)
        ++m_condensationStart;
    if (m_remaining.empty())
        m_flopsRemaining = 0.0; // no drift of the sum
    return wave;
}

} // namespace oos::reduction
