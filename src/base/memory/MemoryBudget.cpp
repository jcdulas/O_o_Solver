#include "oos/base/memory/MemoryBudget.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace oos::base
{

namespace
{

// How many consumers a refusal names: enough to see where the memory went, short enough for one message.
constexpr std::size_t LARGEST_CONSUMERS_REPORTED = 5;

} // namespace

MemoryReservation::MemoryReservation(MemoryReservation&& other) noexcept
    : m_budget(std::exchange(other.m_budget, nullptr)),
      m_bytes(std::exchange(other.m_bytes, 0)),
      m_label(std::move(other.m_label))
{
}

MemoryReservation& MemoryReservation::operator=(MemoryReservation&& other) noexcept
{
    if (this != &other)
    {
        release();
        m_budget = std::exchange(other.m_budget, nullptr);
        m_bytes = std::exchange(other.m_bytes, 0);
        m_label = std::move(other.m_label);
    }
    return *this;
}

MemoryReservation::~MemoryReservation()
{
    release();
}

void MemoryReservation::release() noexcept
{
    if (m_budget != nullptr)
        m_budget->giveBack(m_bytes, m_label);
    m_budget = nullptr;
    m_bytes = 0;
}

MemoryReservation MemoryBudget::reserve(std::uint64_t bytes, const std::string& label)
{
    // The reservation's copy of the label is made before anything is counted: if it fails (std::bad_alloc), the
    // budget is unchanged; after the counting only noexcept moves remain.
    std::string ownLabel(label);
    std::lock_guard lock(m_mutex);
    const std::uint64_t available = m_limit - m_used;
    if (bytes > available)
    {
        MemoryShortfall shortfall{bytes, available, m_limit, largestConsumersLocked(LARGEST_CONSUMERS_REPORTED)};
        std::string message =
            concatenate({"memory budget exceeded: '", label, "' needs ", formatBytes(bytes), ", ",
                         formatBytes(available), " of the budget of ", formatBytes(m_limit), " are left"});
        if (!shortfall.largestConsumers.empty())
        {
            message.append("; largest consumers:");
            const char* separator = " ";
            for (const MemoryConsumer& consumer : shortfall.largestConsumers)
            {
                message.append(separator)
                    .append(consumer.label)
                    .append(" (")
                    .append(formatBytes(consumer.bytes))
                    .append(")");
                separator = ", ";
            }
        }
        message.append(". Raise memory=, condense more units, or restrict the requested outputs");
        throw MemoryBudgetError(message, std::move(shortfall));
    }
    // The label is counted first: if its map node cannot be allocated, nothing has changed.
    m_byLabel[ownLabel] += bytes;
    m_used += bytes;
    m_peak = std::max(m_peak, m_used);
    return MemoryReservation(this, bytes, std::move(ownLabel));
}

bool MemoryBudget::fits(std::uint64_t bytes) const
{
    std::lock_guard lock(m_mutex);
    return bytes <= m_limit - m_used;
}

std::uint64_t MemoryBudget::used() const
{
    std::lock_guard lock(m_mutex);
    return m_used;
}

std::uint64_t MemoryBudget::available() const
{
    std::lock_guard lock(m_mutex);
    return m_limit - m_used;
}

std::uint64_t MemoryBudget::peak() const
{
    std::lock_guard lock(m_mutex);
    return m_peak;
}

std::vector<MemoryConsumer> MemoryBudget::largestConsumers(std::size_t count) const
{
    std::lock_guard lock(m_mutex);
    return largestConsumersLocked(count);
}

std::vector<MemoryConsumer> MemoryBudget::largestConsumersLocked(std::size_t count) const
{
    std::vector<MemoryConsumer> consumers;
    consumers.reserve(m_byLabel.size());
    for (const auto& [label, bytes] : m_byLabel)
        consumers.push_back({label, bytes});
    std::sort(consumers.begin(), consumers.end(), [](const MemoryConsumer& a, const MemoryConsumer& b)
              { return a.bytes > b.bytes || (a.bytes == b.bytes && a.label < b.label); });
    if (consumers.size() > count)
        consumers.resize(count);
    return consumers;
}

void MemoryBudget::giveBack(std::uint64_t bytes, const std::string& label) noexcept
{
    std::lock_guard lock(m_mutex);
    m_used -= bytes;
    auto found = m_byLabel.find(label);
    if (found != m_byLabel.end())
    {
        found->second -= bytes;
        if (found->second == 0)
            m_byLabel.erase(found);
    }
}

std::string formatBytes(std::uint64_t bytes)
{
    constexpr std::array<const char*, 6> UNITS = {"B", "kB", "MB", "GB", "TB", "PB"};
    constexpr std::uint64_t STEP = 1000;
    if (bytes < STEP)
        return std::to_string(bytes).append(" B");
    // 999.5 and above print as "1e+03" with three significant digits: they go to the next unit.
    constexpr double NEXT_UNIT_THRESHOLD = 999.5;
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= NEXT_UNIT_THRESHOLD && unit + 1 < UNITS.size())
    {
        value /= static_cast<double>(STEP);
        ++unit;
    }
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.3g %s", value, UNITS[unit]);
    return text.data();
}

} // namespace oos::base
