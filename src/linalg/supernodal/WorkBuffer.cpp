#include "WorkBuffer.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace oos::linalg
{

void WorkspaceMeter::add(std::int64_t values) noexcept
{
    const std::int64_t live = m_live.fetch_add(values, std::memory_order_relaxed) + values;
    std::int64_t peak = m_peak.load(std::memory_order_relaxed);
    while (live > peak && !m_peak.compare_exchange_weak(peak, live, std::memory_order_relaxed))
    {
    }
}

WorkBuffer::WorkBuffer(WorkBuffer&& other) noexcept
    : m_plannedValues(std::exchange(other.m_plannedValues, 0)),
      m_label(std::move(other.m_label)),
      m_meter(std::exchange(other.m_meter, nullptr)),
      m_pool(std::exchange(other.m_pool, nullptr)),
      m_growth(std::move(other.m_growth)),
      m_values(std::move(other.m_values))
{
}

WorkBuffer& WorkBuffer::operator=(WorkBuffer&& other) noexcept
{
    if (this != &other)
    {
        release();
        m_plannedValues = std::exchange(other.m_plannedValues, 0);
        m_label = std::move(other.m_label);
        m_meter = std::exchange(other.m_meter, nullptr);
        m_pool = std::exchange(other.m_pool, nullptr);
        m_growth = std::move(other.m_growth);
        m_values = std::move(other.m_values);
    }
    return *this;
}

void WorkBuffer::ensure(std::int64_t requested, base::MemoryBudget& budget, std::int64_t keptValues)
{
    if (requested <= capacity())
        return;
    const std::int64_t values = m_pool != nullptr ? WorkspacePool::classSize(requested) : requested;
    // What the plan does not cover: the values beyond it, and, while the old contents are copied, the planned part of
    // the old buffer (both buffers are alive then; the new one's planned part takes over the plan once the old one is
    // freed).
    const std::int64_t oldPlanned = plannedPartOf(capacity());
    const std::int64_t newPlanned = plannedPartOf(values);
    base::MemoryReservation growth;
    if (values > m_plannedValues)
        growth = budget.reserve(static_cast<std::uint64_t>(values - m_plannedValues) * sizeof(double),
                                std::string(m_label).append(" (beyond the planned working storage)"));
    base::MemoryReservation transition;
    if (oldPlanned > 0)
        transition = budget.reserve(static_cast<std::uint64_t>(oldPlanned) * sizeof(double),
                                    std::string(m_label).append(" (old contents while they are copied)"));
    base::AlignedBuffer<double> grown =
        m_pool != nullptr ? m_pool->acquire(values)
                          : base::AlignedBuffer<double>(static_cast<std::size_t>(values), base::FOR_OVERWRITE);
    if (m_meter != nullptr)
        m_meter->add(newPlanned - oldPlanned); // >= 0: a buffer only grows
    std::copy_n(m_values.data(), static_cast<std::size_t>(std::min(keptValues, capacity())), grown.data());
    std::swap(m_values, grown);
    if (m_pool != nullptr)
        m_pool->release(std::move(grown));
    m_growth = std::move(growth);
}

void WorkBuffer::release() noexcept
{
    if (m_meter != nullptr)
        m_meter->remove(plannedPartOf(capacity()));
    if (m_pool != nullptr)
        m_pool->release(std::move(m_values));
    m_values = base::AlignedBuffer<double>();
    m_growth.release();
}

} // namespace oos::linalg
