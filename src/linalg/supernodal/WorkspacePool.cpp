#include "WorkspacePool.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <new>
#include <utility>

namespace oos::linalg
{

std::int64_t WorkspacePool::classSize(std::int64_t values) noexcept
{
    if (values < KEPT_MINIMUM_VALUES)
        return values;
    // 2^e·(1 + k/CLASSES_PER_OCTAVE): the octave of `values`, then the step of its classes rounded up.
    const auto octave = std::int64_t{1} << (std::bit_width(static_cast<std::uint64_t>(values)) - 1);
    const std::int64_t step = octave / CLASSES_PER_OCTAVE;
    return octave + (values - octave + step - 1) / step * step;
}

base::AlignedBuffer<double> WorkspacePool::acquire(std::int64_t values)
{
    const std::int64_t size = classSize(values);
    {
        const std::scoped_lock lock(m_mutex);
        if (const auto kept = m_kept.find(size); kept != m_kept.end())
        {
            base::AlignedBuffer<double> block = std::move(kept->second);
            m_kept.erase(kept);
            m_keptValues -= size;
            m_inUseValues += size;
            return block;
        }
        // Room for the new block: the kept blocks go, the smallest first (the largest save the most page faults).
        while (!m_kept.empty() && m_inUseValues + m_keptValues + size > m_capacity)
        {
            m_keptValues -= m_kept.begin()->first;
            m_kept.erase(m_kept.begin());
        }
        m_inUseValues += size;
        m_heldPeak = std::max(m_heldPeak, m_inUseValues + m_keptValues);
    }
    try
    {
        return base::AlignedBuffer<double>(static_cast<std::size_t>(size), base::FOR_OVERWRITE);
    }
    catch (...)
    {
        const std::scoped_lock lock(m_mutex);
        m_inUseValues -= size;
        throw;
    }
}

void WorkspacePool::release(base::AlignedBuffer<double> block) noexcept
{
    const auto size = static_cast<std::int64_t>(block.size());
    if (size == 0)
        return;
    const std::scoped_lock lock(m_mutex);
    m_inUseValues -= size;
    if (size < KEPT_MINIMUM_VALUES || m_inUseValues + m_keptValues + size > m_capacity)
        return; // `block` is freed on return
    try
    {
        m_kept.emplace(size, std::move(block));
        m_keptValues += size;
    }
    catch (const std::bad_alloc&)
    {
        // No room to list it: the block is freed (keeping it only saves time).
    }
}

std::int64_t WorkspacePool::heldPeak() const
{
    const std::scoped_lock lock(m_mutex);
    return m_heldPeak;
}

} // namespace oos::linalg
