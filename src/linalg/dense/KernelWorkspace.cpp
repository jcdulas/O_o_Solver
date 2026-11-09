#include "oos/linalg/dense/KernelWorkspace.hpp"

#include <algorithm>
#include <bit>
#include <new>
#include <utility>

namespace oos::linalg
{

ScratchBuffer KernelWorkspace::acquire(std::size_t values)
{
    {
        const std::scoped_lock lock(m_mutex);
        auto best = m_kept.end();
        for (auto kept = m_kept.begin(); kept != m_kept.end(); ++kept)
            if (kept->size >= values && (best == m_kept.end() || kept->size < best->size))
                best = kept;
        if (best != m_kept.end())
        {
            ScratchBuffer buffer = std::move(*best);
            m_kept.erase(best);
            return buffer;
        }
    }
    const std::size_t size = std::bit_ceil(std::max(values, SMALLEST_CLASS_VALUES));
    return {std::make_unique_for_overwrite<double[]>(size), size};
}

void KernelWorkspace::release(ScratchBuffer buffer) noexcept
{
    if (buffer.values == nullptr)
        return;
    ScratchBuffer dropped;
    {
        const std::scoped_lock lock(m_mutex);
        try
        {
            m_kept.push_back(std::move(buffer));
        }
        catch (const std::bad_alloc&)
        {
            return; // no room to list it: `buffer` is freed (keeping it only saves time)
        }
        if (m_kept.size() > KEPT_BUFFER_COUNT)
        {
            const auto smallest =
                std::min_element(m_kept.begin(), m_kept.end(),
                                 [](const ScratchBuffer& a, const ScratchBuffer& b) { return a.size < b.size; });
            dropped = std::move(*smallest);
            m_kept.erase(smallest);
        }
    }
    // `dropped` is freed outside the lock.
}

std::uint64_t KernelWorkspace::keptBytes() const
{
    const std::scoped_lock lock(m_mutex);
    std::uint64_t bytes = 0;
    for (const ScratchBuffer& buffer : m_kept)
        bytes += static_cast<std::uint64_t>(buffer.size) * sizeof(double);
    return bytes;
}

} // namespace oos::linalg
