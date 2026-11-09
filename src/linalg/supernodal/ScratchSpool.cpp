#include "oos/linalg/supernodal/ScratchSpool.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{
namespace
{

constexpr std::size_t ALIGNMENT = base::ScratchFile::ALIGNMENT;

std::uint64_t roundUp(std::uint64_t value, std::uint64_t multiple) noexcept
{
    return (value + multiple - 1) / multiple * multiple;
}

} // namespace

SpoolBuffer::SpoolBuffer(base::MemoryBudget& budget, std::size_t bytes, const char* label)
    : m_reservation(budget.reserve(bytes, label)), m_size(bytes)
{
    if (bytes > 0)
        m_data = static_cast<char*>(::operator new[](bytes, std::align_val_t{ALIGNMENT}));
}

SpoolBuffer::SpoolBuffer(SpoolBuffer&& other) noexcept
    : m_reservation(std::move(other.m_reservation)),
      m_data(std::exchange(other.m_data, nullptr)),
      m_size(std::exchange(other.m_size, 0))
{
}

SpoolBuffer& SpoolBuffer::operator=(SpoolBuffer&& other) noexcept
{
    if (this != &other)
    {
        free();
        m_reservation = std::move(other.m_reservation);
        m_data = std::exchange(other.m_data, nullptr);
        m_size = std::exchange(other.m_size, 0);
    }
    return *this;
}

SpoolBuffer::~SpoolBuffer()
{
    free();
}

void SpoolBuffer::free() noexcept
{
    if (m_data != nullptr)
        ::operator delete[](m_data, std::align_val_t{ALIGNMENT});
    m_data = nullptr;
    m_size = 0;
}

ScratchSpool::ScratchSpool(const Options& options, base::MemoryBudget& budget)
    : m_budget(budget),
      m_blockBytes(static_cast<std::size_t>(roundUp(std::max<std::size_t>(options.blockBytes, 1), ALIGNMENT)))
{
    if (options.isOnDisk)
        m_file = base::ScratchFile(options.folder, options.isUnbuffered);
}

std::uint64_t ScratchSpool::store(SpoolBuffer&& block, std::size_t bytes)
{
    const std::uint64_t padded = roundUp(bytes, ALIGNMENT);
    const std::uint64_t offset = m_end.fetch_add(padded);
    if (m_file.isOpen())
    {
        // The padding is what the block held before: never read (a record ends before it).
        m_file.write(offset, block.data(), static_cast<std::size_t>(padded));
        m_written.fetch_add(padded);
        return offset;
    }
    // In memory: a block flushed less than half full is kept in a buffer of its own size, so that the spool holds
    // about its records' bytes.
    std::unique_ptr<SpoolBuffer> kept;
    if (padded * 2 < block.size())
    {
        kept = std::make_unique<SpoolBuffer>(m_budget, static_cast<std::size_t>(padded), "scratch spool: records");
        std::memcpy(kept->data(), block.data(), bytes);
    }
    else
        kept = std::make_unique<SpoolBuffer>(std::move(block));
    m_written.fetch_add(padded);
    const std::scoped_lock lock(m_mutex);
    const auto at = std::upper_bound(m_blocks.begin(), m_blocks.end(), offset,
                                     [](std::uint64_t value, const auto& entry) { return value < entry.first; });
    m_blocks.emplace(at, offset, std::move(kept));
    return offset;
}

void ScratchSpool::readSpan(std::uint64_t offset, std::uint64_t bytes, void* into) const
{
    if (bytes == 0)
        return;
    m_read.fetch_add(bytes);
    if (m_file.isOpen())
    {
        if (!m_file.isUnbuffered())
        {
            m_file.read(offset, into, static_cast<std::size_t>(bytes));
            return;
        }
        const std::uint64_t first = offset / ALIGNMENT * ALIGNMENT;
        const std::uint64_t end = roundUp(offset + bytes, ALIGNMENT);
        if (first == offset && end == offset + bytes && reinterpret_cast<std::uintptr_t>(into) % ALIGNMENT == 0)
        {
            m_file.read(offset, into, static_cast<std::size_t>(bytes));
            return;
        }
        SpoolBuffer bounce(m_budget, static_cast<std::size_t>(end - first), "scratch spool: aligned reads");
        m_file.read(first, bounce.data(), bounce.size());
        std::memcpy(into, bounce.data() + (offset - first), static_cast<std::size_t>(bytes));
        return;
    }
    const SpoolBuffer* block = nullptr;
    std::uint64_t blockOffset = 0;
    {
        const std::scoped_lock lock(m_mutex);
        const auto at = std::upper_bound(m_blocks.begin(), m_blocks.end(), offset,
                                         [](std::uint64_t value, const auto& entry) { return value < entry.first; });
        if (at != m_blocks.begin())
        {
            const auto& entry = *(at - 1);
            blockOffset = entry.first;
            block = entry.second.get();
        }
    }
    if (block == nullptr || offset + bytes > blockOffset + block->size())
        throw std::out_of_range(base::concatenate(
            {"ScratchSpool::readSpan: the bytes [", std::to_string(offset), ", ", std::to_string(offset + bytes),
             ") are not in one flushed block; read records after their writer has flushed them"}));
    std::memcpy(into, block->data() + (offset - blockOffset), static_cast<std::size_t>(bytes));
}

void ScratchSpool::read(const SpoolRecord& record, void* into) const
{
    readSpan(record.offset, record.bytes, into);
}

ScratchSpool::Writer::Writer(ScratchSpool& spool) : m_spool(spool) {}

ScratchSpool::Writer::~Writer()
{
    try
    {
        flush();
    }
    catch (...) // NOLINT(bugprone-empty-catch): a destructor must not throw; flush() first to see the error
    {
    }
}

void ScratchSpool::Writer::append(const void* data, std::size_t bytes, SpoolRecord& slot)
{
    if (m_used + bytes > m_block.size())
    {
        flush();
        const std::size_t size = std::max(m_spool.m_blockBytes, static_cast<std::size_t>(roundUp(bytes, ALIGNMENT)));
        if (m_block.size() != size)
            m_block = SpoolBuffer(m_spool.m_budget, size, "scratch spool: a writer's block");
    }
    if (bytes > 0)
        std::memcpy(m_block.data() + m_used, data, bytes);
    slot.bytes = bytes;
    m_pending.emplace_back(&slot, m_used);
    m_used += bytes;
}

void ScratchSpool::Writer::flush()
{
    if (m_pending.empty())
        return;
    // On disk the block is written and kept for the next records; in memory the spool takes it (or a copy of its used
    // part, the block then kept here).
    const std::uint64_t offset = m_spool.store(std::move(m_block), m_used);
    for (const auto& [slot, at] : m_pending)
        slot->offset = offset + at;
    m_pending.clear();
    m_used = 0;
}

} // namespace oos::linalg
