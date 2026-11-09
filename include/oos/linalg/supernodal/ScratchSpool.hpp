#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/platform/ScratchFile.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace oos::linalg
{

// Where a record of a ScratchSpool lies: its first byte in the spool's address space and its size. Set by the writer
// when the record's block is flushed (Writer::append's slot): read only after that.
struct SpoolRecord
{
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

// A memory block aligned for the unbuffered reads and writes of a scratch file (base::ScratchFile::ALIGNMENT), its
// bytes reserved in a budget. Movable, not copyable.
class SpoolBuffer
{
public:
    SpoolBuffer() = default;
    SpoolBuffer(base::MemoryBudget& budget, std::size_t bytes, const char* label);
    SpoolBuffer(const SpoolBuffer&) = delete;
    SpoolBuffer& operator=(const SpoolBuffer&) = delete;
    SpoolBuffer(SpoolBuffer&& other) noexcept;
    SpoolBuffer& operator=(SpoolBuffer&& other) noexcept;
    ~SpoolBuffer();

    [[nodiscard]] char* data() const noexcept { return m_data; }
    [[nodiscard]] std::size_t size() const noexcept { return m_size; }

private:
    void free() noexcept;

    base::MemoryReservation m_reservation;
    char* m_data = nullptr;
    std::size_t m_size = 0;
};

// Records written once and read back many times: the panels of a factor out of core, the modes and
// panels of multilevel substructuring, the Lanczos vectors out of core. On disk (a base::ScratchFile in the options'
// folder) or, when the plan gives them room, in memory: the same records either way, their bits unchanged.
//
// Writing. Each thread that writes takes a Writer of its own; it appends records into a block of its own (blockBytes),
// and flushes the block when the next record does not fit (or at the end): the block's place in the spool's address
// space is taken then (the bytes written so far, the place after them), so that the file grows by appending whole
// blocks and a writer's records lie in large contiguous runs whatever the other threads do. A record larger than a
// block is a block of its own. The place of a record depends on the order the writers flush, never its bits.
//
// Reading. read() and readSpan() copy bytes of flushed blocks, from any number of threads at once; an unbuffered file
// is read by aligned requests through a bounce buffer.
//
// Memory: the writers' blocks and, in memory, every flushed block, reserved in the budget (MemoryBudgetError).
//
// Thread-safety: Writers are per thread; read(), readSpan() and the statistics are safe concurrently with writers of
// other records. Ownership: owns its file and blocks; references the budget.
class ScratchSpool
{
public:
    // A block of 8 MB: 2 000 reads a gigabyte, each long enough for the disk's sequential rate (the campaign's probe:
    // 2.4 to 2.6 GB/s by requests of 16 MB).
    static constexpr std::size_t DEFAULT_BLOCK_BYTES = std::size_t{8} << 20;

    struct Options
    {
        // On disk: the folder of the scratch file; in memory otherwise.
        bool isOnDisk = false;
        std::filesystem::path folder;
        // On disk: without the system's file cache (base::ScratchFile).
        bool isUnbuffered = true;
        std::size_t blockBytes = DEFAULT_BLOCK_BYTES;
    };

    class Writer
    {
    public:
        explicit Writer(ScratchSpool& spool);
        Writer(const Writer&) = delete;
        Writer& operator=(const Writer&) = delete;
        // Flushes what is left (a failure then is lost: call flush() first to see it).
        ~Writer();
        // Appends `bytes` bytes; `slot` receives the record's place when its block is flushed and must outlive that.
        // Throws MemoryBudgetError, FileError.
        void append(const void* data, std::size_t bytes, SpoolRecord& slot);
        // Writes the current block, setting its records' slots. Throws FileError.
        void flush();

    private:
        ScratchSpool& m_spool;
        SpoolBuffer m_block;
        std::size_t m_used = 0;
        std::vector<std::pair<SpoolRecord*, std::uint64_t>> m_pending; // slot, offset in the block; bounded by a block
    };

    ScratchSpool(const Options& options, base::MemoryBudget& budget);
    ScratchSpool(const ScratchSpool&) = delete;
    ScratchSpool& operator=(const ScratchSpool&) = delete;

    [[nodiscard]] bool isOnDisk() const noexcept { return m_file.isOpen(); }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_file.path(); }
    [[nodiscard]] std::size_t blockBytes() const noexcept { return m_blockBytes; }

    // Copies a record. Throws FileError.
    void read(const SpoolRecord& record, void* into) const;
    // Copies `bytes` bytes from `offset` (within one flushed block). Throws FileError, std::out_of_range.
    void readSpan(std::uint64_t offset, std::uint64_t bytes, void* into) const;

    // The bytes the spool holds (its address space, the blocks' padding included), and the bytes written and read so
    // far (the disk's traffic when on disk).
    [[nodiscard]] std::uint64_t heldBytes() const noexcept { return m_end.load(); }
    [[nodiscard]] std::uint64_t writtenBytes() const noexcept { return m_written.load(); }
    [[nodiscard]] std::uint64_t readBytes() const noexcept { return m_read.load(); }

private:
    friend class Writer;
    // Takes the place of a flushed block of `bytes` bytes (its padded size) and stores it.
    std::uint64_t store(SpoolBuffer&& block, std::size_t bytes);

    base::MemoryBudget& m_budget;
    std::size_t m_blockBytes;
    base::ScratchFile m_file;
    std::atomic<std::uint64_t> m_end{0};
    mutable std::atomic<std::uint64_t> m_written{0};
    mutable std::atomic<std::uint64_t> m_read{0};
    // In memory: the flushed blocks by their first byte (sorted; std::vector: bounded by the spool's blocks).
    mutable std::mutex m_mutex;
    std::vector<std::pair<std::uint64_t, std::unique_ptr<SpoolBuffer>>> m_blocks;
};

} // namespace oos::linalg
