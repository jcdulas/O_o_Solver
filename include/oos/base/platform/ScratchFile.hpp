#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace oos::base
{

// A file of scratch data (the panels of a factor out of core, the modes of multilevel substructuring, the Lanczos
// vectors out of core): made in a folder of the caller's choice under a unique name, opened exclusive, read and written
// by position from any number of threads at once, and removed when the object dies, whatever the end of the analysis
// (Windows: FILE_FLAG_DELETE_ON_CLOSE, so that even a killed process leaves nothing; POSIX: unlinked as soon as it is
// open).
//
// Unbuffered (isUnbuffered): the reads and writes bypass the system's file cache (Windows FILE_FLAG_NO_BUFFERING,
// POSIX O_DIRECT where the system has it), so that a large spool does not push the process's memory out and the disk's
// own rate is what is measured. Every offset, size and buffer address must then be a multiple of ALIGNMENT
// (std::invalid_argument otherwise); a buffered file takes any.
//
// Thread-safety: read() and write() are safe concurrently (positional, no shared file pointer). Ownership: owns the
// file; movable, not copyable.
class ScratchFile
{
public:
    // The alignment of the offsets, sizes and buffers of an unbuffered file: a multiple of every disk's sector (4 kB
    // sectors and the 512-byte sectors of older disks).
    static constexpr std::size_t ALIGNMENT = 4096;

    ScratchFile() = default;
    // Creates the file in `folder` (made when missing). Throws FileError when it cannot be created.
    ScratchFile(const std::filesystem::path& folder, bool isUnbuffered);
    ScratchFile(const ScratchFile&) = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;
    ScratchFile(ScratchFile&& other) noexcept;
    ScratchFile& operator=(ScratchFile&& other) noexcept;
    ~ScratchFile();

    [[nodiscard]] bool isOpen() const noexcept;
    [[nodiscard]] bool isUnbuffered() const noexcept { return m_isUnbuffered; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

    // Writes `bytes` bytes at `offset` (the file grows as needed). Throws FileError when the write fails (the disk is
    // full), std::invalid_argument for an unaligned request of an unbuffered file.
    void write(std::uint64_t offset, const void* data, std::size_t bytes) const;
    // Reads `bytes` bytes at `offset`. Throws FileError when the read fails or comes short.
    void read(std::uint64_t offset, void* data, std::size_t bytes) const;

    // The bytes free for this process on the disk of `folder` (the first existing folder of its path). 0 when unknown.
    [[nodiscard]] static std::uint64_t freeBytes(const std::filesystem::path& folder) noexcept;

private:
    void checkAlignment(const char* operation, std::uint64_t offset, const void* data, std::size_t bytes) const;
    void close() noexcept;

    std::filesystem::path m_path;
    bool m_isUnbuffered = false;
#if defined(_WIN32)
    void* m_handle = nullptr;
#else
    int m_descriptor = -1;
#endif
};

} // namespace oos::base
