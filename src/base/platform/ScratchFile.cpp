#include "oos/base/platform/ScratchFile.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/FileError.hpp"

#include <algorithm>
#include <atomic>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace oos::base
{
namespace
{

// The largest request handed to the system at once: below the 4 GB of a Windows DWORD, a multiple of ALIGNMENT.
constexpr std::size_t LARGEST_REQUEST_BYTES = std::size_t{1} << 30;

// A name no other scratch file of this machine uses at the same time: a per-process random number and a counter.
std::string uniqueName()
{
    static const std::uint64_t PROCESS_PART = []
    {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }();
    static std::atomic<std::uint64_t> counter{0};
    return concatenate(
        {"oos-scratch-", std::to_string(PROCESS_PART), "-", std::to_string(counter.fetch_add(1)), ".bin"});
}

std::string systemMessage()
{
#if defined(_WIN32)
    return std::error_code(static_cast<int>(GetLastError()), std::system_category()).message();
#else
    return std::error_code(errno, std::generic_category()).message();
#endif
}

#if !defined(_WIN32)

// Transfers `bytes` bytes by calls of `transferFrom(done)`, which makes one pwrite or pread of the bytes from `done`
// on and returns its count. A call may transfer fewer bytes than asked without failing (POSIX: a signal, a file
// system's limit): it is made again for the rest, and again when a signal interrupted it before any byte (EINTR).
// Returns nothing, or what stopped the transfer: the system's message of the failing call, read at once, or
// `stoppedText` and the count reached when a call transferred nothing (the end of the file, a device that takes no
// more: errno says nothing then).
template <typename TransferFrom>
std::optional<std::string> transferAll(std::size_t bytes, std::string_view stoppedText,
                                       const TransferFrom& transferFrom)
{
    std::size_t done = 0;
    while (done < bytes)
    {
        const ssize_t count = transferFrom(done);
        if (count < 0)
        {
            if (errno == EINTR)
                continue;
            return systemMessage();
        }
        if (count == 0)
            return concatenate({stoppedText, std::to_string(done), " bytes"});
        done += static_cast<std::size_t>(count);
    }
    return std::nullopt;
}

#endif

} // namespace

ScratchFile::ScratchFile(const std::filesystem::path& folder, bool isUnbuffered) : m_isUnbuffered(isUnbuffered)
{
    std::error_code ignored;
    std::filesystem::create_directories(folder, ignored);
    m_path = folder / uniqueName();
#if defined(_WIN32)
    DWORD flags = FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE;
    if (isUnbuffered)
        flags |= FILE_FLAG_NO_BUFFERING;
    HANDLE handle = CreateFileW(m_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        throw FileError(concatenate({"cannot create the scratch file '", m_path.string(), "': ", systemMessage(),
                                     "; give a scratch folder that exists and is writable (the scratch "
                                     "folder of the eigenvalue solver settings, or the OOS_SCRATCH environment "
                                     "variable)"}));
    m_handle = handle;
#else
    int flags = O_RDWR | O_CREAT | O_EXCL;
#if defined(O_DIRECT)
    if (isUnbuffered)
        flags |= O_DIRECT;
#endif
    m_descriptor = ::open(m_path.c_str(), flags, S_IRUSR | S_IWUSR);
    if (m_descriptor < 0)
        throw FileError(concatenate({"cannot create the scratch file '", m_path.string(), "': ", systemMessage(),
                                     "; give a scratch folder that exists and is writable (the scratch "
                                     "folder of the eigenvalue solver settings, or the OOS_SCRATCH environment "
                                     "variable)"}));
    // Removed from the folder at once: the file lives as long as its descriptor, whatever the end of the process.
    ::unlink(m_path.c_str());
#endif
}

ScratchFile::ScratchFile(ScratchFile&& other) noexcept
    : m_path(std::move(other.m_path)),
      m_isUnbuffered(other.m_isUnbuffered)
#if defined(_WIN32)
      ,
      m_handle(std::exchange(other.m_handle, nullptr))
#else
      ,
      m_descriptor(std::exchange(other.m_descriptor, -1))
#endif
{
}

ScratchFile& ScratchFile::operator=(ScratchFile&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_path = std::move(other.m_path);
        m_isUnbuffered = other.m_isUnbuffered;
#if defined(_WIN32)
        m_handle = std::exchange(other.m_handle, nullptr);
#else
        m_descriptor = std::exchange(other.m_descriptor, -1);
#endif
    }
    return *this;
}

ScratchFile::~ScratchFile()
{
    close();
}

bool ScratchFile::isOpen() const noexcept
{
#if defined(_WIN32)
    return m_handle != nullptr;
#else
    return m_descriptor >= 0;
#endif
}

void ScratchFile::close() noexcept
{
#if defined(_WIN32)
    if (m_handle != nullptr)
        CloseHandle(static_cast<HANDLE>(m_handle));
    m_handle = nullptr;
#else
    if (m_descriptor >= 0)
        ::close(m_descriptor);
    m_descriptor = -1;
#endif
}

void ScratchFile::checkAlignment(const char* operation, std::uint64_t offset, const void* data, std::size_t bytes) const
{
    if (!isOpen())
        throw std::logic_error(concatenate({"ScratchFile::", operation, ": the file is not open; make it first"}));
    if (!m_isUnbuffered)
        return;
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    if (offset % ALIGNMENT != 0 || bytes % ALIGNMENT != 0 || address % ALIGNMENT != 0)
        throw std::invalid_argument(concatenate(
            {"ScratchFile::", operation, ": an unbuffered scratch file takes offsets, sizes and buffers aligned to ",
             std::to_string(ALIGNMENT), " bytes, found offset ", std::to_string(offset), " and size ",
             std::to_string(bytes), "; align the request (the spool's blocks are)"}));
}

void ScratchFile::write(std::uint64_t offset, const void* data, std::size_t bytes) const
{
    checkAlignment("write", offset, data, bytes);
    const auto* from = static_cast<const char*>(data);
    while (bytes > 0)
    {
        const std::size_t part = std::min(bytes, LARGEST_REQUEST_BYTES);
#if defined(_WIN32)
        OVERLAPPED at{};
        at.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        at.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD done = 0;
        const BOOL isWritten = WriteFile(static_cast<HANDLE>(m_handle), from, static_cast<DWORD>(part), &done, &at);
        const std::optional<std::string> failure =
            isWritten == 0 || done != part ? std::optional<std::string>(systemMessage()) : std::nullopt;
#else
        const std::optional<std::string> failure = transferAll(
            part, "no byte written after ", [&](std::size_t done)
            { return ::pwrite(m_descriptor, from + done, part - done, static_cast<off_t>(offset + done)); });
#endif
        if (failure)
            throw FileError(concatenate({"cannot write ", std::to_string(part), " bytes at ", std::to_string(offset),
                                         " of the scratch file '", m_path.string(), "': ", *failure,
                                         "; free disk space there or give another scratch folder (the scratch "
                                         "folder of the eigenvalue solver settings)"}));
        from += part;
        offset += part;
        bytes -= part;
    }
}

void ScratchFile::read(std::uint64_t offset, void* data, std::size_t bytes) const
{
    checkAlignment("read", offset, data, bytes);
    auto* to = static_cast<char*>(data);
    while (bytes > 0)
    {
        const std::size_t part = std::min(bytes, LARGEST_REQUEST_BYTES);
#if defined(_WIN32)
        OVERLAPPED at{};
        at.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        at.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD done = 0;
        const BOOL isRead = ReadFile(static_cast<HANDLE>(m_handle), to, static_cast<DWORD>(part), &done, &at);
        const std::optional<std::string> failure =
            isRead == 0 || done != part ? std::optional<std::string>(systemMessage()) : std::nullopt;
#else
        const std::optional<std::string> failure = transferAll(
            part, "end of the file after ", [&](std::size_t done)
            { return ::pread(m_descriptor, to + done, part - done, static_cast<off_t>(offset + done)); });
#endif
        if (failure)
            throw FileError(concatenate({"cannot read ", std::to_string(part), " bytes at ", std::to_string(offset),
                                         " of the scratch file '", m_path.string(), "': ", *failure,
                                         " (no remedy: the scratch file was written by this analysis; report it if "
                                         "the disk is healthy)"}));
        to += part;
        offset += part;
        bytes -= part;
    }
}

std::uint64_t ScratchFile::freeBytes(const std::filesystem::path& folder) noexcept
{
    std::error_code error;
    std::filesystem::path existing = folder;
    while (!existing.empty() && !std::filesystem::exists(existing, error))
    {
        std::filesystem::path parent = existing.parent_path();
        if (parent == existing)
            break;
        existing = std::move(parent);
    }
    if (existing.empty())
        existing = std::filesystem::current_path(error);
    const std::filesystem::space_info space = std::filesystem::space(existing, error);
    return error ? 0 : static_cast<std::uint64_t>(space.available);
}

} // namespace oos::base
