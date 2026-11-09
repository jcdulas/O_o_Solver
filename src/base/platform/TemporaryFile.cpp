#include "oos/base/platform/TemporaryFile.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/FileError.hpp"

#include <atomic>
#include <cstdint>
#include <random>
#include <string>
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
#include <cerrno>
#include <cstdio>
#endif

namespace oos::base
{

namespace
{

// A suffix no other writer uses at the same time: a per-process random number (two processes writing the same cache
// entry) and a counter (two threads of one process).
std::string uniqueSuffix()
{
    static const std::uint64_t PROCESS_PART = []
    {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }();
    static std::atomic<std::uint64_t> counter{0};
    return concatenate({".tmp-", std::to_string(PROCESS_PART), "-", std::to_string(counter.fetch_add(1))});
}

// Renames `from` onto `to`, replacing `to` if it exists, in one step. std::filesystem::rename does not promise to
// replace an existing file on Windows, hence MoveFileExW there; POSIX rename() replaces atomically.
std::error_code replaceFile(const std::filesystem::path& from, const std::filesystem::path& to)
{
#if defined(_WIN32)
    if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0)
        return {static_cast<int>(GetLastError()), std::system_category()};
    return {};
#else
    if (std::rename(from.c_str(), to.c_str()) != 0)
        return {errno, std::generic_category()};
    return {};
#endif
}

} // namespace

TemporaryFile::TemporaryFile(std::filesystem::path target) : m_target(std::move(target))
{
    m_temporary = m_target;
    m_temporary += uniqueSuffix();
    m_stream.open(m_temporary, std::ios::binary | std::ios::trunc);
    if (!m_stream.is_open())
        throw FileError(
            concatenate({"cannot create the temporary file '", m_temporary.string(), "' to write '", m_target.string(),
                         "': check that its folder exists and is writable, and that the disk has room"}));
}

TemporaryFile::TemporaryFile(TemporaryFile&& other) noexcept
    : m_target(std::move(other.m_target)),
      m_temporary(std::move(other.m_temporary)),
      m_stream(std::move(other.m_stream)),
      m_isCommitted(std::exchange(other.m_isCommitted, true))
{
    other.m_temporary.clear();
}

TemporaryFile& TemporaryFile::operator=(TemporaryFile&& other) noexcept
{
    if (this != &other)
    {
        discard();
        m_target = std::move(other.m_target);
        m_temporary = std::move(other.m_temporary);
        m_stream = std::move(other.m_stream);
        m_isCommitted = std::exchange(other.m_isCommitted, true);
        other.m_temporary.clear();
    }
    return *this;
}

TemporaryFile::~TemporaryFile()
{
    discard();
}

void TemporaryFile::commit()
{
    // A second commit has nothing left to do: the target already holds the content (the stream is closed).
    if (m_isCommitted)
        return;
    m_stream.flush();
    const bool hasWritten = m_stream.good();
    m_stream.close();
    if (!hasWritten || m_stream.fail())
    {
        discard();
        throw FileError(concatenate(
            {"cannot write the temporary file of '", m_target.string(),
             "': a write failed (is the disk full?); the target is unchanged: free disk space and write it again"}));
    }
    if (const std::error_code error = replaceFile(m_temporary, m_target); error)
    {
        discard();
        throw FileError(
            concatenate({"cannot rename the temporary file onto '", m_target.string(), "': ", error.message(),
                         "; the target is unchanged: close the program that has it open, or remove the "
                         "folder that has its name, and write it again"}));
    }
    m_isCommitted = true;
}

void TemporaryFile::discard() noexcept
{
    if (m_isCommitted || m_temporary.empty())
        return;
    if (m_stream.is_open())
        m_stream.close();
    std::error_code ignored;
    std::filesystem::remove(m_temporary, ignored);
    m_temporary.clear();
}

} // namespace oos::base
