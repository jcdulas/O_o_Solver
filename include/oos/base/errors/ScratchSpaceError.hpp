#pragma once

#include "oos/base/errors/FileError.hpp"

#include <cstdint>
#include <string>

namespace oos::base
{

// The scratch folder of an analysis has less free space than its plan needs (the factor out of core,
// the Lanczos vectors spooled), refused before any work; the message names the folder, the bytes needed and free and
// the remedies. A FileError, as the disk's other failures.
class ScratchSpaceError : public FileError
{
public:
    ScratchSpaceError(const std::string& message, std::uint64_t neededBytes, std::uint64_t freeBytes)
        : FileError(message), m_neededBytes(neededBytes), m_freeBytes(freeBytes)
    {
    }

    [[nodiscard]] std::uint64_t neededBytes() const noexcept { return m_neededBytes; }
    [[nodiscard]] std::uint64_t freeBytes() const noexcept { return m_freeBytes; }

private:
    std::uint64_t m_neededBytes;
    std::uint64_t m_freeBytes;
};

} // namespace oos::base
