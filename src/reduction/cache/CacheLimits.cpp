#include "oos/reduction/cache/CacheLimits.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <stdexcept>
#include <string>
#include <system_error>

namespace oos::reduction
{

CacheLimits CacheLimits::resolve(const std::filesystem::path& folder, std::uint64_t heldBytes,
                                 std::optional<std::uint64_t> totalBytes, std::optional<std::uint64_t> recoveryBytes,
                                 std::optional<std::chrono::seconds> maxAge)
{
    CacheLimits limits;
    if (totalBytes)
    {
        limits.totalBytes = *totalBytes;
    }
    else
    {
        std::error_code error;
        const std::filesystem::space_info space = std::filesystem::space(folder, error);
        const std::uint64_t usable = error ? 0 : static_cast<std::uint64_t>(space.available) + heldBytes;
        limits.totalBytes = static_cast<std::uint64_t>(DEFAULT_DISK_FRACTION * static_cast<double>(usable));
    }
    limits.recoveryBytes =
        recoveryBytes ? *recoveryBytes
                      : static_cast<std::uint64_t>(DEFAULT_RECOVERY_FRACTION * static_cast<double>(limits.totalBytes));
    if (limits.recoveryBytes > limits.totalBytes)
        throw std::invalid_argument(base::concatenate(
            {"cache limits: the recovery-data limit (cache_recovery_max_size, ",
             base::formatBytes(limits.recoveryBytes), ") exceeds the total limit (cache_max_size, ",
             base::formatBytes(limits.totalBytes), "); give a recovery limit at most the total one"}));
    if (maxAge)
    {
        if (maxAge->count() <= 0)
            throw std::invalid_argument(base::concatenate(
                {"cache limits: expected a positive cache_max_age, found ", std::to_string(maxAge->count()), " s"}));
        limits.maxAge = *maxAge;
    }
    return limits;
}

} // namespace oos::reduction
