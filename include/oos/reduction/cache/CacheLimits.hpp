#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>

namespace oos::reduction
{

// The volume limits of a condensation cache on disk (two predefined limits, both settable):
// - totalBytes: cache_max_size, everything the cache holds; by default DEFAULT_DISK_FRACTION of the disk space the
//   cache could use when it is opened (the free space plus what the cache already holds, so that the default does not
//   shrink as the cache fills the disk);
// - recoveryBytes: cache_recovery_max_size, the recovery data (the Craig-Bampton modes); by default half of totalBytes;
// - maxAge: cache_max_age, an entry not used for longer is expired (removed second, after the superseded ones);
//   DEFAULT_MAX_AGE by default.
//
// Thread-safety: a value.
struct CacheLimits
{
    static constexpr double DEFAULT_DISK_FRACTION = 0.2;
    static constexpr double DEFAULT_RECOVERY_FRACTION = 0.5;
    static constexpr std::chrono::seconds DEFAULT_MAX_AGE{std::chrono::hours{24 * 30}};

    std::uint64_t totalBytes = 0;
    std::uint64_t recoveryBytes = 0;
    std::chrono::seconds maxAge = DEFAULT_MAX_AGE;

    // The limits of a cache in `folder` holding `heldBytes`: the given ones, the defaults for those not given.
    // Throws std::invalid_argument when the recovery limit exceeds the total one.
    [[nodiscard]] static CacheLimits resolve(const std::filesystem::path& folder, std::uint64_t heldBytes,
                                             std::optional<std::uint64_t> totalBytes,
                                             std::optional<std::uint64_t> recoveryBytes,
                                             std::optional<std::chrono::seconds> maxAge);
};

} // namespace oos::reduction
