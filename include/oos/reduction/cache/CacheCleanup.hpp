#pragma once

#include "oos/reduction/cache/CacheEntry.hpp"
#include "oos/reduction/cache/CacheLimits.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::reduction
{

// One entry of a cache as its cleanup sees it.
struct CacheEntryRecord
{
    CacheEntryKind kind = CacheEntryKind::Reduction;
    std::uint64_t unitIdentity = 0; // the unit (CacheKey::unitIdentity)
    std::uint64_t keyHash = 0;      // CacheKey::hash
    std::uint64_t bytes = 0;
    std::int64_t lastUsed = 0; // seconds since the epoch: written or read last
    bool isInUse = false;      // used by an analysis running in this process: never removed
};

// Why an entry is removed, in the order the cleanup takes them.
enum class EvictionReason : std::uint8_t
{
    Superseded,       // the same unit with another (older) key
    Expired,          // not used for longer than the limits' maxAge
    LeastRecentlyUsed // to make room, the recovery data before the reductions
};

[[nodiscard]] constexpr std::string_view evictionReasonName(EvictionReason reason) noexcept
{
    switch (reason)
    {
    case EvictionReason::Superseded:
        return "superseded";
    case EvictionReason::Expired:
        return "expired";
    case EvictionReason::LeastRecentlyUsed:
        return "least_recently_used";
    }
    return "";
}

struct Eviction
{
    std::size_t entry = 0; // index into the entries given
    EvictionReason reason = EvictionReason::Superseded;
};

// What adding an entry does to a cache: the entries to remove, in order, or why the entry is not stored.
struct CleanupPlan
{
    bool isStorable = false;
    std::vector<Eviction> evictions; // empty when not storable: nothing is removed for an entry that is not stored
    std::string refusal;             // why it is not stored
};

// The cleanup of a cache when an entry is added: if the new entry would exceed a limit (the total one, or the
// recovery-data one for recovery data), entries are
// removed in this order until it fits:
//   (1) the superseded entries (the same unit, identified by the hash of its element IDs, with another key);
//   (2) the entries not used for longer than the limits' maxAge;
//   (3) the least recently used ones, the recovery data before the reductions;
// each group from the least recently used. An entry in use is never removed; an entry is removed only when it frees
// room under a limit that is exceeded (a reduction frees nothing under the recovery-data limit). An entry larger than
// its limit, or one that would not fit even with every entry not in use removed, is not stored, and nothing is removed
// for it.
//
// Thread-safety: a pure function.
class CacheCleanup
{
public:
    [[nodiscard]] static CleanupPlan plan(std::span<const CacheEntryRecord> entries, const CacheEntryRecord& added,
                                          const CacheLimits& limits, std::int64_t now);
};

} // namespace oos::reduction
