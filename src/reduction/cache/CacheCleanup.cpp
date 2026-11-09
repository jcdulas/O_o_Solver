#include "oos/reduction/cache/CacheCleanup.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <algorithm>
#include <string>

namespace oos::reduction
{

CleanupPlan CacheCleanup::plan(std::span<const CacheEntryRecord> entries, const CacheEntryRecord& added,
                               const CacheLimits& limits, std::int64_t now)
{
    CleanupPlan result;
    const bool isRecovery = added.kind == CacheEntryKind::RecoveryData;
    if (added.bytes > limits.totalBytes || (isRecovery && added.bytes > limits.recoveryBytes))
    {
        const std::uint64_t limit = isRecovery ? std::min(limits.totalBytes, limits.recoveryBytes) : limits.totalBytes;
        result.refusal = base::concatenate(
            {"not stored: the entry (", base::formatBytes(added.bytes), ") is larger than the limit of its kind (",
             isRecovery ? "cache_recovery_max_size" : "cache_max_size", ", ", base::formatBytes(limit),
             "); it is computed again when needed, to the same bits"});
        return result;
    }
    std::uint64_t total = 0;
    std::uint64_t recovery = 0;
    for (const CacheEntryRecord& entry : entries)
    {
        total += entry.bytes;
        recovery += entry.kind == CacheEntryKind::RecoveryData ? entry.bytes : 0;
    }
    const auto exceedsTotal = [&] { return total + added.bytes > limits.totalBytes; };
    const auto exceedsRecovery = [&] { return isRecovery && recovery + added.bytes > limits.recoveryBytes; };

    // The candidates of each group, least recently used first (ties by index: a fixed order).
    std::vector<std::size_t> order(entries.size());
    for (std::size_t k = 0; k < order.size(); ++k)
        order[k] = k;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return entries[a].lastUsed < entries[b].lastUsed; });
    const std::int64_t expiry = now - static_cast<std::int64_t>(limits.maxAge.count());
    std::vector<Eviction> candidates;
    std::vector<bool> isTaken(entries.size(), false);
    const auto take = [&](EvictionReason reason, const auto& isInGroup)
    {
        for (const std::size_t k : order)
        {
            const CacheEntryRecord& entry = entries[k];
            if (isTaken[k] || entry.isInUse || entry.keyHash == added.keyHash || !isInGroup(entry))
                continue;
            isTaken[k] = true;
            candidates.push_back({k, reason});
        }
    };
    take(EvictionReason::Superseded,
         [&](const CacheEntryRecord& entry) { return entry.unitIdentity == added.unitIdentity; });
    take(EvictionReason::Expired, [&](const CacheEntryRecord& entry) { return entry.lastUsed < expiry; });
    take(EvictionReason::LeastRecentlyUsed,
         [](const CacheEntryRecord& entry) { return entry.kind == CacheEntryKind::RecoveryData; });
    take(EvictionReason::LeastRecentlyUsed, [](const CacheEntryRecord&) { return true; });

    for (const Eviction& candidate : candidates)
    {
        if (!exceedsTotal() && !exceedsRecovery())
            break;
        const CacheEntryRecord& entry = entries[candidate.entry];
        const bool isRecoveryEntry = entry.kind == CacheEntryKind::RecoveryData;
        if (!exceedsTotal() && !isRecoveryEntry)
            continue; // only the recovery-data limit is exceeded: a reduction frees nothing under it
        result.evictions.push_back(candidate);
        total -= entry.bytes;
        recovery -= isRecoveryEntry ? entry.bytes : 0;
    }
    if (exceedsTotal() || exceedsRecovery())
    {
        result.evictions.clear();
        result.refusal = base::concatenate(
            {"not stored: the entry (", base::formatBytes(added.bytes),
             ") does not fit the cache limits even with every entry not in use removed (the entries of the running "
             "analysis are kept); raise cache_max_size",
             isRecovery ? " or cache_recovery_max_size" : "", ", or condense fewer units at once"});
        return result;
    }
    result.isStorable = true;
    return result;
}

} // namespace oos::reduction
