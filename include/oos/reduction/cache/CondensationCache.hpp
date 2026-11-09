#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/reduction/cache/CacheCleanup.hpp"
#include "oos/reduction/cache/CacheEntry.hpp"
#include "oos/reduction/cache/CacheKey.hpp"
#include "oos/reduction/cache/CacheLimits.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace oos::reduction
{

// The threads that write the files of a session's stores off the critical path (CondensationCache::Session::
// storeReductionsLater): a store is mostly the latency of the file system (about 1 ms on the reference PC
// for a small entry: the file made, written, renamed and timed), which files written side by side hide; these threads
// wait on the disk, not on the processors (one of them is the session's writer itself).
inline constexpr std::size_t CACHE_WRITER_FILE_THREADS = 4;

// The age of the listing of its folder a store decides on. A listing per store read every unit's folder:
// the square of the units' count in directory reads for a run of 10 000 units (35 s of a thousand units' 52 s of
// condensation on the reference PC). A store lists the folder again when its listing is older than this, or when its
// entry would not fit the limits without removing others (every eviction and refusal is decided on a fresh listing);
// otherwise it takes the listing it keeps up to date with its own stores and removals. Another
// process' entries of the last seconds are then not counted when no eviction is needed: the limits may be passed by
// what other processes stored meanwhile, until the next listing.
inline constexpr std::int64_t CACHE_LISTING_REFRESH_SECONDS = 30;

// How a condensation cache is opened.
struct CacheOptions
{
    // The memory taken by the entries kept in memory between analyses, by default.
    static constexpr std::uint64_t DEFAULT_MEMORY_BYTES = std::uint64_t{512} << 20U;
    // The longest a store waits for another process' cleanup or write (CacheLock).
    static constexpr std::chrono::milliseconds DEFAULT_LOCK_WAIT{std::chrono::minutes{10}};

    // The folder of the cache on disk (created when missing); none: the cache lives in memory only.
    std::optional<std::filesystem::path> folder;
    // cache_max_size, cache_recovery_max_size, cache_max_age (CacheLimits; none: the defaults).
    std::optional<std::uint64_t> maxBytes;
    std::optional<std::uint64_t> recoveryMaxBytes;
    std::optional<std::chrono::seconds> maxAge;
    std::uint64_t memoryMaxBytes = DEFAULT_MEMORY_BYTES;
    // The version of O_o_Solver (part of every key: an entry of another version is never used).
    std::string solverVersion;
    std::chrono::milliseconds lockWait = DEFAULT_LOCK_WAIT;
    // The clock of the entries' times, in seconds since the epoch; empty: the system clock (the tests give their own).
    std::function<std::int64_t()> clock;
};

// Where a condensation came from.
enum class CacheOutcome : std::uint8_t
{
    MemoryHit, // kept in memory since an earlier analysis of this process
    DiskHit,   // read from the cache folder
    Miss       // not in the cache, or refused (the cause says which and why)
};

[[nodiscard]] constexpr std::string_view cacheOutcomeName(CacheOutcome outcome) noexcept
{
    switch (outcome)
    {
    case CacheOutcome::MemoryHit:
        return "memory_hit";
    case CacheOutcome::DiskHit:
        return "disk_hit";
    case CacheOutcome::Miss:
        return "miss";
    }
    return "";
}

// The answer of a lookup: the entry, where it came from, and on a miss its cause ("not in the cache (the first
// condensation of this unit)", "the unit's revision stamp changed ...", "the cache file was refused: corrupted (its
// checksum differs)", "written by O_o_Solver 1.0.0, this is ...").
template <typename T>
struct CacheLookup
{
    std::shared_ptr<const T> entry;
    CacheOutcome outcome = CacheOutcome::Miss;
    std::string cause;

    [[nodiscard]] bool isHit() const noexcept { return entry != nullptr; }
};

// The answer of a store: whether the entry is on disk now, what was removed for it, and why it was not stored.
struct CacheStore
{
    bool isStored = false;   // on disk (or, without a folder, kept in memory)
    bool isInMemory = false; // kept in memory too
    std::string note;        // why it is not stored (the report says so)
    std::vector<std::pair<std::string, EvictionReason>> evicted; // the entries removed for it (file names)
};

// What the cache did since it was opened, in this process (hits, misses, bytes, causes; given to the caller).
struct CacheStatistics
{
    std::int64_t memoryHits = 0;
    std::int64_t diskHits = 0;
    std::int64_t misses = 0;
    std::int64_t stores = 0;
    std::int64_t refusedStores = 0; // entries not stored (too large, no room, a write that failed)
    std::int64_t evictions = 0;
    std::int64_t refusedFiles = 0; // files read and refused (another version or format, corrupted, truncated)
    std::uint64_t bytesRead = 0;
    std::uint64_t bytesWritten = 0;
    std::map<std::string, std::int64_t> missCauses; // the cause of each miss, counted
    // The listings of the folder the stores made: one per LISTING_REFRESH_SECONDS, and one for an entry
    // that needs others removed or cannot be stored.
    std::int64_t listings = 0;
};

// One entry as cache info lists it.
struct CacheEntrySummary
{
    std::string fileName;
    CacheEntryKind kind = CacheEntryKind::Reduction;
    std::int64_t unitId = 0;
    std::string modelTag;
    std::uint64_t bytes = 0;
    std::int64_t lastUsed = 0; // seconds since the epoch
};

// The content of a cache (size per kind, units, hits, oldest entries).
struct CacheInfo
{
    std::optional<std::filesystem::path> folder;
    CacheLimits limits;
    std::uint64_t reductionBytes = 0;
    std::uint64_t recoveryBytes = 0;
    std::int64_t reductionEntries = 0;
    std::int64_t recoveryEntries = 0;
    std::int64_t unitCount = 0; // distinct units (identities)
    std::uint64_t memoryBytes = 0;
    std::int64_t memoryEntries = 0;
    std::vector<CacheEntrySummary> oldest; // the least recently used, at most OLDEST_LISTED
    CacheStatistics statistics;

    static constexpr std::size_t OLDEST_LISTED = 10;
};

// What clear() removes (by kind, age or model): every entry when nothing is given; the entries of one kind; those last
// used before a time; those of one model.
struct CacheClearFilter
{
    std::optional<CacheEntryKind> kind;
    std::optional<std::int64_t> usedBefore; // seconds since the epoch
    std::optional<std::string> modelTag;
};

struct CacheClearResult
{
    std::int64_t removedEntries = 0;
    std::uint64_t removedBytes = 0;
    std::int64_t keptInUse = 0; // matched, but used by a running analysis of this process (never removed)
};

// An entry file that does not read back (CondensationCache::verify), and why.
struct CacheFileProblem
{
    std::string fileName;
    std::string refusal;
};

// The condensation cache: never condense twice what has not changed, and store only what is light.
// - What it holds: per key (CacheKey), a CachedReduction (K*, f*_k, the inertial patterns, the rigid-body mass
//   properties, the pattern work, the pivot ratio) or CachedRecoveryData (the Craig-Bampton modes); never an interior
//   factor, never a recovery matrix (both are large).
// - In memory: the entries of this process, up to CacheOptions::memoryMaxBytes (their own budget), the least recently
//   used dropped first, never one a running analysis uses. On disk (with a folder): one CacheFile per entry,
//   "<folder>/entries/<R|M>-<unit identity>-<key hash>.nsvc" (R a reduction, M recovery data): the file name is the
//   index by key, a lookup opens it directly, and the entries of a unit are found by their prefix to name why they do
//   not serve. A file's time is its last use (written or read: the least recently used ones go first).
// - A lookup never takes an entry silently when it does not serve: the key of the file must match in full and its
//   checksum hold; otherwise the miss says why (a version, a kernel, a parameter, the unit's stamps, a corrupted or
//   truncated file, which is then removed).
// - A store cleans the cache up under the lock (CacheLock) by CacheCleanup's order, never removing an entry an
//   analysis of this process uses, then writes the entry through a temporary file; an entry larger than its limit is
//   not stored and the store says so. A failure to store is a note, never an error of the analysis: the entry is
//   computed again when needed, to the same bits.
// The entries an analysis uses are pinned by its Session for the time of the analysis.
//
// Thread-safety: every member may be called from several threads (a mutex for the memory layer and the statistics,
// the lock for the folder). Ownership: owns the memory layer and its budget; must outlive its sessions and the entries
// it hands out.
class CondensationCache
{
public:
    // Throws std::invalid_argument for options that cannot be met (no solver version, a recovery limit above the total
    // one), base::FileError when the folder cannot be created.
    explicit CondensationCache(CacheOptions options);
    CondensationCache(const CondensationCache&) = delete;
    CondensationCache& operator=(const CondensationCache&) = delete;
    ~CondensationCache();

    class Session;
    // A session of an analysis: its lookups and stores, the entries it uses pinned until it ends. `modelTag` names the
    // model in the entries it stores (clear(model=...)).
    [[nodiscard]] Session openSession(std::string modelTag);

    [[nodiscard]] const std::optional<std::filesystem::path>& folder() const noexcept { return m_options.folder; }
    [[nodiscard]] const CacheLimits& limits() const noexcept { return m_limits; }
    [[nodiscard]] const std::string& solverVersion() const noexcept { return m_options.solverVersion; }
    [[nodiscard]] CacheInfo info() const;
    [[nodiscard]] CacheStatistics statistics() const;
    // Removes the entries the filter selects, on disk and in memory, but those a running analysis uses.
    CacheClearResult clear(const CacheClearFilter& filter);
    // Reads every entry of the folder in full (its size and checksum) and lists those that do not read back, without
    // removing them (a lookup removes a broken entry it meets). Each entry is held in memory while it is read.
    [[nodiscard]] std::vector<CacheFileProblem> verify() const;

private:
    struct MemoryEntry
    {
        CacheKey key;
        CacheEntryKind kind = CacheEntryKind::Reduction;
        std::shared_ptr<const CachedReduction> reduction;
        std::shared_ptr<const CachedRecoveryData> recovery;
        std::uint64_t bytes = 0;
        std::uint64_t lastUse = 0;   // the use counter at its last use (the order of the memory layer)
        std::int64_t lastUsedAt = 0; // seconds since the epoch (clear(usedBefore))
        std::string modelTag;        // the model of the session that stored it ("" when read from the folder)
    };
    struct DiskEntry
    {
        std::filesystem::path path;
        CacheEntryRecord record;
    };
    using EntryId = std::pair<CacheEntryKind, std::uint64_t>; // the kind and the key hash

    [[nodiscard]] std::int64_t now() const;
    [[nodiscard]] std::filesystem::path entryPath(CacheEntryKind kind, const CacheKey& key) const;
    [[nodiscard]] std::vector<DiskEntry> listEntries() const;
    void touch(const std::filesystem::path& path) const;
    [[nodiscard]] std::string missCause(CacheEntryKind kind, const CacheKey& key) const;
    // Makes room in the memory layer for `bytes` (dropping unpinned entries, least recently used first); false when it
    // cannot. Called under m_mutex.
    bool makeMemoryRoom(std::uint64_t bytes);
    template <typename T>
    CacheLookup<T> lookup(CacheEntryKind kind, const CacheKey& key, base::MemoryBudget& budget,
                          std::vector<EntryId>& pins);
    template <typename T>
    CacheStore store(CacheEntryKind kind, const CacheKey& key, const T& entry, const std::string& modelTag,
                     std::vector<EntryId>& pins);
    // store() of each entry, in their order (the memory layer, the listing, its evictions and refusals decided entry
    // after entry as store() decides them), under one CacheLock for all; their files written on `pool` (null: the
    // calling thread), each its own file.
    template <typename T>
    std::vector<CacheStore> storeAll(CacheEntryKind kind, std::span<const std::pair<const CacheKey*, const T*>> entries,
                                     const std::string& modelTag, std::vector<EntryId>& pins, base::ThreadPool* pool);
    void pin(const EntryId& id, std::vector<EntryId>& pins);
    void unpin(const std::vector<EntryId>& pins);

    CacheOptions m_options;
    CacheLimits m_limits;
    std::filesystem::path m_entries; // <folder>/entries
    mutable std::mutex m_mutex;
    base::MemoryBudget m_memory;
    std::map<EntryId, MemoryEntry> m_inMemory;
    std::map<EntryId, std::int64_t> m_pinned; // the pins of the running sessions
    std::uint64_t m_useCounter = 0;
    mutable CacheStatistics m_statistics;
    // The stores of this process one at a time: the CacheLock (a lock folder) orders the stores of different
    // processes, but a file system operation is no synchronisation of the C++ memory model, so the threads of one
    // process take this mutex first (m_listing below has its happens-before through it).
    std::mutex m_storeMutex;
    // The listing of the folder a store decides on (CACHE_LISTING_REFRESH_SECONDS), sorted by file name as
    // listEntries(). Read and written by a store holding m_storeMutex and the CacheLock (stores run one at a time).
    std::vector<DiskEntry> m_listing;
    std::optional<std::int64_t> m_listedAt;
};

// The lookups and stores of one analysis (CondensationCache::openSession). The entries it found or stored stay pinned
// (never removed from the memory layer or the folder by this process) until it ends. Its stores may be made off the
// critical path (storeReductionsLater): a writer thread of the session makes them in the order given, as
// storeReductions makes them, while the analysis goes on; the session waits for them when it ends (an abandoned
// analysis: the stores not begun are dropped, an entry being written is completed through its temporary file).
//
// Thread-safety: one session per analysis; its calls from one thread at a time (its writer is its own). Ownership:
// borrows the cache, which must outlive it; owns its writer.
class CondensationCache::Session
{
public:
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept;
    Session& operator=(Session&& other) = delete;
    ~Session();

    // The entry of `key`: from memory, else from the folder (its matrices reserved in the cache's memory budget when
    // they fit it, in `budget` otherwise; MemoryBudgetError when neither can take them), else a miss with its cause.
    [[nodiscard]] CacheLookup<CachedReduction> lookupReduction(const CacheKey& key, base::MemoryBudget& budget);
    [[nodiscard]] CacheLookup<CachedRecoveryData> lookupRecoveryData(const CacheKey& key, base::MemoryBudget& budget);
    // Stores an entry made by this analysis (copied into the memory layer when it fits, written to the folder).
    CacheStore storeReduction(const CacheKey& key, const CachedReduction& entry);
    CacheStore storeRecoveryData(const CacheKey& key, const CachedRecoveryData& entry);
    // storeReduction() of several entries (the units of a wave of condensations), in their order: the same entries
    // in memory and on disk, the same answers, under one lock of the folder, the files written on `pool`.
    // PERFORMANCE: a store is mostly the latency of the file system (the lock folder made and removed, the file
    // written and its time set: about 1.7 ms on the reference PC, 1.7 s for 1 000 units), now one lock per wave and
    // the files written beside each other.
    // Measured on a condensed analysis of a large model.
    std::vector<CacheStore> storeReductions(std::span<const std::pair<const CacheKey*, const CachedReduction*>> entries,
                                            base::ThreadPool* pool);
    // storeReductions() of the entries made by the session's writer (a thread of its own, made by the first call, with
    // CACHE_WRITER_FILE_THREADS writing the files), after the entries given before, while the caller goes on; the
    // entries are kept alive until written. Their answers: finishStores().
    // PERFORMANCE: the stores of a wave of small units (a lock of the folder, the files written, renamed and timed:
    // 0.56 of the 1.55 s of condensation of 1 000 units of 744 DOFs) made while the next waves are condensed and the
    // interface problem solved: the same files, answers and statistics, the order of their decisions kept.
    // Measured on the reanalysis of a condensed model.
    void storeReductionsLater(std::vector<std::pair<CacheKey, std::shared_ptr<const CachedReduction>>> entries);
    // The answers of every store given to storeReductionsLater since the last call, in their order, once made; rethrows
    // what a store threw (a lock of the folder not obtained in time: CacheError). With a pool (the caller's threads,
    // free by then), the batches the writer has not begun are made by the calling thread, after the writer's, their
    // files written on that pool: the same order, files and answers.
    [[nodiscard]] std::vector<CacheStore> finishStores(base::ThreadPool* pool = nullptr);

    [[nodiscard]] CondensationCache& cache() noexcept { return *m_cache; }

private:
    friend class CondensationCache;
    struct Writer;
    Session(CondensationCache& cache, std::string modelTag);

    CondensationCache* m_cache;
    std::string m_modelTag;
    std::vector<EntryId> m_pins;      // one per pin taken
    std::unique_ptr<Writer> m_writer; // made by the first storeReductionsLater
};

} // namespace oos::reduction
