#include "oos/reduction/cache/CondensationCache.hpp"

#include "oos/base/errors/CacheError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/FileError.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/reduction/cache/CacheFile.hpp"
#include "oos/reduction/cache/CacheLock.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace oos::reduction
{

namespace
{

constexpr const char* ENTRIES_FOLDER = "entries";
constexpr std::size_t HEX_DIGITS = 16;
// A temporary file of an entry older than this is the leftover of a writer that ended before renaming it (a crash): the
// longest a writer may take between two refreshes of its lock, which it holds while it writes.
constexpr std::chrono::seconds ABANDONED_TEMPORARY_AGE = CacheLock::STALE_AGE;
// A K* is stored as its lower triangle: in memory it takes up to twice its file.
constexpr std::uint64_t MEMORY_PER_FILE_BYTE = 2;

char prefixOf(CacheEntryKind kind)
{
    return kind == CacheEntryKind::Reduction ? 'R' : 'M';
}

std::string hexOf(std::uint64_t value)
{
    constexpr std::string_view DIGITS = "0123456789abcdef";
    std::string text(HEX_DIGITS, '0');
    for (std::size_t k = 0; k < HEX_DIGITS; ++k)
        text[HEX_DIGITS - 1 - k] = DIGITS[(value >> (4 * k)) & 0xFU];
    return text;
}

std::optional<std::uint64_t> valueOfHex(std::string_view text)
{
    if (text.size() != HEX_DIGITS)
        return std::nullopt;
    std::uint64_t value = 0;
    for (const char c : text)
    {
        value <<= 4U;
        if (c >= '0' && c <= '9')
            value |= static_cast<std::uint64_t>(c - '0');
        else if (c >= 'a' && c <= 'f')
            value |= static_cast<std::uint64_t>(c - 'a' + 10);
        else
            return std::nullopt;
    }
    return value;
}

// The kind and the key hash of an entry file "<R|M>-<16 hex digits>.nsvc" (nothing for another file).
std::optional<std::pair<CacheEntryKind, std::uint64_t>> parseEntryName(const std::string& name)
{
    const std::size_t length = 2 + HEX_DIGITS + CacheFile::EXTENSION.size();
    if (name.size() != length || name[1] != '-' || !name.ends_with(CacheFile::EXTENSION))
        return std::nullopt;
    CacheEntryKind kind = CacheEntryKind::Reduction;
    if (name[0] == 'M')
        kind = CacheEntryKind::RecoveryData;
    else if (name[0] != 'R')
        return std::nullopt;
    const std::optional<std::uint64_t> hash = valueOfHex(std::string_view(name).substr(2, HEX_DIGITS));
    if (!hash)
        return std::nullopt;
    return std::make_pair(kind, *hash);
}

// The file clock's epoch is not the system clock's before C++20's clock_cast (not in GCC 12): a time is converted
// through the two clocks' present, exact to their reading a few microseconds apart, far below the second the cache
// counts in.
std::int64_t secondsOf(std::filesystem::file_time_type time)
{
    const auto fileNow = std::filesystem::file_time_type::clock::now();
    const auto systemNow = std::chrono::system_clock::now();
    const auto system = systemNow + std::chrono::duration_cast<std::chrono::system_clock::duration>(time - fileNow);
    return std::chrono::duration_cast<std::chrono::seconds>(system.time_since_epoch()).count();
}

std::filesystem::file_time_type fileTimeOf(std::int64_t seconds)
{
    const auto fileNow = std::filesystem::file_time_type::clock::now();
    const auto systemNow = std::chrono::system_clock::now();
    const std::chrono::system_clock::time_point system{std::chrono::seconds{seconds}};
    return fileNow + std::chrono::duration_cast<std::filesystem::file_time_type::duration>(system - systemNow);
}

linalg::DenseMatrix copyOf(const linalg::DenseMatrix& source, base::MemoryBudget& budget, const std::string& label)
{
    linalg::DenseMatrix copy(budget, source.rows(), source.columns(), label);
    for (std::size_t j = 0; j < source.columns(); ++j)
        for (std::size_t i = 0; i < source.rows(); ++i)
            copy(i, j) = source(i, j);
    return copy;
}

std::shared_ptr<const CachedReduction> copyOf(const CachedReduction& source, base::MemoryBudget& budget)
{
    auto copy = std::make_shared<CachedReduction>();
    if (source.reduced)
    {
        const ReducedUnit& reduced = *source.reduced;
        base::BudgetedBuffer<UnitDofLabel> labels(budget, reduced.labels().size(), "cache entry labels");
        for (std::size_t k = 0; k < labels.size(); ++k)
            labels[k] = reduced.labels()[k];
        copy->reduced = std::make_shared<const ReducedUnit>(
            reduced.unitId(), std::move(labels), copyOf(reduced.stiffness(), budget, "cache entry K*"),
            copyOf(reduced.patterns(), budget, "cache entry patterns"), reduced.patternStamp(),
            copyOf(reduced.inertialPatterns(), budget, "cache entry inertial patterns"), reduced.rigidBodyMass());
    }
    copy->patternWork = copyOf(source.patternWork, budget, "cache entry pattern work");
    copy->largestPivotRatio = source.largestPivotRatio;
    copy->interiorDofCount = source.interiorDofCount;
    copy->equationStamp = source.equationStamp;
    copy->boundaryMass = copyOf(source.boundaryMass, budget, "cache entry boundary mass");
    if (source.modal)
    {
        const ModalReduction& modal = *source.modal;
        copy->modal = std::make_shared<const ModalReduction>(
            copyOf(modal.eigenvalues(), budget, "cache entry fixed-interface eigenvalues"),
            copyOf(modal.modalStiffness(), budget, "cache entry K_qq"),
            copyOf(modal.couplingMass(), budget, "cache entry M_aq"),
            copyOf(modal.modalMass(), budget, "cache entry M_qq"));
    }
    return copy;
}

std::shared_ptr<const CachedRecoveryData> copyOf(const CachedRecoveryData& source, base::MemoryBudget& budget)
{
    auto copy = std::make_shared<CachedRecoveryData>();
    copy->modes = copyOf(source.modes, budget, "cache entry modes");
    copy->eigenvalues = copyOf(source.eigenvalues, budget, "cache entry eigenvalues");
    return copy;
}

// Whether a refusal of a file says the file itself is broken (removed then), rather than unreadable now.
bool isBroken(const std::string& refusal)
{
    return refusal.starts_with("corrupted") || refusal.starts_with("truncated") ||
           refusal.starts_with("not a O_o_Solver cache file");
}

void removeQuietly(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::remove(path, error);
}

} // namespace

CondensationCache::CondensationCache(CacheOptions options)
    : m_options(std::move(options)), m_memory(m_options.memoryMaxBytes)
{
    if (m_options.solverVersion.empty())
        throw std::invalid_argument(
            "CondensationCache: the options need the version of O_o_Solver (part of every key)");
    if (m_options.folder)
    {
        m_entries = *m_options.folder / ENTRIES_FOLDER;
        std::error_code error;
        std::filesystem::create_directories(m_entries, error);
        if (error || !std::filesystem::is_directory(m_entries))
            throw base::FileError(base::concatenate(
                {"cannot create the condensation cache folder '", m_entries.string(), "': ", error.message(),
                 "; give a writable cache_dir, or cache_dir=False for a cache in memory only"}));
        std::uint64_t held = 0;
        for (const DiskEntry& entry : listEntries())
            held += entry.record.bytes;
        m_limits = CacheLimits::resolve(*m_options.folder, held, m_options.maxBytes, m_options.recoveryMaxBytes,
                                        m_options.maxAge);
    }
    else
    {
        m_limits.totalBytes = m_options.memoryMaxBytes;
        m_limits.recoveryBytes = m_options.memoryMaxBytes;
    }
}

CondensationCache::~CondensationCache() = default;

CondensationCache::Session CondensationCache::openSession(std::string modelTag)
{
    return Session(*this, std::move(modelTag));
}

std::int64_t CondensationCache::now() const
{
    if (m_options.clock)
        return m_options.clock();
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::filesystem::path CondensationCache::entryPath(CacheEntryKind kind, const CacheKey& key) const
{
    std::string name(1, prefixOf(kind));
    name.append("-").append(hexOf(key.hash())).append(CacheFile::EXTENSION);
    return m_entries / hexOf(key.unitIdentity) / name;
}

std::vector<CondensationCache::DiskEntry> CondensationCache::listEntries() const
{
    std::vector<DiskEntry> entries;
    if (!m_options.folder)
        return entries;
    std::error_code error;
    for (std::filesystem::directory_iterator unit(m_entries, error), end; !error && unit != end; unit.increment(error))
    {
        const std::optional<std::uint64_t> identity = valueOfHex(unit->path().filename().string());
        if (!identity || !unit->is_directory())
            continue;
        std::error_code inner;
        for (std::filesystem::directory_iterator file(unit->path(), inner), last; !inner && file != last;
             file.increment(inner))
        {
            const std::optional<std::pair<CacheEntryKind, std::uint64_t>> parsed =
                parseEntryName(file->path().filename().string());
            if (!parsed)
                continue;
            std::error_code sizeError;
            const std::uintmax_t bytes = file->file_size(sizeError);
            const std::filesystem::file_time_type time = file->last_write_time(sizeError);
            if (sizeError)
                continue; // removed by another process meanwhile
            DiskEntry entry;
            entry.path = file->path();
            entry.record.kind = parsed->first;
            entry.record.unitIdentity = *identity;
            entry.record.keyHash = parsed->second;
            entry.record.bytes = static_cast<std::uint64_t>(bytes);
            entry.record.lastUsed = secondsOf(time);
            entries.push_back(std::move(entry));
        }
    }
    // A fixed order whatever the file system lists (the cleanup's ties).
    std::sort(entries.begin(), entries.end(),
              [](const DiskEntry& a, const DiskEntry& b) { return a.path.filename() < b.path.filename(); });
    return entries;
}

void CondensationCache::touch(const std::filesystem::path& path) const
{
    std::error_code error;
    std::filesystem::last_write_time(path, fileTimeOf(now()), error);
}

std::string CondensationCache::missCause(CacheEntryKind kind, const CacheKey& key) const
{
    // The most recently used entry of the same unit and kind names the first part of the key that differs.
    std::optional<CacheKey> sibling;
    std::string refusal;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::uint64_t latest = 0;
        for (const auto& [id, entry] : m_inMemory)
            if (id.first == kind && entry.key.unitIdentity == key.unitIdentity && entry.lastUse >= latest)
            {
                latest = entry.lastUse;
                sibling = entry.key;
            }
    }
    if (!sibling && m_options.folder)
    {
        const std::filesystem::path unitFolder = m_entries / hexOf(key.unitIdentity);
        std::int64_t latest = std::numeric_limits<std::int64_t>::min();
        std::error_code error;
        for (std::filesystem::directory_iterator file(unitFolder, error), end; !error && file != end;
             file.increment(error))
        {
            const std::optional<std::pair<CacheEntryKind, std::uint64_t>> parsed =
                parseEntryName(file->path().filename().string());
            if (!parsed || parsed->first != kind)
                continue;
            std::error_code timeError;
            const std::int64_t used = secondsOf(file->last_write_time(timeError));
            if (timeError || used < latest)
                continue;
            const CacheFileRead<CacheFileHeader> header = CacheFile::readHeader(file->path());
            latest = used;
            if (header.isRead())
            {
                sibling = header.value->key;
                refusal.clear();
            }
            else
            {
                sibling.reset();
                refusal = header.refusal;
            }
        }
    }
    if (sibling)
    {
        const std::string difference = key.differenceFrom(*sibling);
        return difference.empty() ? std::string("not in the cache (the entry of the same key was removed)")
                                  : difference;
    }
    if (!refusal.empty())
        return base::concatenate({"an entry of the unit was refused: ", refusal});
    return "not in the cache (the first condensation of this unit)";
}

bool CondensationCache::makeMemoryRoom(std::uint64_t bytes)
{
    if (bytes > m_memory.limit())
        return false;
    while (m_memory.available() < bytes)
    {
        // The least recently used entry no session pins.
        auto oldest = m_inMemory.end();
        for (auto it = m_inMemory.begin(); it != m_inMemory.end(); ++it)
            if (!m_pinned.contains(it->first) &&
                (oldest == m_inMemory.end() || it->second.lastUse < oldest->second.lastUse))
                oldest = it;
        if (oldest == m_inMemory.end())
            return false;
        m_inMemory.erase(oldest);
    }
    return true;
}

void CondensationCache::pin(const EntryId& id, std::vector<EntryId>& pins)
{
    ++m_pinned[id];
    pins.push_back(id);
}

void CondensationCache::unpin(const std::vector<EntryId>& pins)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const EntryId& id : pins)
    {
        const auto found = m_pinned.find(id);
        if (found != m_pinned.end() && --found->second <= 0)
            m_pinned.erase(found);
    }
}

namespace
{

template <typename T>
CacheFileRead<std::shared_ptr<const T>> readEntry(const std::filesystem::path& path, const CacheKey& key,
                                                  base::MemoryBudget& budget)
{
    if constexpr (std::is_same_v<T, CachedReduction>)
        return CacheFile::readReduction(path, key, budget);
    else
        return CacheFile::readRecoveryData(path, key, budget);
}

} // namespace

template <typename T>
CacheLookup<T> CondensationCache::lookup(CacheEntryKind kind, const CacheKey& key, base::MemoryBudget& budget,
                                         std::vector<EntryId>& pins)
{
    const EntryId id{kind, key.hash()};
    CacheLookup<T> result;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto found = m_inMemory.find(id);
        if (found != m_inMemory.end() && key.matches(found->second.key))
        {
            found->second.lastUse = ++m_useCounter;
            found->second.lastUsedAt = now();
            if constexpr (std::is_same_v<T, CachedReduction>)
                result.entry = found->second.reduction;
            else
                result.entry = found->second.recovery;
            result.outcome = CacheOutcome::MemoryHit;
            pin(id, pins);
            ++m_statistics.memoryHits;
        }
    }
    if (result.isHit())
    {
        if (m_options.folder)
            touch(entryPath(kind, key));
        return result;
    }
    if (m_options.folder)
    {
        const std::filesystem::path path = entryPath(kind, key);
        std::error_code error;
        const std::uintmax_t fileBytes = std::filesystem::file_size(path, error);
        if (!error)
        {
            bool isKeptInMemory = false;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                isKeptInMemory = makeMemoryRoom(MEMORY_PER_FILE_BYTE * static_cast<std::uint64_t>(fileBytes));
            }
            CacheFileRead<std::shared_ptr<const T>> read;
            if (isKeptInMemory)
            {
                try
                {
                    read = readEntry<T>(path, key, m_memory);
                }
                catch (const base::MemoryBudgetError&)
                {
                    isKeptInMemory = false; // another thread took the room meanwhile
                }
            }
            if (!isKeptInMemory)
                read = readEntry<T>(path, key, budget);
            if (read.isRead())
            {
                touch(path);
                std::lock_guard<std::mutex> lock(m_mutex);
                result.entry = *read.value;
                result.outcome = CacheOutcome::DiskHit;
                ++m_statistics.diskHits;
                m_statistics.bytesRead += read.bytesRead;
                if (isKeptInMemory)
                {
                    MemoryEntry entry;
                    entry.key = key;
                    entry.kind = kind;
                    if constexpr (std::is_same_v<T, CachedReduction>)
                        entry.reduction = result.entry;
                    else
                        entry.recovery = result.entry;
                    entry.bytes = result.entry->byteCount();
                    entry.lastUse = ++m_useCounter;
                    entry.lastUsedAt = now();
                    m_inMemory[id] = std::move(entry);
                }
                pin(id, pins);
                return result;
            }
            result.cause = base::concatenate({"the cache file was refused: ", read.refusal});
            if (isBroken(read.refusal))
                removeQuietly(path);
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_statistics.refusedFiles;
            ++m_statistics.misses;
            ++m_statistics.missCauses[result.cause];
            return result;
        }
    }
    result.cause = missCause(kind, key);
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_statistics.misses;
    ++m_statistics.missCauses[result.cause];
    return result;
}

template <typename T>
CacheStore CondensationCache::store(CacheEntryKind kind, const CacheKey& key, const T& entry,
                                    const std::string& modelTag, std::vector<EntryId>& pins)
{
    const std::pair<const CacheKey*, const T*> one{&key, &entry};
    return storeAll<T>(kind, std::span<const std::pair<const CacheKey*, const T*>>(&one, 1), modelTag, pins, nullptr)
        .front();
}

template <typename T>
std::vector<CacheStore> CondensationCache::storeAll(CacheEntryKind kind,
                                                    std::span<const std::pair<const CacheKey*, const T*>> entries,
                                                    const std::string& modelTag, std::vector<EntryId>& pins,
                                                    base::ThreadPool* pool)
{
    std::vector<CacheStore> results(entries.size()); // bounded by the entries given
    // The memory layer: a copy in the cache's own budget, when it fits.
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const CacheKey& key = *entries[i].first;
        const T& entry = *entries[i].second;
        CacheStore& result = results[i];
        const EntryId id{kind, key.hash()};
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_inMemory.contains(id))
        {
            result.isInMemory = true;
        }
        else if (const std::uint64_t bytes = entry.byteCount(); makeMemoryRoom(bytes))
        {
            try
            {
                MemoryEntry kept;
                kept.key = key;
                kept.kind = kind;
                if constexpr (std::is_same_v<T, CachedReduction>)
                    kept.reduction = copyOf(entry, m_memory);
                else
                    kept.recovery = copyOf(entry, m_memory);
                kept.bytes = bytes;
                kept.lastUse = ++m_useCounter;
                kept.lastUsedAt = now();
                kept.modelTag = modelTag;
                m_inMemory[id] = std::move(kept);
                result.isInMemory = true;
            }
            catch (const base::MemoryBudgetError&)
            {
                result.isInMemory = false;
            }
        }
        pin(id, pins);
    }
    if (!m_options.folder)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            CacheStore& result = results[i];
            result.isStored = result.isInMemory;
            if (result.isStored)
            {
                ++m_statistics.stores;
            }
            else
            {
                result.note = base::concatenate(
                    {"not stored: the entry (", base::formatBytes(entries[i].second->byteCount()),
                     ") does not fit the memory of the cache (", base::formatBytes(m_options.memoryMaxBytes),
                     ", the entries in use kept); it is computed again when needed, to the same bits"});
                ++m_statistics.refusedStores;
            }
        }
        return results;
    }

    // The folder: each entry decided in order under one lock (store() decided them one lock each), then the files of
    // those to write written beside each other, each its own.
    struct Planned
    {
        std::size_t entry = 0;
        std::filesystem::path path;
        CacheFileHeader header;
        std::int64_t removed = 0;   // the entries removed to make its room
        std::int64_t touchedAt = 0; // the time its file is given once written (store() took now() after the write)
    };
    std::vector<Planned> planned;                           // bounded by the entries given
    std::vector<DiskEntry> pending;                         // the planned entries, not written yet
    std::vector<std::uint8_t> isDecided(entries.size(), 0); // bounded by the entries given
    try
    {
        const std::lock_guard<std::mutex> store(m_storeMutex);
        CacheLock lock(*m_options.folder, m_options.lockWait);
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const CacheKey& key = *entries[i].first;
            const T& entry = *entries[i].second;
            CacheStore& result = results[i];
            const EntryId id{kind, key.hash()};
            CacheFileHeader header;
            header.kind = kind;
            header.createdAt = now();
            header.modelTag = modelTag;
            header.key = key;
            const std::filesystem::path path = entryPath(kind, key);
            std::error_code error;
            if (std::filesystem::exists(path, error))
            {
                touch(path); // stored meanwhile by another process: the same bits
                result.isStored = true;
                isDecided[i] = 1;
                continue;
            }
            CacheEntryRecord added;
            added.kind = kind;
            added.unitIdentity = key.unitIdentity;
            added.keyHash = id.second;
            added.bytes = CacheFile::fileBytes(header, entry);
            added.lastUsed = header.createdAt;
            added.isInUse = true;
            // The listing (m_listing): this store's own, kept up to date, unless it is old or the entry needs room.
            const auto planOn = [&](bool isListedAgain, std::vector<CacheEntryRecord>& records)
            {
                if (isListedAgain)
                {
                    // The folder listed again, with the entries this call planned and has not written yet.
                    m_listing = listEntries();
                    m_listedAt = header.createdAt;
                    for (const DiskEntry& waiting : pending)
                    {
                        const auto at = std::lower_bound(m_listing.begin(), m_listing.end(), waiting,
                                                         [](const DiskEntry& a, const DiskEntry& b)
                                                         { return a.path.filename() < b.path.filename(); });
                        if (at == m_listing.end() || at->path.filename() != waiting.path.filename())
                            m_listing.insert(at, waiting);
                    }
                }
                records.clear();
                records.reserve(m_listing.size());
                std::lock_guard<std::mutex> guard(m_mutex);
                m_statistics.listings += isListedAgain ? 1 : 0;
                for (DiskEntry& disk : m_listing)
                {
                    disk.record.isInUse = m_pinned.contains({disk.record.kind, disk.record.keyHash});
                    records.push_back(disk.record);
                }
                return CacheCleanup::plan(records, added, m_limits, header.createdAt);
            };
            const bool isStale = !m_listedAt || header.createdAt - *m_listedAt > CACHE_LISTING_REFRESH_SECONDS ||
                                 header.createdAt < *m_listedAt;
            std::vector<CacheEntryRecord> records;
            CleanupPlan plan = planOn(isStale, records);
            if (!isStale && (!plan.isStorable || !plan.evictions.empty()))
                plan = planOn(true, records);
            std::vector<DiskEntry>& listed = m_listing;
            if (!plan.isStorable)
            {
                result.note = plan.refusal;
                isDecided[i] = 1;
                std::lock_guard<std::mutex> guard(m_mutex);
                ++m_statistics.refusedStores;
                continue;
            }
            // Remove what the plan names; a file another process holds open cannot be removed on Windows: it stays,
            // and the entry is stored only if the limits still hold without it.
            std::uint64_t total = 0;
            std::uint64_t recovery = 0;
            for (const CacheEntryRecord& record : records)
            {
                total += record.bytes;
                recovery += record.kind == CacheEntryKind::RecoveryData ? record.bytes : 0;
            }
            std::int64_t removed = 0;
            std::vector<std::size_t> removedEntries; // bounded by the entries of the folder
            for (const Eviction& eviction : plan.evictions)
            {
                const DiskEntry& victim = listed[eviction.entry];
                std::error_code removeError;
                if (!std::filesystem::remove(victim.path, removeError) || removeError)
                    continue;
                ++removed;
                removedEntries.push_back(eviction.entry);
                total -= victim.record.bytes;
                recovery -= victim.record.kind == CacheEntryKind::RecoveryData ? victim.record.bytes : 0;
                result.evicted.emplace_back(victim.path.filename().string(), eviction.reason);
                std::filesystem::remove(victim.path.parent_path(), removeError); // the unit's folder, when now empty
            }
            // The listing without the entries removed (from the last, so that the indexes hold).
            std::sort(removedEntries.begin(), removedEntries.end());
            for (auto victim = removedEntries.rbegin(); victim != removedEntries.rend(); ++victim)
                m_listing.erase(m_listing.begin() + static_cast<std::ptrdiff_t>(*victim));
            const bool isRecovery = kind == CacheEntryKind::RecoveryData;
            if (total + added.bytes > m_limits.totalBytes ||
                (isRecovery && recovery + added.bytes > m_limits.recoveryBytes))
            {
                result.note =
                    "not stored: an entry to remove for it is in use by another process; it is computed again "
                    "when needed, to the same bits";
                isDecided[i] = 1;
                std::lock_guard<std::mutex> guard(m_mutex);
                ++m_statistics.refusedStores;
                m_statistics.evictions += removed;
                continue;
            }
            // The listing with the entry to be stored, in its place by file name (taken out again if its write fails),
            // so that the next entries plan with it as store() planned with the entries stored before.
            DiskEntry stored;
            stored.path = path;
            stored.record = added;
            stored.record.isInUse = false;
            const auto place =
                std::lower_bound(m_listing.begin(), m_listing.end(), stored, [](const DiskEntry& a, const DiskEntry& b)
                                 { return a.path.filename() < b.path.filename(); });
            pending.push_back(stored);
            m_listing.insert(place, std::move(stored));
            planned.push_back({i, path, std::move(header), removed, now()});
        }
        // The files, each written and timed by one task: the lock is refreshed between their sections (one at a time).
        std::vector<std::uint64_t> written(planned.size(), 0); // bounded by the entries given
        std::vector<std::string> failures(planned.size());     // bounded by the entries given
        std::mutex refreshing;
        const auto refresh = [&]
        {
            const std::lock_guard<std::mutex> guard(refreshing);
            lock.refresh();
        };
        base::parallelFor(pool, planned.size(), 1,
                          [&](std::size_t first, std::size_t end)
                          {
                              for (std::size_t k = first; k < end; ++k)
                              {
                                  try
                                  {
                                      // The unit's folder (another entry of the unit may make it at the same time).
                                      std::error_code folderError;
                                      std::filesystem::create_directories(planned[k].path.parent_path(), folderError);
                                      written[k] = CacheFile::write(planned[k].path, planned[k].header,
                                                                    *entries[planned[k].entry].second, refresh);
                                      // touch(), with the time taken in order above (the clock is not called here).
                                      std::error_code touchError;
                                      std::filesystem::last_write_time(planned[k].path,
                                                                       fileTimeOf(planned[k].touchedAt), touchError);
                                  }
                                  catch (const base::SolverError& error) // FileError (the write)
                                  {
                                      failures[k] = error.what();
                                  }
                              }
                          });
        for (std::size_t k = 0; k < planned.size(); ++k)
        {
            CacheStore& result = results[planned[k].entry];
            isDecided[planned[k].entry] = 1;
            std::lock_guard<std::mutex> guard(m_mutex);
            if (failures[k].empty())
            {
                result.isStored = true;
                ++m_statistics.stores;
                m_statistics.bytesWritten += written[k];
                m_statistics.evictions += planned[k].removed;
                continue;
            }
            result.note = base::concatenate({"not stored: ", failures[k]});
            ++m_statistics.refusedStores;
            const auto listed = std::find_if(m_listing.begin(), m_listing.end(), [&](const DiskEntry& disk)
                                             { return disk.path.filename() == planned[k].path.filename(); });
            if (listed != m_listing.end())
                m_listing.erase(listed);
        }
    }
    catch (const base::SolverError& error) // CacheError (the lock)
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            if (isDecided[i] != 0)
                continue;
            results[i].isStored = false;
            results[i].note = base::concatenate({"not stored: ", error.what()});
            ++m_statistics.refusedStores;
        }
    }
    return results;
}

CacheInfo CondensationCache::info() const
{
    CacheInfo info;
    info.folder = m_options.folder;
    info.limits = m_limits;
    std::vector<DiskEntry> entries = listEntries();
    std::set<std::uint64_t> units;
    for (const DiskEntry& entry : entries)
    {
        units.insert(entry.record.unitIdentity);
        if (entry.record.kind == CacheEntryKind::Reduction)
        {
            info.reductionBytes += entry.record.bytes;
            ++info.reductionEntries;
        }
        else
        {
            info.recoveryBytes += entry.record.bytes;
            ++info.recoveryEntries;
        }
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const DiskEntry& a, const DiskEntry& b) { return a.record.lastUsed < b.record.lastUsed; });
    for (std::size_t k = 0; k < entries.size() && k < CacheInfo::OLDEST_LISTED; ++k)
    {
        CacheEntrySummary summary;
        summary.fileName = entries[k].path.filename().string();
        summary.kind = entries[k].record.kind;
        summary.bytes = entries[k].record.bytes;
        summary.lastUsed = entries[k].record.lastUsed;
        const CacheFileRead<CacheFileHeader> header = CacheFile::readHeader(entries[k].path);
        if (header.isRead())
        {
            summary.unitId = header.value->key.unitId;
            summary.modelTag = header.value->modelTag;
        }
        info.oldest.push_back(std::move(summary));
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& [id, entry] : m_inMemory)
    {
        info.memoryBytes += entry.bytes;
        ++info.memoryEntries;
        if (!m_options.folder)
        {
            units.insert(entry.key.unitIdentity);
            if (id.first == CacheEntryKind::Reduction)
            {
                info.reductionBytes += entry.bytes;
                ++info.reductionEntries;
            }
            else
            {
                info.recoveryBytes += entry.bytes;
                ++info.recoveryEntries;
            }
        }
    }
    info.unitCount = static_cast<std::int64_t>(units.size());
    info.statistics = m_statistics;
    return info;
}

CacheStatistics CondensationCache::statistics() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_statistics;
}

CacheClearResult CondensationCache::clear(const CacheClearFilter& filter)
{
    CacheClearResult result;
    const auto isSelected = [&](CacheEntryKind kind, std::int64_t lastUsed)
    { return (!filter.kind || *filter.kind == kind) && (!filter.usedBefore || lastUsed < *filter.usedBefore); };
    std::set<EntryId> removedIds;
    if (m_options.folder)
    {
        CacheLock lock(*m_options.folder, m_options.lockWait);
        for (const DiskEntry& entry : listEntries())
        {
            if (!isSelected(entry.record.kind, entry.record.lastUsed))
                continue;
            if (filter.modelTag)
            {
                const CacheFileRead<CacheFileHeader> header = CacheFile::readHeader(entry.path);
                if (!header.isRead() || header.value->modelTag != *filter.modelTag)
                    continue;
            }
            {
                std::lock_guard<std::mutex> guard(m_mutex);
                if (m_pinned.contains({entry.record.kind, entry.record.keyHash}))
                {
                    ++result.keptInUse;
                    continue;
                }
            }
            std::error_code error;
            if (std::filesystem::remove(entry.path, error) && !error)
            {
                ++result.removedEntries;
                result.removedBytes += entry.record.bytes;
                removedIds.insert({entry.record.kind, entry.record.keyHash});
                std::filesystem::remove(entry.path.parent_path(), error); // the unit's folder, when now empty
            }
        }
        // The temporary files a writer left when it ended before renaming them.
        std::error_code error;
        const std::int64_t abandoned = now() - ABANDONED_TEMPORARY_AGE.count();
        for (std::filesystem::recursive_directory_iterator file(m_entries, error), end; !error && file != end;
             file.increment(error))
        {
            std::error_code timeError;
            if (file->is_regular_file(timeError) &&
                file->path().filename().string().find(".tmp-") != std::string::npos &&
                secondsOf(file->last_write_time(timeError)) < abandoned && !timeError)
                removeQuietly(file->path());
        }
    }
    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto it = m_inMemory.begin(); it != m_inMemory.end();)
    {
        // With a folder the files decide (a memory copy goes with its file); in memory only, the filter.
        const MemoryEntry& entry = it->second;
        const bool isMatch = m_options.folder ? removedIds.contains(it->first)
                                              : isSelected(it->first.first, entry.lastUsedAt) &&
                                                    (!filter.modelTag || entry.modelTag == *filter.modelTag);
        if (!m_options.folder && isMatch && m_pinned.contains(it->first))
            ++result.keptInUse;
        if (!isMatch || m_pinned.contains(it->first))
        {
            ++it;
            continue;
        }
        if (!m_options.folder)
        {
            ++result.removedEntries;
            result.removedBytes += it->second.bytes;
        }
        it = m_inMemory.erase(it);
    }
    return result;
}

std::vector<CacheFileProblem> CondensationCache::verify() const
{
    std::vector<CacheFileProblem> problems;
    for (const DiskEntry& entry : listEntries())
    {
        const CacheFileRead<CacheFileHeader> header = CacheFile::readHeader(entry.path);
        std::string refusal = header.refusal;
        if (header.isRead())
        {
            base::MemoryBudget scratch(base::MemoryBudget::UNLIMITED);
            refusal = header.value->kind == CacheEntryKind::Reduction
                          ? CacheFile::readReduction(entry.path, header.value->key, scratch).refusal
                          : CacheFile::readRecoveryData(entry.path, header.value->key, scratch).refusal;
            if (refusal.empty() && entry.record.keyHash != header.value->key.hash())
                refusal = "its key is not the one its name says";
        }
        if (!refusal.empty())
            problems.push_back({entry.path.filename().string(), std::move(refusal)});
    }
    return problems;
}

// The writer of a session's stores off the critical path (storeReductionsLater): a queue of batches, made one after
// the other by its thread through storeAll (the decisions of a batch in order under one lock of the folder, its files
// written by `files`), their answers appended in order. Its pins are its own until the session takes them
// (finishStores, the session's end): the session's are the calling thread's.
struct CondensationCache::Session::Writer
{
    using Batch = std::vector<std::pair<CacheKey, std::shared_ptr<const CachedReduction>>>;

    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Batch> queue;         // guarded by mutex
    bool isBusy = false;             // a batch is being made (guarded by mutex)
    bool isClosing = false;          // the session ends: the batches not begun are dropped (guarded by mutex)
    std::vector<CacheStore> answers; // guarded by mutex
    std::vector<EntryId> pins;       // guarded by mutex
    std::exception_ptr failure;      // the first exception of a batch (guarded by mutex); the next batches dropped
    base::ThreadPool files{CACHE_WRITER_FILE_THREADS};
    std::string modelTag; // the session's (a copy: the session may be moved)
    std::jthread thread;  // declared last: joined first, before the members it uses go

    Writer(CondensationCache& cache, std::string tag) : modelTag(std::move(tag))
    {
#if defined(OOS_WASM)
        // WebAssembly has no threads: no writer thread is started, and finishStores makes the queued batches itself.
        (void)cache;
#else
        thread = std::jthread([this, &cache] { run(cache); });
#endif
    }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer()
    {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            isClosing = true;
            queue.clear();
        }
        changed.notify_all();
    }

    void run(CondensationCache& cache)
    {
        while (true)
        {
            Batch batch;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return isClosing || !queue.empty(); });
                if (queue.empty())
                    return;
                batch = std::move(queue.front());
                queue.pop_front();
                isBusy = true;
            }
            std::vector<std::pair<const CacheKey*, const CachedReduction*>> entries; // bounded by the batch
            entries.reserve(batch.size());
            for (const auto& [key, entry] : batch)
                entries.emplace_back(&key, entry.get());
            std::vector<EntryId> batchPins;
            std::vector<CacheStore> batchAnswers;
            std::exception_ptr batchFailure;
            try
            {
                batchAnswers =
                    cache.storeAll<CachedReduction>(CacheEntryKind::Reduction, entries, modelTag, batchPins, &files);
            }
            catch (...)
            {
                batchFailure = std::current_exception();
            }
            {
                const std::lock_guard<std::mutex> lock(mutex);
                answers.insert(answers.end(), std::make_move_iterator(batchAnswers.begin()),
                               std::make_move_iterator(batchAnswers.end()));
                pins.insert(pins.end(), batchPins.begin(), batchPins.end());
                if (batchFailure && !failure)
                    failure = batchFailure;
                if (failure)
                    queue.clear();
                isBusy = false;
            }
            changed.notify_all();
        }
    }
};

CondensationCache::Session::Session(CondensationCache& cache, std::string modelTag)
    : m_cache(&cache), m_modelTag(std::move(modelTag))
{
}

CondensationCache::Session::Session(Session&& other) noexcept
    : m_cache(other.m_cache),
      m_modelTag(std::move(other.m_modelTag)),
      m_pins(std::move(other.m_pins)),
      m_writer(std::move(other.m_writer))
{
    other.m_pins.clear();
}

CondensationCache::Session::~Session()
{
    if (m_writer)
    {
        // The batches not begun dropped, the one being made completed (its files through their temporary files).
        std::vector<EntryId> writerPins;
        {
            const std::lock_guard<std::mutex> lock(m_writer->mutex);
            m_writer->isClosing = true;
            m_writer->queue.clear();
        }
        m_writer->changed.notify_all();
        if (m_writer->thread.joinable())
            m_writer->thread.join();
        writerPins = std::move(m_writer->pins);
        m_writer.reset();
        m_pins.insert(m_pins.end(), writerPins.begin(), writerPins.end());
    }
    if (m_cache != nullptr && !m_pins.empty())
        m_cache->unpin(m_pins);
}

void CondensationCache::Session::storeReductionsLater(
    std::vector<std::pair<CacheKey, std::shared_ptr<const CachedReduction>>> entries)
{
    if (entries.empty())
        return;
    if (!m_writer)
        m_writer = std::make_unique<Writer>(*m_cache, m_modelTag);
    {
        const std::lock_guard<std::mutex> lock(m_writer->mutex);
        m_writer->queue.push_back(std::move(entries));
    }
    m_writer->changed.notify_all();
}

std::vector<CacheStore> CondensationCache::Session::finishStores(base::ThreadPool* pool)
{
    if (!m_writer)
        return {};
    std::vector<CacheStore> answers;
    std::exception_ptr failure;
    std::deque<Writer::Batch> remaining;
    {
        std::unique_lock<std::mutex> lock(m_writer->mutex);
        // PERFORMANCE: the batches the writer has not begun are made here, their files written by the caller's pool:
        // at the end of an analysis every thread is free, and the stores of large units (their K* compressed, a
        // processor's work) left in the queue would wait for the writer's CACHE_WRITER_FILE_THREADS.
        // Measured on the reanalysis of a condensed model.
        // Without a writer thread (WebAssembly), every batch is made here.
        if (pool != nullptr || !m_writer->thread.joinable())
            remaining.swap(m_writer->queue);
        m_writer->changed.wait(lock, [&] { return m_writer->queue.empty() && !m_writer->isBusy; });
        answers = std::move(m_writer->answers);
        m_writer->answers.clear();
        m_pins.insert(m_pins.end(), m_writer->pins.begin(), m_writer->pins.end());
        m_writer->pins.clear();
        failure = std::exchange(m_writer->failure, nullptr);
    }
    if (failure)
        std::rethrow_exception(failure);
    for (const Writer::Batch& batch : remaining)
    {
        std::vector<std::pair<const CacheKey*, const CachedReduction*>> entries; // bounded by the batch
        entries.reserve(batch.size());
        for (const auto& [key, entry] : batch)
            entries.emplace_back(&key, entry.get());
        std::vector<CacheStore> batchAnswers =
            m_cache->storeAll<CachedReduction>(CacheEntryKind::Reduction, entries, m_modelTag, m_pins, pool);
        answers.insert(answers.end(), std::make_move_iterator(batchAnswers.begin()),
                       std::make_move_iterator(batchAnswers.end()));
    }
    return answers;
}

CacheLookup<CachedReduction> CondensationCache::Session::lookupReduction(const CacheKey& key,
                                                                         base::MemoryBudget& budget)
{
    return m_cache->lookup<CachedReduction>(CacheEntryKind::Reduction, key, budget, m_pins);
}

CacheLookup<CachedRecoveryData> CondensationCache::Session::lookupRecoveryData(const CacheKey& key,
                                                                               base::MemoryBudget& budget)
{
    return m_cache->lookup<CachedRecoveryData>(CacheEntryKind::RecoveryData, key, budget, m_pins);
}

CacheStore CondensationCache::Session::storeReduction(const CacheKey& key, const CachedReduction& entry)
{
    return m_cache->store(CacheEntryKind::Reduction, key, entry, m_modelTag, m_pins);
}

CacheStore CondensationCache::Session::storeRecoveryData(const CacheKey& key, const CachedRecoveryData& entry)
{
    return m_cache->store(CacheEntryKind::RecoveryData, key, entry, m_modelTag, m_pins);
}

std::vector<CacheStore> CondensationCache::Session::storeReductions(
    std::span<const std::pair<const CacheKey*, const CachedReduction*>> entries, base::ThreadPool* pool)
{
    return m_cache->storeAll<CachedReduction>(CacheEntryKind::Reduction, entries, m_modelTag, m_pins, pool);
}

} // namespace oos::reduction
