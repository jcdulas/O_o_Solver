// The checks of the folder of the condensation cache (reduction/cache): a store replaces a file that stands at the
// path of its entry without being that entry (another key of the same file name, another format), which every lookup
// refuses; and a lock is released by its holder only, never the lock of the waiter that broke it and took its place.

#include "CheckRunner.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/reduction/cache/CacheFile.hpp"
#include "oos/reduction/cache/CacheLock.hpp"
#include "oos/reduction/cache/CondensationCache.hpp"
#include "oos/reduction/condense/ReducedUnit.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <utility>

namespace
{

namespace reduction = oos::reduction;

constexpr const char* SOLVER_VERSION = "check";
constexpr std::size_t BOUNDARY_DOFS = 2;
constexpr std::size_t PATTERN_COUNT = 1;
constexpr std::size_t INERTIAL_PATTERN_COUNT = 6;
constexpr std::chrono::milliseconds LOCK_WAIT{2000};

// A folder of its own under the temporary folder of the machine, removed with what it holds when the check ends.
class ScratchFolder
{
public:
    explicit ScratchFolder(const std::string& name)
    {
        std::random_device device;
        m_path = std::filesystem::temp_directory_path() /
                 ("oos-check-" + name + "-" + std::to_string(device()) + "-" + std::to_string(device()));
        std::filesystem::create_directories(m_path);
    }
    ScratchFolder(const ScratchFolder&) = delete;
    ScratchFolder& operator=(const ScratchFolder&) = delete;
    ~ScratchFolder()
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

private:
    std::filesystem::path m_path;
};

// The key of one unit at revision `revision`: two revisions are two keys of the same unit (the same folder).
reduction::CacheKey keyOf(std::int64_t revision)
{
    reduction::CacheKey key;
    key.solverVersion = SOLVER_VERSION;
    key.kernel = "reference";
    key.unitId = 1;
    key.unitIdentity = 7;
    key.revision = revision;
    key.inputStamp = 11;
    return key;
}

// A reduction of two boundary DOFs and one pattern, its K* scaled by `scale`.
reduction::CachedReduction entryOf(oos::base::MemoryBudget& budget, double scale)
{
    oos::base::BudgetedBuffer<reduction::UnitDofLabel> labels(budget, BOUNDARY_DOFS, "check labels");
    for (std::size_t k = 0; k < BOUNDARY_DOFS; ++k)
        labels[k] = reduction::UnitDofLabel{reduction::UnitPointKind::Grid, 5, reduction::FastenerSide::None,
                                            static_cast<std::uint8_t>(k + 1)};
    oos::linalg::DenseMatrix stiffness(budget, BOUNDARY_DOFS, BOUNDARY_DOFS, "check K*");
    stiffness(0, 0) = 2.0 * scale;
    stiffness(1, 1) = 3.0 * scale;
    stiffness(1, 0) = -scale;
    stiffness(0, 1) = -scale;
    oos::linalg::DenseMatrix patterns(budget, BOUNDARY_DOFS, PATTERN_COUNT, "check patterns");
    patterns(0, 0) = 1.0;
    patterns(1, 0) = -1.0;
    oos::linalg::DenseMatrix inertial(budget, BOUNDARY_DOFS, INERTIAL_PATTERN_COUNT, "check inertial patterns");
    reduction::CachedReduction entry;
    entry.reduced = std::make_shared<const reduction::ReducedUnit>(
        1, std::move(labels), std::move(stiffness), std::move(patterns), 3, std::move(inertial), oos::base::MatN<6>{});
    entry.patternWork = oos::linalg::DenseMatrix(budget, PATTERN_COUNT, PATTERN_COUNT, "check pattern work");
    entry.patternWork(0, 0) = 0.5;
    entry.largestPivotRatio = 1.0;
    entry.interiorDofCount = 4;
    entry.equationStamp = 99;
    return entry;
}

reduction::CacheOptions optionsOf(const std::filesystem::path& folder)
{
    reduction::CacheOptions options;
    options.folder = folder;
    options.solverVersion = SOLVER_VERSION;
    options.lockWait = LOCK_WAIT;
    return options;
}

// The one entry file of the cache folder (nothing when it holds none, or several).
std::optional<std::filesystem::path> onlyEntryFileOf(const std::filesystem::path& folder)
{
    std::optional<std::filesystem::path> found;
    std::size_t count = 0;
    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(folder))
        if (entry.is_regular_file() && entry.path().extension() == reduction::CacheFile::EXTENSION)
        {
            found = entry.path();
            ++count;
        }
    return count == 1 ? found : std::nullopt;
}

std::string textOf(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

CHECK_CASE("condensation cache: a store replaces a file of its name that is not its entry")
{
    const ScratchFolder scratch("cache-store");
    oos::base::MemoryBudget budget(oos::base::MemoryBudget::UNLIMITED);
    const reduction::CacheKey key = keyOf(1);
    const reduction::CachedReduction entry = entryOf(budget, 1.0);
    {
        reduction::CondensationCache cache(optionsOf(scratch.path()));
        reduction::CondensationCache::Session session = cache.openSession("check");
        runner.expect(session.storeReduction(key, entry).isStored, "the first store writes the entry");
    }
    const std::optional<std::filesystem::path> file = onlyEntryFileOf(scratch.path());
    runner.expect(file.has_value(), "the folder holds the one entry file");
    if (!file)
        return;
    // The entry of another key written at this entry's path: what a file of another key of the same hash is to a
    // lookup, a valid file that names another entry.
    reduction::CacheFileHeader other;
    other.kind = reduction::CacheEntryKind::Reduction;
    other.key = keyOf(2);
    reduction::CacheFile::write(*file, other, entryOf(budget, 5.0));
    {
        // Another cache object on the folder: its memory layer is empty, so the lookup reads the file.
        reduction::CondensationCache cache(optionsOf(scratch.path()));
        reduction::CondensationCache::Session session = cache.openSession("check");
        runner.expect(!session.lookupReduction(key, budget).isHit(), "the file of another key is refused");
        runner.expect(std::filesystem::exists(*file), "a refused file that is not broken is left by the lookup");
        runner.expect(session.storeReduction(key, entry).isStored, "the store over the unusable file is made");
    }
    {
        reduction::CondensationCache cache(optionsOf(scratch.path()));
        reduction::CondensationCache::Session session = cache.openSession("check");
        const reduction::CacheLookup<reduction::CachedReduction> found = session.lookupReduction(key, budget);
        runner.expect(found.isHit() && found.outcome == reduction::CacheOutcome::DiskHit,
                      "the entry is read back from the folder: " + found.cause);
        if (found.isHit() && found.entry->reduced)
            runner.expectNear(found.entry->reduced->stiffness()(0, 0), 2.0, 0.0,
                              "K* of the entry stored, not the other's");
        runner.expect(onlyEntryFileOf(scratch.path()).has_value(), "the folder still holds one entry file");
    }
}

CHECK_CASE("condensation cache: a store takes the entry another process stored for its own")
{
    const ScratchFolder scratch("cache-same");
    oos::base::MemoryBudget budget(oos::base::MemoryBudget::UNLIMITED);
    const reduction::CacheKey key = keyOf(1);
    const reduction::CachedReduction entry = entryOf(budget, 1.0);
    {
        reduction::CondensationCache cache(optionsOf(scratch.path()));
        reduction::CondensationCache::Session session = cache.openSession("check");
        runner.expect(session.storeReduction(key, entry).isStored, "the first store writes the entry");
    }
    const std::optional<std::filesystem::path> file = onlyEntryFileOf(scratch.path());
    runner.expect(file.has_value(), "the folder holds the one entry file");
    if (!file)
        return;
    const std::string before = textOf(*file);
    {
        // The same key stored again by a cache that does not hold it in memory (another process): the file stays.
        reduction::CondensationCache cache(optionsOf(scratch.path()));
        reduction::CondensationCache::Session session = cache.openSession("check");
        runner.expect(session.storeReduction(key, entryOf(budget, 5.0)).isStored, "the entry is on disk");
        runner.expect(cache.statistics().bytesWritten == 0, "no file is written for an entry already stored");
    }
    runner.expect(textOf(*file) == before, "the file of the entry is the one stored first");
}

CHECK_CASE("condensation cache: a lock is released by its holder")
{
    const ScratchFolder scratch("cache-lock");
    const std::filesystem::path lockFolder = scratch.path() / "lock";
    std::string firstOwner;
    {
        const reduction::CacheLock lock(scratch.path(), LOCK_WAIT);
        runner.expect(std::filesystem::is_directory(lockFolder), "the lock is a folder of the cache");
        firstOwner = textOf(lockFolder / "owner");
        runner.expect(!firstOwner.empty(), "the lock holds its owner file");
    }
    runner.expect(!std::filesystem::exists(lockFolder), "the holder releases its lock");
    {
        const reduction::CacheLock lock(scratch.path(), LOCK_WAIT);
        runner.expect(textOf(lockFolder / "owner") != firstOwner, "each lock writes a token of its own");
    }
    runner.expect(!std::filesystem::exists(lockFolder), "the second holder releases its lock");
}

CHECK_CASE("condensation cache: a lock broken and taken by a waiter is left by its first holder")
{
    const ScratchFolder scratch("cache-broken-lock");
    const std::filesystem::path lockFolder = scratch.path() / "lock";
    const std::string waiterOwner = "the lock of a waiter\nits token\n";
    {
        const reduction::CacheLock lock(scratch.path(), LOCK_WAIT);
        // What a waiter does to a lock it takes for abandoned (its holder worked for longer than STALE_AGE without a
        // refresh): the folder removed, then made again with the waiter's owner file.
        std::filesystem::remove_all(lockFolder);
        std::filesystem::create_directory(lockFolder);
        std::ofstream(lockFolder / "owner", std::ios::binary) << waiterOwner;
    }
    runner.expect(std::filesystem::is_directory(lockFolder), "the first holder leaves the waiter's lock");
    runner.expect(textOf(lockFolder / "owner") == waiterOwner, "the waiter's owner file is untouched");
}
