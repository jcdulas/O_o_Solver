#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/reduction/cache/CacheEntry.hpp"
#include "oos/reduction/cache/CacheKey.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace oos::reduction
{

// The header of a cache file: what an entry is, read without its matrices.
struct CacheFileHeader
{
    CacheEntryKind kind = CacheEntryKind::Reduction;
    std::int64_t createdAt = 0; // seconds since the epoch
    std::string modelTag;       // the model the entry was condensed for, as the caller named it ("" when none)
    CacheKey key;
};

// The outcome of reading a cache file: the value, or why the file is not used (the refusal names the cause: "not a
// O_o_Solver cache file", "format version 2, this version reads 1", "truncated", "corrupted (its checksum differs)",
// "another entry", an I/O failure).
template <typename T>
struct CacheFileRead
{
    std::optional<T> value;
    std::string refusal;
    std::uint64_t bytesRead = 0;

    [[nodiscard]] bool isRead() const noexcept { return value.has_value(); }
};

// One entry of the condensation cache as a file: a versioned header (a magic word, FORMAT_VERSION, a byte-order mark,
// the kind, the creation time, the model tag), the key in full (every part of CacheKey, so that a reader names why an
// entry of the same unit does not serve), the sections of the entry, and a checksum of everything before it.
// - Little-endian 64-bit words throughout (texts as their length then their bytes padded to a word), so that the file
//   is the same on every host; on a little-endian host the matrices are written and read as they lie in memory.
// - A reduction: a word of the parts it has and their counts, then the unit part (the boundary labels, K* as its lower
//   triangle by columns (K* is symmetric to the bit: its upper triangle is the mirror of the lower one,
//   linalg::PartialFactor), f*_k and the inertial patterns by columns, the rigid-body mass properties, the pattern
//   work G as its lower triangle), the reduced boundary mass M_aa of a Craig-Bampton unit as its lower triangle, and
//   the modal part of a Craig-Bampton unit (the eigenvalues, K_qq and M_qq as lower triangles, M_aq by columns);
//   recovery data: the modes by columns and their eigenvalues. Format 2 added the mode cut-off to the key
//   and the parts word; format 3 the key's input stamp beside the equations' checksum, which is no longer
//   matched (CacheKey); format 4 the key's element formulation version after the kernel; a file of
//   an earlier format is refused by name and condensed again.
// - Written through a temporary file renamed onto the entry (base::TemporaryFile): a reader sees the old file, the new
//   one or none, never a part; a reader checks the size and the checksum (CacheHasher over every word) and refuses a
//   file that does not match, naming why: a stale or corrupted entry is never used silently.
// No compression: it would be kept only if it saved time, and measurement showed it does not.
//
// Thread-safety: static functions; distinct files may be written and read at once. Ownership: none.
class CacheFile
{
public:
    static constexpr std::uint32_t FORMAT_VERSION = 4;
    static constexpr std::string_view EXTENSION = ".nsvc";

    // Writes an entry to `path` through a temporary file. `beforeSection` (may be empty) is called before each section
    // (the cache refreshes its lock there). Returns the bytes written. Throws base::FileError when the file cannot be
    // written or renamed (the target is then unchanged).
    static std::uint64_t write(const std::filesystem::path& path, const CacheFileHeader& header,
                               const CachedReduction& entry, const std::function<void()>& beforeSection = {});
    static std::uint64_t write(const std::filesystem::path& path, const CacheFileHeader& header,
                               const CachedRecoveryData& entry, const std::function<void()>& beforeSection = {});

    // The size of the file of an entry (what it weighs against the cache limits), before it is written.
    [[nodiscard]] static std::uint64_t fileBytes(const CacheFileHeader& header, const CachedReduction& entry);
    [[nodiscard]] static std::uint64_t fileBytes(const CacheFileHeader& header, const CachedRecoveryData& entry);

    // The header of a file (its kind, key, model tag, creation time). The checksum is not checked (it covers the whole
    // file): a header serves to name a cause or to list the cache, never to use an entry.
    [[nodiscard]] static CacheFileRead<CacheFileHeader> readHeader(const std::filesystem::path& path);

    // The entry of a file whose key is `expected` (another key: refused as "another entry"), its matrices reserved in
    // `budget` (MemoryBudgetError when they do not fit), its checksum checked.
    [[nodiscard]] static CacheFileRead<std::shared_ptr<const CachedReduction>> readReduction(
        const std::filesystem::path& path, const CacheKey& expected, base::MemoryBudget& budget);
    [[nodiscard]] static CacheFileRead<std::shared_ptr<const CachedRecoveryData>> readRecoveryData(
        const std::filesystem::path& path, const CacheKey& expected, base::MemoryBudget& budget);
};

} // namespace oos::reduction
