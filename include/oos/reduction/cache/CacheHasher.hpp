#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <string_view>

namespace oos::reduction
{

// The 64-bit hash of the condensation cache: the parts of a key (CacheKey), the content checksums of what a unit is
// condensed from, and the checksum of a cache file (CacheFile).
//
// Method: FNV-1a (Fowler, Noll, Vo) on 64-bit words, hash = (hash xor word)·FNV_PRIME, as the checksum of
// linalg::PartialFactor: one multiplication per word, 0.15 s per GB on the reference PC (measured), so that the
// checksum of the matrix of a large unit or of its cache file (a few GB) costs little beside its factorisation or the
// reading of the file. A double enters by its bits (+0.0 and −0.0 differ, as their bits do: a key changes only when an
// input's bits change, which is what makes a cached result the same bits). A text enters by its length then its bytes
// in words of 8, little-endian. Not cryptographic: a key of the cache is never trusted alone (CacheFile also checks the
// content it reads; StaticCondensation checks a recomputed K* against the cached one).
//
// Thread-safety: a value; one object per thread.
class CacheHasher
{
public:
    static constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    static constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

    void mix(std::uint64_t word) noexcept
    {
        m_hash ^= word;
        m_hash *= FNV_PRIME;
    }
    void mix(std::int64_t word) noexcept { mix(std::bit_cast<std::uint64_t>(word)); }
    void mix(std::int32_t word) noexcept { mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(word))); }
    void mix(double value) noexcept { mix(std::bit_cast<std::uint64_t>(value)); }
    void mix(std::string_view text) noexcept
    {
        mix(static_cast<std::uint64_t>(text.size()));
        mixBytes(std::span<const unsigned char>(reinterpret_cast<const unsigned char*>(text.data()), text.size()));
    }
    void mix(std::span<const double> values) noexcept
    {
        mix(static_cast<std::uint64_t>(values.size()));
        for (const double value : values)
            mix(value);
    }
    // Bytes in words of 8 (the last one zero-padded), little-endian: the same hash on every host.
    void mixBytes(std::span<const unsigned char> bytes) noexcept
    {
        std::size_t k = 0;
        for (; k + 8 <= bytes.size(); k += 8)
            mix(wordAt(bytes.data() + k, 8));
        if (k < bytes.size())
            mix(wordAt(bytes.data() + k, bytes.size() - k));
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return m_hash; }

private:
    [[nodiscard]] static std::uint64_t wordAt(const unsigned char* bytes, std::size_t count) noexcept
    {
        std::uint64_t word = 0;
        for (std::size_t b = 0; b < count; ++b)
            word |= static_cast<std::uint64_t>(bytes[b]) << (8 * b);
        return word;
    }

    std::uint64_t m_hash = FNV_OFFSET_BASIS;
};

} // namespace oos::reduction
