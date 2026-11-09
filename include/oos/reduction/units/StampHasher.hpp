#pragma once

#include <cstdint>
#include <cstring>

namespace oos::reduction
{

// The 64-bit stamp of something made of stamped parts (an absorbed cluster is part of the unit's stamp): FNV-1a
// (Fowler, Noll, Vo; 64 bits) over the bytes of the words mixed in, in the order given. The callers
// mix in a fixed order (parts sorted by ID), so the stamp depends on the content only and is the same in every session
// as O_o_Reader's stamps are. Stamps are stored as the signed int64 of O_o_Reader's tables; the bits are what count.
class StampHasher
{
public:
    static constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    static constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

    void mix(std::uint64_t word) noexcept
    {
        for (int shift = 0; shift < 64; shift += 8)
        {
            m_hash ^= (word >> shift) & 0xFFU;
            m_hash *= FNV_PRIME;
        }
    }
    void mix(std::int64_t word) noexcept
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &word, sizeof(bits));
        mix(bits);
    }

    [[nodiscard]] std::int64_t stamp() const noexcept
    {
        std::int64_t value = 0;
        std::memcpy(&value, &m_hash, sizeof(value));
        return value;
    }

private:
    std::uint64_t m_hash = FNV_OFFSET_BASIS;
};

} // namespace oos::reduction
