#pragma once

#include <bit>
#include <cstdint>

namespace oos::base
{

// Which rows of a column have a value (one of the typed views of ContractReader, base/arrow/ColumnViews.hpp). Without a
// validity bitmap every row has one; a column of the Arrow null type has none. Points into the producer's bitmap.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
class ValidityView
{
public:
    ValidityView() = default; // every row valid, size 0
    ValidityView(const std::uint8_t* bits, std::int64_t firstBit, std::int64_t length, bool isAllNull = false)
        : m_bits(bits), m_firstBit(firstBit), m_length(length), m_isAllNull(isAllNull)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    [[nodiscard]] bool isValid(std::int64_t row) const noexcept
    {
        if (m_isAllNull)
            return false;
        if (m_bits == nullptr)
            return true;
        // Arrow validity bitmaps are least-significant-bit first: bit k of byte k / 8 is row k (1 = valid).
        const std::int64_t bit = m_firstBit + row;
        return ((m_bits[bit >> 3] >> (bit & 7)) & 1U) != 0;
    }
    // The rows without a value: the partial bytes at both ends bit by bit, the whole bytes between by popcount.
    [[nodiscard]] std::int64_t nullCount() const noexcept
    {
        if (m_isAllNull)
            return m_length;
        if (m_bits == nullptr || m_length == 0)
            return 0;
        std::int64_t valid = 0;
        std::int64_t bit = m_firstBit;
        const std::int64_t end = m_firstBit + m_length;
        for (; bit < end && (bit & 7) != 0; ++bit)
            valid += (m_bits[bit >> 3] >> (bit & 7)) & 1U;
        for (; bit + 8 <= end; bit += 8)
            valid += std::popcount(static_cast<unsigned>(m_bits[bit >> 3]));
        for (; bit < end; ++bit)
            valid += (m_bits[bit >> 3] >> (bit & 7)) & 1U;
        return m_length - valid;
    }

private:
    const std::uint8_t* m_bits = nullptr;
    std::int64_t m_firstBit = 0;
    std::int64_t m_length = 0;
    bool m_isAllNull = false;
};

} // namespace oos::base
