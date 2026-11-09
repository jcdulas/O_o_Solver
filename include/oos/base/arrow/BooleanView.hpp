#pragma once

#include <cstdint>

namespace oos::base
{

// A "bool" column, read in place: Arrow packs booleans in bits, laid out as a validity bitmap (least-significant bit
// first). One of the typed views of ContractReader (base/arrow/ColumnViews.hpp); the value of a null row is whatever
// the producer left there.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
class BooleanView
{
public:
    BooleanView() = default;
    BooleanView(const std::uint8_t* bits, std::int64_t firstBit, std::int64_t length)
        : m_bits(bits), m_firstBit(firstBit), m_length(length)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    [[nodiscard]] bool operator[](std::int64_t row) const noexcept
    {
        const std::int64_t bit = m_firstBit + row;
        return ((m_bits[bit >> 3] >> (bit & 7)) & 1U) != 0;
    }

private:
    const std::uint8_t* m_bits = nullptr;
    std::int64_t m_firstBit = 0;
    std::int64_t m_length = 0;
};

} // namespace oos::base
