#pragma once

#include <bit>
#include <cstdint>
#include <string>
#include <string_view>

namespace oos::base
{

// A set of the six components of a grid point, as the Nastran bulk data format writes it in a field (any unique
// combination of the digits 1 to 6 with no embedded blanks; 0 for a scalar point): "123456", "135", "0". Held as a bit
// mask, bit k-1 for component k. Immutable value type.
class ComponentCode
{
public:
    constexpr ComponentCode() = default;

    // Reads the digits of a field, in any order. Throws std::invalid_argument for a character outside 0..6, a repeated
    // digit, or a 0 mixed with other digits, naming the text. "0" gives the empty set with isScalar() true; "" gives
    // the empty set.
    [[nodiscard]] static ComponentCode parse(std::string_view digits);

    [[nodiscard]] static constexpr ComponentCode all() { return ComponentCode(ALL_MASK); }
    [[nodiscard]] static constexpr ComponentCode fromMask(std::uint8_t mask)
    {
        return ComponentCode(static_cast<std::uint8_t>(mask & ALL_MASK));
    }
    [[nodiscard]] static constexpr ComponentCode scalar() { return ComponentCode(0, true); }

    // Component k in 1..6; false for any other k.
    [[nodiscard]] constexpr bool contains(int component) const
    {
        return component >= 1 && component <= 6 && ((m_mask >> (component - 1)) & 1U) != 0;
    }
    [[nodiscard]] constexpr std::uint8_t mask() const { return m_mask; }
    [[nodiscard]] constexpr bool isScalar() const { return m_isScalar; }
    [[nodiscard]] constexpr bool isEmpty() const { return m_mask == 0; }
    [[nodiscard]] constexpr int count() const { return std::popcount(static_cast<unsigned>(m_mask)); }

    // The digits in increasing order, "123456"; "0" for a scalar point; "" for the empty set.
    [[nodiscard]] std::string text() const;

    friend constexpr bool operator==(ComponentCode, ComponentCode) = default;

private:
    static constexpr std::uint8_t ALL_MASK = 0x3F;

    constexpr explicit ComponentCode(std::uint8_t mask, bool isScalar = false) : m_mask(mask), m_isScalar(isScalar) {}

    std::uint8_t m_mask = 0;
    bool m_isScalar = false;
};

} // namespace oos::base
