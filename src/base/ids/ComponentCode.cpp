#include "oos/base/ids/ComponentCode.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::base
{

ComponentCode ComponentCode::parse(std::string_view digits)
{
    if (digits == "0")
        return scalar();
    std::uint8_t mask = 0;
    for (const char digit : digits)
    {
        const std::string_view found(&digit, 1);
        if (digit < '1' || digit > '6')
            throw std::invalid_argument(concatenate({"component code '", digits,
                                                     "': expected the digits 1 to 6 (or 0 alone for a scalar point), "
                                                     "found '",
                                                     found, "'"}));
        const auto bit = static_cast<std::uint8_t>(1U << (digit - '1'));
        if ((mask & bit) != 0)
            throw std::invalid_argument(concatenate({"component code '", digits, "': each digit may appear once, '",
                                                     found, "' appears twice; write each component once"}));
        mask = static_cast<std::uint8_t>(mask | bit);
    }
    return ComponentCode(mask);
}

std::string ComponentCode::text() const
{
    if (m_isScalar)
        return "0";
    std::string result;
    for (int component = 1; component <= 6; ++component)
    {
        if (contains(component))
            result += static_cast<char>('0' + component);
    }
    return result;
}

} // namespace oos::base
