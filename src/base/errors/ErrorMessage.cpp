#include "oos/base/errors/ErrorMessage.hpp"

#include <cstddef>

namespace oos::base
{

std::string concatenate(std::initializer_list<std::string_view> pieces)
{
    std::size_t length = 0;
    for (const std::string_view piece : pieces)
        length += piece.size();
    std::string text;
    text.reserve(length);
    for (const std::string_view piece : pieces)
        text.append(piece);
    return text;
}

} // namespace oos::base
