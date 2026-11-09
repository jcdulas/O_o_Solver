#pragma once

#include <initializer_list>
#include <string>
#include <string_view>

namespace oos::base
{

// Joins the pieces of an error message: concatenate({"table '", name, "': ", std::to_string(count), " rows"}).
// It replaces chains of operator+, on which GCC 12 reports a false -Wrestrict when a literal is added to a temporary
// string (GCC bug 105329), and it allocates once.
[[nodiscard]] std::string concatenate(std::initializer_list<std::string_view> pieces);

} // namespace oos::base
