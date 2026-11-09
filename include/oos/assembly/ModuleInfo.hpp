#pragma once

#include <string_view>

namespace oos::assembly
{

// The name of the module, as the layering check and the support reports name it.
[[nodiscard]] std::string_view moduleName() noexcept;

} // namespace oos::assembly
