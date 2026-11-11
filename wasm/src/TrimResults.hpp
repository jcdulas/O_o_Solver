#pragma once

#include "oos/solutions/sol144/TrimSolution.hpp"

#include <emscripten/val.h>

namespace oos::wasm
{
// Owned JavaScript arrays; no view into solver memory survives the conversion.
[[nodiscard]] emscripten::val trimResultsObject(const solutions::TrimResults& results);
} // namespace oos::wasm
