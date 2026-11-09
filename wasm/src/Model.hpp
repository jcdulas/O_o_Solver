#pragma once

#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <memory>

namespace oos::wasm
{
struct Model
{
    std::shared_ptr<const input::ModelSnapshot> snapshot;
};
} // namespace oos::wasm
