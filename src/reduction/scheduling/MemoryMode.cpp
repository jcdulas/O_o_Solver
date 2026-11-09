#include "oos/reduction/scheduling/MemoryMode.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::reduction
{

const char* nameOf(MemoryMode mode) noexcept
{
    switch (mode)
    {
    case MemoryMode::Auto:
        return "auto";
    case MemoryMode::Speed:
        return "speed";
    case MemoryMode::LowMemory:
        return "low_memory";
    }
    return "";
}

MemoryMode memoryModeOf(std::string_view name)
{
    for (const MemoryMode mode : {MemoryMode::Auto, MemoryMode::Speed, MemoryMode::LowMemory})
        if (name == nameOf(mode))
            return mode;
    throw std::invalid_argument(
        base::concatenate({"memory_mode: expected \"auto\", \"speed\" or \"low_memory\", found \"", std::string(name),
                           "\"; give one of the three"}));
}

} // namespace oos::reduction
