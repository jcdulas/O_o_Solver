#pragma once

#include <cstdint>

namespace oos::base
{

// The physical memory of the machine, as the defaults of memory= and the memory-aware scheduler read it.
// A plain value; queried anew each time (the available memory changes).
struct MemoryInfo
{
    std::uint64_t totalBytes = 0;     // the installed physical memory
    std::uint64_t availableBytes = 0; // what can be allocated now without swapping (Windows: "available"; Linux:
                                      // MemAvailable of /proc/meminfo)

    // Asks the operating system. Never throws: a value the system does not give is 0, and the caller then falls back
    // on its own default.
    [[nodiscard]] static MemoryInfo query() noexcept;
};

} // namespace oos::base
