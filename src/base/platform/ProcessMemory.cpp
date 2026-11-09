#include "oos/base/platform/ProcessMemory.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// windows.h first: psapi.h uses its types. GetProcessMemoryInfo is kernel32's K32GetProcessMemoryInfo (PSAPI_VERSION
// 2, Windows 7 and later): no psapi library to link.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#include <malloc.h>
#else
#include <sys/resource.h>

#include <fstream>
#include <unistd.h>
#endif

namespace oos::base
{

#if defined(_WIN32)

ProcessMemory ProcessMemory::query() noexcept
{
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0)
        return {};
    return {counters.PageFaultCount, counters.WorkingSetSize, counters.PeakWorkingSetSize};
}

#elif defined(__EMSCRIPTEN__)

// WebAssembly has no pages nor page faults: the resident bytes are the bytes malloc has handed out, and the peak is the
// size the linear memory has grown to (it never shrinks).
ProcessMemory ProcessMemory::query() noexcept
{
    const struct mallinfo usage = mallinfo();
    return {0, static_cast<std::uint64_t>(usage.uordblks), static_cast<std::uint64_t>(emscripten_get_heap_size())};
}

#else

namespace
{

// The resident pages of /proc/self/statm (its second field) in bytes; 0 when absent.
std::uint64_t readResidentBytes() noexcept
{
    try
    {
        std::ifstream statm("/proc/self/statm");
        std::uint64_t sizePages = 0;
        std::uint64_t residentPages = 0;
        if (!(statm >> sizePages >> residentPages))
            return 0;
        const long pageSize = sysconf(_SC_PAGESIZE);
        return pageSize > 0 ? residentPages * static_cast<std::uint64_t>(pageSize) : 0;
    }
    catch (...)
    {
        return 0;
    }
}

} // namespace

ProcessMemory ProcessMemory::query() noexcept
{
    constexpr std::uint64_t BYTES_PER_KB = 1024; // ru_maxrss is in kilobytes on Linux
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return {0, readResidentBytes(), 0};
    return {static_cast<std::uint64_t>(usage.ru_minflt) + static_cast<std::uint64_t>(usage.ru_majflt),
            readResidentBytes(), static_cast<std::uint64_t>(usage.ru_maxrss) * BYTES_PER_KB};
}

#endif

} // namespace oos::base
