#include "oos/base/platform/MemoryInfo.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#include <malloc.h>
#else
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#endif

namespace oos::base
{

#if defined(_WIN32)

MemoryInfo MemoryInfo::query() noexcept
{
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) == 0)
        return {};
    return {status.ullTotalPhys, status.ullAvailPhys};
}

#elif defined(__EMSCRIPTEN__)

// WebAssembly: the "machine" is the module's linear memory. Its total is the largest size the heap may grow to (set at
// link time, wasm/CMakeLists.txt), and what is available is that size less the bytes malloc has handed out.
MemoryInfo MemoryInfo::query() noexcept
{
    const std::uint64_t total = emscripten_get_heap_max();
    const struct mallinfo usage = mallinfo();
    const std::uint64_t used = static_cast<std::uint64_t>(usage.uordblks);
    return {total, used < total ? total - used : 0};
}

#else

namespace
{

// MemAvailable of /proc/meminfo (Linux 3.14 and later), in kB in the file; 0 when absent.
std::uint64_t readAvailableBytes() noexcept
{
    try
    {
        constexpr std::uint64_t BYTES_PER_KB = 1024;
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line))
        {
            if (line.rfind("MemAvailable:", 0) != 0)
                continue;
            std::istringstream fields(line.substr(sizeof("MemAvailable:") - 1));
            std::uint64_t kilobytes = 0;
            fields >> kilobytes;
            return kilobytes * BYTES_PER_KB;
        }
    }
    catch (...)
    {
    }
    return 0;
}

} // namespace

MemoryInfo MemoryInfo::query() noexcept
{
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGESIZE);
    const std::uint64_t total =
        pages > 0 && pageSize > 0 ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(pageSize) : 0;
    return {total, readAvailableBytes()};
}

#endif

} // namespace oos::base
