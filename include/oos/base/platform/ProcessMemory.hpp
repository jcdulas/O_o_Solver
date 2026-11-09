#pragma once

#include <cstdint>

namespace oos::base
{

// The memory counters of this process as the operating system keeps them: the page faults since it started and its
// resident memory, now and at the peak (to measure the factorisation's memory: a fresh buffer's
// pages are faulted in one by one at their first write; a reused one's are not). A plain value; queried anew each
// time.
struct ProcessMemory
{
    std::uint64_t pageFaults = 0;        // Windows: PageFaultCount (soft and hard); Linux: ru_minflt + ru_majflt
    std::uint64_t residentBytes = 0;     // Windows: the working set; Linux: the resident set of /proc/self/statm
    std::uint64_t peakResidentBytes = 0; // Windows: the peak working set; Linux: ru_maxrss

    // Asks the operating system. Never throws: a value the system does not give is 0.
    [[nodiscard]] static ProcessMemory query() noexcept;
};

} // namespace oos::base
