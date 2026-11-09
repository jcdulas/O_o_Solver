#pragma once

#include <chrono>

namespace oos::base
{

// The wall time of the phases of an operation, for its report: lap() gives the seconds since the last lap (or the
// construction). Not thread-safe; one per timed sequence.
class Stopwatch
{
public:
    [[nodiscard]] double lap() noexcept
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - m_last).count();
        m_last = now;
        return seconds;
    }

private:
    std::chrono::steady_clock::time_point m_last = std::chrono::steady_clock::now();
};

} // namespace oos::base
