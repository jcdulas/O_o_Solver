#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

namespace oos::base
{

// The progress of one stage of a long operation ("factorisation", "condensation"). Worker threads advance it; the
// calling thread polls it and reports at most ten times a second: the caller's callbacks are only ever
// called from the calling thread. Thread-safe for advance(); poll from one thread only.
class ProgressCounter
{
public:
    // The shortest time between two reports.
    static constexpr std::chrono::milliseconds MIN_REPORT_INTERVAL{100};

    ProgressCounter(std::string stage, std::uint64_t total) : m_stage(std::move(stage)), m_total(total) {}

    void advance(std::uint64_t amount = 1) noexcept { m_done.fetch_add(amount, std::memory_order_relaxed); }

    [[nodiscard]] const std::string& stage() const noexcept { return m_stage; }
    [[nodiscard]] std::uint64_t total() const noexcept { return m_total; }
    [[nodiscard]] std::uint64_t done() const noexcept { return m_done.load(std::memory_order_relaxed); }

    // True when a report is due: the first call, the last unit of work, or MIN_REPORT_INTERVAL after the last report.
    [[nodiscard]] bool isReportDue(std::chrono::steady_clock::time_point now);

private:
    std::string m_stage;
    std::uint64_t m_total;
    std::atomic<std::uint64_t> m_done{0};
    std::chrono::steady_clock::time_point m_lastReport{};
    bool m_hasReported = false;
    bool m_hasReportedEnd = false;
};

} // namespace oos::base
