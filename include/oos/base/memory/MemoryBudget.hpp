#pragma once

#include "oos/base/errors/MemoryBudgetError.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace oos::base
{

class MemoryBudget;

// A part of the budget taken by one buffer or one phase; given back when the reservation dies. Movable, not copyable;
// an empty (default) reservation holds nothing. Must not outlive its budget.
class MemoryReservation
{
public:
    MemoryReservation() = default;
    MemoryReservation(const MemoryReservation&) = delete;
    MemoryReservation& operator=(const MemoryReservation&) = delete;
    MemoryReservation(MemoryReservation&& other) noexcept;
    MemoryReservation& operator=(MemoryReservation&& other) noexcept;
    ~MemoryReservation();

    [[nodiscard]] std::uint64_t bytes() const noexcept { return m_bytes; }
    [[nodiscard]] const std::string& label() const noexcept { return m_label; }

    // Gives the bytes back now (the reservation becomes empty).
    void release() noexcept;

private:
    friend class MemoryBudget;
    MemoryReservation(MemoryBudget* budget, std::uint64_t bytes, std::string label)
        : m_budget(budget), m_bytes(bytes), m_label(std::move(label))
    {
    }

    MemoryBudget* m_budget = nullptr;
    std::uint64_t m_bytes = 0;
    std::string m_label;
};

// The memory budget of an analysis: every allocation whose size grows with the model is estimated
// and reserved **before** it is made; a reservation that does not fit is refused with MemoryBudgetError, naming the
// estimate, what is left and the largest consumers, so that nothing large is ever allocated beyond the budget. The
// budget counts what is reserved, not what the allocator really uses: an estimate must cover the allocations that
// follow it.
//
// Thread-safety: every member is thread-safe (one mutex; a reservation is a few operations under it, never on a hot
// path). Ownership: the budget must outlive its reservations.
class MemoryBudget
{
public:
    // The limit of a budget that bounds nothing, for tests and for callers that do not bound the memory.
    static constexpr std::uint64_t UNLIMITED = std::numeric_limits<std::uint64_t>::max();

    explicit MemoryBudget(std::uint64_t limitBytes) : m_limit(limitBytes) {}
    MemoryBudget(const MemoryBudget&) = delete;
    MemoryBudget& operator=(const MemoryBudget&) = delete;
    ~MemoryBudget() = default;

    // Reserves `bytes` for `label` ("factor of unit 12"); throws MemoryBudgetError when it does not fit. A label may
    // hold several reservations; the refusal adds them up.
    [[nodiscard]] MemoryReservation reserve(std::uint64_t bytes, const std::string& label);

    // True when `bytes` would fit now (the scheduler asks before starting a unit).
    [[nodiscard]] bool fits(std::uint64_t bytes) const;

    [[nodiscard]] std::uint64_t limit() const noexcept { return m_limit; }
    [[nodiscard]] std::uint64_t used() const;
    [[nodiscard]] std::uint64_t available() const;
    // The largest used() since the budget was made.
    [[nodiscard]] std::uint64_t peak() const;
    // The labels holding the most memory now, largest first (ties by label), at most `count` of them.
    [[nodiscard]] std::vector<MemoryConsumer> largestConsumers(std::size_t count) const;

private:
    friend class MemoryReservation;
    void giveBack(std::uint64_t bytes, const std::string& label) noexcept;
    [[nodiscard]] std::vector<MemoryConsumer> largestConsumersLocked(std::size_t count) const;

    const std::uint64_t m_limit;
    mutable std::mutex m_mutex;
    std::uint64_t m_used = 0;
    std::uint64_t m_peak = 0;
    std::map<std::string, std::uint64_t> m_byLabel;
};

// "1.5 GB", "320 MB", "12 kB", "512 B": sizes as reports and messages print them (powers of 1000, three significant
// digits).
[[nodiscard]] std::string formatBytes(std::uint64_t bytes);

} // namespace oos::base
