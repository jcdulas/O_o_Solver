#pragma once

#include "WorkspacePool.hpp"

#include "oos/base/memory/AlignedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <atomic>
#include <cstdint>
#include <string>

namespace oos::linalg
{

// The live working storage of one factorisation, measured: the values of the WorkBuffers that the up-front
// reservation of the working storage covers (their planned part), now and at the peak. The peak is what the bound of
// FactorSchedule must cover; the tests compare them (an independent measure, not the budget's own accounting, which
// only sees the up-front reservation).
//
// Thread-safety: every member is thread-safe (atomics; the peak is a compare-and-swap maximum).
class WorkspaceMeter
{
public:
    void add(std::int64_t values) noexcept;
    void remove(std::int64_t values) noexcept { m_live.fetch_sub(values, std::memory_order_relaxed); }

    [[nodiscard]] std::int64_t live() const noexcept { return m_live.load(std::memory_order_relaxed); }
    [[nodiscard]] std::int64_t peak() const noexcept { return m_peak.load(std::memory_order_relaxed); }

private:
    std::atomic<std::int64_t> m_live{0};
    std::atomic<std::int64_t> m_peak{0};
};

// A buffer of the working storage of the multifrontal factorisation (fronts, stacks and update matrices). Its first
// plannedValues values are covered by the up-front reservation of the whole working storage (FactorSchedule) and
// counted live in the meter; beyond them (delayed pivots, or a subtree run beyond the plan) the growth is reserved in
// the budget while the buffer holds it.
//
// The values are not initialised: every user writes before it reads (assembleFront clears a front; an update matrix
// is copied in whole). With a pool, the values come from it and go back to it (its class sizes: the planned values are
// class sizes too, FactorSchedule).
//
// Thread-safety: one owner at a time. Ownership: owns its values and its growth reservation; movable, not copyable.
// The meter, the pool and the budget outlive it.
class WorkBuffer
{
public:
    WorkBuffer() = default;
    WorkBuffer(std::int64_t plannedValues, std::string label, WorkspaceMeter* meter, WorkspacePool* pool = nullptr)
        : m_plannedValues(plannedValues), m_label(std::move(label)), m_meter(meter), m_pool(pool)
    {
    }
    WorkBuffer(const WorkBuffer&) = delete;
    WorkBuffer& operator=(const WorkBuffer&) = delete;
    WorkBuffer(WorkBuffer&& other) noexcept;
    WorkBuffer& operator=(WorkBuffer&& other) noexcept;
    ~WorkBuffer() { release(); }

    // Makes room for `values` values; keeps the first keptValues of the old contents.
    void ensure(std::int64_t values, base::MemoryBudget& budget, std::int64_t keptValues = 0);
    void release() noexcept;

    [[nodiscard]] double* data() noexcept { return m_values.data(); }
    [[nodiscard]] const double* data() const noexcept { return m_values.data(); }
    [[nodiscard]] std::int64_t capacity() const noexcept { return static_cast<std::int64_t>(m_values.size()); }

private:
    // The part of a capacity covered by the plan (counted in the meter).
    [[nodiscard]] std::int64_t plannedPartOf(std::int64_t capacity) const noexcept
    {
        return capacity < m_plannedValues ? capacity : m_plannedValues;
    }

    std::int64_t m_plannedValues = 0;
    std::string m_label;
    WorkspaceMeter* m_meter = nullptr;
    WorkspacePool* m_pool = nullptr;
    base::MemoryReservation m_growth; // declared before the values: reserved before, given back after
    base::AlignedBuffer<double> m_values;
};

} // namespace oos::linalg
