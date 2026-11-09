#pragma once

#include "oos/base/memory/AlignedBuffer.hpp"

#include <cstdint>
#include <map>
#include <mutex>

namespace oos::linalg
{

// The blocks of the working storage of one factorisation (fronts and update matrices), kept for reuse when released
// (internal to linalg/supernodal). A large block freed goes back to the system, and a new one of that size comes back
// as fresh pages that the first write faults in one by one (0.19 ms per MB on the reference PC against 0.07 ms for
// writing memory already mapped): the fronts of a factorisation are thousands of such blocks.
//
// PERFORMANCE: a block of at least KEPT_MINIMUM_VALUES released is kept, while the blocks in use and kept stay within
// the capacity (the bound of the working storage reserved in the budget); a request takes a kept block of
// exactly its size class (classSize), else a new one. Smaller blocks come and go through the heap, which keeps their
// pages. Every block is uninitialised: its user writes before it reads (WorkBuffer).
// Measured: one thread, the shell of 100 000 DOFs 0.50 -> 0.43 s, the 2D grid
// of 100 000 DOFs 0.41 -> 0.33 s; six threads, the shell of 1 M DOFs 3.69 -> 3.45 s.
//
// Memory: the blocks in use plus the blocks kept never exceed the capacity, except while the blocks in use alone do (a
// growth beyond the plan, reserved in the budget by its WorkBuffer): then nothing is kept. The planned sizes of the
// working storage are class sizes (FactorSchedule), so the blocks in use are what the bound counts.
//
// Thread-safety: every member is thread-safe (a mutex). Ownership: owns the kept blocks; a block handed out is its
// user's until released.
class WorkspacePool
{
public:
    // 256 kB: the Windows heap gives blocks above about 512 kB fresh pages from the system each time.
    static constexpr std::int64_t KEPT_MINIMUM_VALUES = std::int64_t{1} << 15;
    // Size classes of the large blocks: CLASSES_PER_OCTAVE per doubling, so that a class is at most 1/16 above the size
    // asked (a block of a class serves every request of that class).
    static constexpr int CLASSES_PER_OCTAVE = 16;

    explicit WorkspacePool(std::int64_t capacityValues) noexcept : m_capacity(capacityValues) {}
    WorkspacePool(const WorkspacePool&) = delete;
    WorkspacePool& operator=(const WorkspacePool&) = delete;

    // The size of the block a request of `values` values gets: `values` below KEPT_MINIMUM_VALUES, else the smallest
    // class size at or above it.
    [[nodiscard]] static std::int64_t classSize(std::int64_t values) noexcept;

    // A block of classSize(values) values (uninitialised): a kept one of that size, else a new one.
    [[nodiscard]] base::AlignedBuffer<double> acquire(std::int64_t values);
    // Takes a block back: kept when large and within the capacity, else freed.
    void release(base::AlignedBuffer<double> block) noexcept;

    // The peak of the values held (in use and kept).
    [[nodiscard]] std::int64_t heldPeak() const;

private:
    mutable std::mutex m_mutex;
    std::multimap<std::int64_t, base::AlignedBuffer<double>> m_kept; // by size: a few dozen large blocks
    std::int64_t m_capacity = 0;
    std::int64_t m_inUseValues = 0;
    std::int64_t m_keptValues = 0;
    std::int64_t m_heldPeak = 0;
};

} // namespace oos::linalg
