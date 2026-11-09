#pragma once

#include "oos/reduction/scheduling/UnitFootprint.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// A unit whose factorisation does less work than this runs on one thread beside others rather than alone on every
// thread: the threaded kernels share a front only above DenseKernelOptions::DEFAULT_PARALLEL_MINIMUM_MULTIPLY_ADDS
// (2²² multiply-adds, about 8 Mflop, a few milliseconds), and a factorisation whose fronts are all below it gains
// nothing from the pool. 10⁹ flops (about 0.15 s on one thread at ESTIMATE_FLOP_RATE) is well above that: its large
// fronts are shared. The tests lower it (SchedulerOptions) so that the shared path runs on small units.
inline constexpr double LARGE_UNIT_MINIMUM_FLOPS = 1.0e9;

struct SchedulerOptions
{
    std::size_t threadCount = 1;
    // The units of a wave of small units: at most this many per thread (the pool deals them to its threads as they
    // finish; more units per wave share its serial steps, the keys and the cache stores).
    std::size_t unitsPerThread = 1;
    double largeUnitMinimumFlops = LARGE_UNIT_MINIMUM_FLOPS;
    // What the analysis needs after the condensation beyond what the units keep (CondensedMemoryPlan::
    // laterPhaseBytes): part of every reserve.
    std::uint64_t laterPhaseBytes = 0;
};

// One wave of the condensation: units condensed at the same time.
struct UnitWave
{
    std::vector<std::size_t> units; // indices into the footprints given to the scheduler, in its order
    // True: one unit with every thread of the pool (a large unit, or the only one the memory admits); false: one
    // thread per unit, at most unitsPerThread × threadCount units.
    bool isShared = false;
};

// The order and the waves of the condensation of the units (units are condensed in parallel as long as their
// estimated peaks fit):
// - order: by decreasing estimated work (UnitFootprint::flopCount), ties by their index: the longest first, so that a
//   wave of units on one thread each ends with the short ones (longest processing time first, Graham 1969);
// - waves, in that order: a large unit (its work at least largeUnitMinimumFlops and at least the remaining work divided
//   by the threads: alone it would hold up a wave of small ones) is a wave of its own with every thread, its fronts
//   shared by the threaded kernels; otherwise the next unit and, first fit in the order, the following units that are
//   not large, while the wave has fewer units than unitsPerThread × threads, the kept data and condensations of its
//   units fit the bytes
//   given (UnitFootprint::keptBytes + condensationBytes()), and what they keep leaves the reserve of the units after
//   them (reserve()). A wave of one unit has every thread. The next unit always makes a wave, even when it alone
//   exceeds the bytes given: the caller's memory plan (CondensedMemoryPlan::requireFits) has checked that the analysis
//   fits one unit at a time in the order, and the units pulled forward into a wave left the reserve of the rest.
// The schedule changes no result: a condensation does not depend on the others nor on its thread count
// (PartialFactor gives the same bits for any number of threads), and each unit writes only its own results.
//
// Thread-safety: not for concurrent use. Ownership: copies the footprints.
class UnitScheduler
{
public:
    // Throws std::invalid_argument for a thread count of 0.
    UnitScheduler(std::span<const UnitFootprint> units, SchedulerOptions options);

    // Every unit, by decreasing work (the order the waves follow, but for the units a wave pulls forward).
    [[nodiscard]] std::span<const std::size_t> order() const noexcept { return m_order; }
    [[nodiscard]] bool isDone() const noexcept { return m_remaining.empty(); }

    // The next wave, within `availableBytes` (the budget's free bytes now). Throws std::logic_error when every unit is
    // scheduled.
    [[nodiscard]] UnitWave nextWave(std::uint64_t availableBytes);

    // What the units not yet scheduled and the later phases need beyond what is held now: Σ kept + max(the largest
    // condensation, the later phase) over the units left, an upper bound of their sequential peak.
    [[nodiscard]] std::uint64_t reserve() const;

private:
    [[nodiscard]] bool isLarge(std::size_t unit) const noexcept;
    // The largest condensation of the units left, but those of `excluded` (a wave being made).
    [[nodiscard]] std::uint64_t largestCondensationExcept(std::span<const std::size_t> excluded) const;

    std::vector<UnitFootprint> m_units;
    SchedulerOptions m_options;
    std::vector<std::size_t> m_order;     // by decreasing work, ties by index
    std::vector<std::size_t> m_remaining; // the units not yet scheduled, in m_order's order
    std::vector<std::uint8_t> m_isScheduled;
    // The units by decreasing condensation (ties by index); the entries before m_condensationStart are scheduled.
    std::vector<std::size_t> m_byCondensation;
    std::size_t m_condensationStart = 0;
    std::uint64_t m_keptRemaining = 0;
    double m_flopsRemaining = 0.0;
};

} // namespace oos::reduction
