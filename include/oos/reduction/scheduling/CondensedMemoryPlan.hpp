#pragma once

#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/reduction/scheduling/InterfaceFootprint.hpp"
#include "oos/reduction/scheduling/MemoryMode.hpp"
#include "oos/reduction/scheduling/UnitFootprint.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::reduction
{

// How the units of a condensed analysis are recovered.
enum class RecoveryStrategy : std::uint8_t
{
    AllAtOnce, // every recovered unit's recovery made before the cases and held through the constraint set
    Sequential // the units recovered in sequence, each unit's recovery released once its cases are handed over
};

// "all at once", "in sequence".
[[nodiscard]] const char* nameOf(RecoveryStrategy strategy) noexcept;

// What a condensed analysis keeps: its recovery and the units whose equations are released after their condensation
// and built again at their recovery.
struct MemoryStrategy
{
    RecoveryStrategy recovery = RecoveryStrategy::AllAtOnce;
    std::vector<std::uint8_t> isReleased; // per unit, in the order of CondensedMemoryPlan::Inputs::units (empty: none)
    std::size_t releasedCount = 0;
    // Step 10.23: every unit condensed without its factor (in sequence only), and the region bound of the
    // recoveries without it: 0, a recovered unit's interior factorised again whole; otherwise the units whose factor
    // exceeds the bound recovered by regions of their elimination tree, none above it.
    bool isWithoutFactor = false;
    std::uint64_t regionBytes = 0;
    // Step 10.25: the interface problem solved without its factor (InterfaceInputs::keepsFactor false), its
    // backward sweep by regions of at most interfaceRegionBytes (its factor's bytes or more: every panel kept from the
    // forward run, the factor made in the solve and released after it).
    bool isInterfaceWithoutFactor = false;
    std::uint64_t interfaceRegionBytes = 0;
};

// The memory plan of a condensed analysis (an analysis that cannot fit stops before it starts, with the estimate and
// the largest consumers), made before any unit is
// condensed from the footprints of its phases, for a strategy (MemoryStrategy):
// 1. condensation: the units one after the other in the scheduler's order (the smallest peak any schedule has: one
//    unit at a time, its factor released after it), each holding what the units before it keep (their equations, or
//    what is left of a released unit), its own equations and its condensation (build, then factorisation), and for a
//    released unit the system without equations made beside them;
// 2. interface and recovery, per constraint set (the largest set counts): what every unit keeps, the set's interface
//    phase (InterfaceFootprint), the recoveries (all at once: the recoveries of every recovered unit and the
//    refactorisation of one released factor; in sequence: the largest pass of one unit, its equations built again when
//    released), one load case of a batch, and the result tables kept to the end; the g-set vectors of the patterns'
//    loads, in the baseline, are released before it.
// minimumPeakBytes() is the larger of the two, beyond what the budget holds before the analysis (the baseline): a
// budget below baseline + minimumPeakBytes() cannot fit and is refused by requireFits() before any work.
// laterPhaseBytes() is what the scheduler's reserves add beyond the units (UnitScheduler::reserve()), within which it
// admits a wave and FactorRetention keeps a factor.
//
// The strategy comes from the memory mode (MemoryMode) and the budget: Speed every recovery at once and every unit's
// equations kept; LowMemory in sequence with every unit condensed without its factor and
// its equations released, recovered by regions when even its factor made again does not fit (the largest region bound
// that fits), and the interface problem without its factor (its factor made in its solve and released
// after it, or its solve by regions under the largest bound that fits) unless keeping its factor needs less; Auto the
// first that fits of: all at once; in
// sequence; in sequence with the units condensed without their factor; with the equations of the fewest
// units released besides, those whose release frees the most first; with every unit's released and the recoveries by
// regions under the largest bound that fits; then, when none of these fits, the same ladder with the interface problem
// without its factor (the first that fits with its least region, then its largest bound that fits;
// its least region, its largest panel, when none fits: the refusal names the smallest need). A strategy the analysis
// does not allow (Inputs::allowsSequential, allowsRelease) is never taken; the interface without its factor needs the
// recovery in sequence to be allowed (a static analysis that keeps no unit's system for later, or the load basis).
//
// The figures are upper bounds of the reservations (UnitFootprint, InterfaceFootprint): a budget that passes may hold
// more units at once than the plan counts, never fewer.
//
// Thread-safety: immutable once built. Ownership: copies the footprints.
class CondensedMemoryPlan
{
public:
    struct Inputs
    {
        std::span<const UnitFootprint> units; // in the order the scheduler condenses them
        InterfaceFootprint interface;         // of the largest constraint set
        // One load case of a batch (CondensedStatics' case vectors: the interface's, the batch's on the g set, those of
        // one unit).
        std::uint64_t caseBytes = 0;
        std::uint64_t resultBytes = 0; // the result tables and vectors kept to the end of the analysis
        // What the baseline holds that is released before the interface phase (the g-set vectors of the patterns).
        std::uint64_t releasedBaselineBytes = 0;
        // Whether the analysis may recover its units in sequence, and release their equations (a static analysis that
        // keeps no unit's system for later: CondensedStatics).
        bool allowsSequential = false;
        bool allowsRelease = false;
    };

    // The plan of MemoryMode::Speed: every recovery at once, every unit's equations kept.
    CondensedMemoryPlan(const Inputs& inputs, std::uint64_t baselineBytes);
    // The plan of `mode` for a budget of `limitBytes` (the class comment).
    CondensedMemoryPlan(const Inputs& inputs, std::uint64_t baselineBytes, MemoryMode mode, std::uint64_t limitBytes);

    [[nodiscard]] MemoryMode mode() const noexcept { return m_mode; }
    // The interface phase it was made from (Inputs::interface).
    [[nodiscard]] const InterfaceFootprint& interface() const noexcept { return m_interface; }
    [[nodiscard]] const MemoryStrategy& strategy() const noexcept { return m_strategy; }
    // Whether the unit at position `unit` of Inputs::units releases its equations.
    [[nodiscard]] bool isReleased(std::size_t unit) const noexcept
    {
        return unit < m_strategy.isReleased.size() && m_strategy.isReleased[unit] != 0;
    }
    // Why the strategy was taken (what Auto switched and why, what LowMemory always does).
    [[nodiscard]] const std::string& reason() const noexcept { return m_reason; }

    // The chosen strategy's figures.
    [[nodiscard]] std::uint64_t baselineBytes() const noexcept { return m_baseline; }
    [[nodiscard]] std::uint64_t condensationPeakBytes() const noexcept { return m_figures.condensationPeak; }
    // The interface and recovery phase beyond what the units keep (the interface of the largest set, the recoveries,
    // one case, the results).
    [[nodiscard]] std::uint64_t laterPhaseBytes() const noexcept { return m_figures.laterPhase; }
    [[nodiscard]] std::uint64_t keptBytes() const noexcept { return m_figures.kept; }
    // max(condensation peak, kept + later phase − the baseline released before it): beyond the baseline.
    [[nodiscard]] std::uint64_t minimumPeakBytes() const noexcept { return m_figures.minimumPeak; }
    // baseline + minimumPeakBytes(): the smallest budget the analysis fits.
    [[nodiscard]] std::uint64_t minimumBudgetBytes() const noexcept { return m_baseline + minimumPeakBytes(); }
    // The smallest budget of the reference strategies (the report): every recovery at once; in sequence with every
    // unit's equations kept; in sequence with every unit's equations released (0 for a strategy not allowed).
    [[nodiscard]] std::uint64_t allAtOnceBudgetBytes() const noexcept { return m_allAtOnceBudget; }
    [[nodiscard]] std::uint64_t sequentialBudgetBytes() const noexcept { return m_sequentialBudget; }
    [[nodiscard]] std::uint64_t releasedBudgetBytes() const noexcept { return m_releasedBudget; }
    // Step 10.23: in sequence with every unit condensed without its factor and its equations released, its interior
    // factorised again whole at its recovery; the smaller of that and of the recoveries by regions under the least
    // bound (the smallest budget of all, what "low_memory" needs at least): 0 when not allowed.
    [[nodiscard]] std::uint64_t withoutFactorBudgetBytes() const noexcept { return m_withoutFactorBudget; }
    [[nodiscard]] std::uint64_t regionsBudgetBytes() const noexcept { return m_regionsBudget; }
    // Step 10.25: the smallest budget with the interface problem without its factor too (its least region beside the
    // units' least strategy: what "low_memory" needs at least), 0 when not allowed; and the flops the chosen strategy's
    // interface regions factorise again (0 with its factor, or every panel kept).
    [[nodiscard]] std::uint64_t interfaceRegionsBudgetBytes() const noexcept { return m_interfaceRegionsBudget; }
    [[nodiscard]] double interfaceRecomputedFlops() const noexcept
    {
        return m_strategy.isInterfaceWithoutFactor ? m_interface.recomputedFlops(m_strategy.interfaceRegionBytes) : 0.0;
    }

    // The consumers of the minimum peak (what the units keep, the largest condensation, the interface problem, its
    // boundary basis, the set's relations, the recoveries, the case batch, the results, what was held before), largest
    // first.
    [[nodiscard]] std::vector<base::MemoryConsumer> consumers() const;

    // Throws base::MemoryBudgetError, before any work, when the budget's limit is below minimumBudgetBytes(): the
    // message gives the estimate, the mode and its strategy, what is left and the largest consumers, and what to do.
    void requireFits(const base::MemoryBudget& budget) const;

private:
    struct Figures
    {
        std::uint64_t kept = 0;
        std::uint64_t condensationPeak = 0;
        std::uint64_t recovery = 0;        // all at once: Σ recoveryBytes; in sequence: the largest pass
        std::uint64_t refactorisation = 0; // all at once: the largest factorisation of a recovered unit
        std::uint64_t laterPhase = 0;
        std::uint64_t minimumPeak = 0;
    };
    [[nodiscard]] Figures figuresOf(const MemoryStrategy& strategy) const;
    // In sequence with the first `count` units of m_releaseOrder released.
    [[nodiscard]] MemoryStrategy releasing(std::size_t count) const;
    // In sequence, every unit condensed without its factor, the first `count` of m_releaseOrder released, the
    // recoveries by regions under regionBytes (0: none).
    [[nodiscard]] MemoryStrategy withoutFactor(std::size_t count, std::uint64_t regionBytes) const;
    // The region bound of the strategy without factors that fits limitBytes: 0 when every recovered unit's factor made
    // again whole fits, else the largest power of two of bytes that does, else the least region (a unit's largest
    // panel) or 0, whichever needs less; the interface problem as `interface` takes it.
    [[nodiscard]] std::uint64_t regionBoundFor(std::uint64_t limitBytes, const MemoryStrategy& interface) const;
    // `units` with the interface problem without its factor, by regions of regionBytes.
    [[nodiscard]] static MemoryStrategy withInterface(MemoryStrategy units, std::uint64_t regionBytes);
    // The interface's region bound beside `units` that fits limitBytes: its factor (every panel kept) when that fits,
    // else the largest power of two of bytes below it that does, else its least region (its largest panel).
    [[nodiscard]] std::uint64_t interfaceBoundFor(const MemoryStrategy& units, std::uint64_t limitBytes) const;
    [[nodiscard]] std::uint64_t leastInterfaceRegion() const noexcept;
    // The ladder of Auto with the interface as `interface` takes it: the first strategy that fits, or the last.
    [[nodiscard]] MemoryStrategy autoLadder(std::uint64_t limitBytes, const MemoryStrategy& interface) const;
    void choose(MemoryMode mode, std::uint64_t limitBytes);
    // ", the units whose factor is above <bound> recovered by regions..." (or factorised again whole: 0).
    [[nodiscard]] static std::string regionText(std::uint64_t regionBytes);
    // ", the interface problem without its factor (...)" (empty with its factor).
    [[nodiscard]] std::string interfaceText(const MemoryStrategy& strategy) const;

    std::vector<UnitFootprint> m_units;
    InterfaceFootprint m_interface;
    std::uint64_t m_caseBytes = 0;
    std::uint64_t m_resultBytes = 0;
    std::uint64_t m_releasedBaseline = 0;
    bool m_allowsSequential = false;
    bool m_allowsRelease = false;
    bool m_allowsInterfaceWithoutFactor = false;
    std::uint64_t m_baseline = 0;
    // The units by decreasing bytes their release frees (ties by position): Auto releases a prefix.
    std::vector<std::size_t> m_releaseOrder;
    MemoryMode m_mode = MemoryMode::Speed;
    MemoryStrategy m_strategy;
    Figures m_figures;
    std::string m_reason;
    std::uint64_t m_allAtOnceBudget = 0;
    std::uint64_t m_sequentialBudget = 0;
    std::uint64_t m_releasedBudget = 0;
    std::uint64_t m_withoutFactorBudget = 0;
    std::uint64_t m_regionsBudget = 0;
    std::uint64_t m_interfaceRegionsBudget = 0;
    std::uint64_t m_leastRegionBytes = 0;   // the largest panel of a recovered unit: a region holds one at least
    std::uint64_t m_largestFactorBytes = 0; // above it, no unit is recovered by regions
};

} // namespace oos::reduction
