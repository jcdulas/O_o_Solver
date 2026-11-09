#include "oos/reduction/scheduling/CondensedMemoryPlan.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <bit>
#include <numeric>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

// How many consumers a refusal names (as MemoryBudget's own refusals).
constexpr std::size_t LISTED_CONSUMERS = 5;

std::string unitsText(std::size_t count)
{
    return base::concatenate({std::to_string(count), count == 1 ? " unit" : " units"});
}

} // namespace

const char* nameOf(RecoveryStrategy strategy) noexcept
{
    switch (strategy)
    {
    case RecoveryStrategy::AllAtOnce:
        return "all at once";
    case RecoveryStrategy::Sequential:
        return "in sequence";
    }
    return "";
}

CondensedMemoryPlan::CondensedMemoryPlan(const Inputs& inputs, std::uint64_t baselineBytes)
    : CondensedMemoryPlan(inputs, baselineBytes, MemoryMode::Speed, 0)
{
}

CondensedMemoryPlan::CondensedMemoryPlan(const Inputs& inputs, std::uint64_t baselineBytes, MemoryMode mode,
                                         std::uint64_t limitBytes)
    : m_units(inputs.units.begin(), inputs.units.end()),
      m_interface(inputs.interface),
      m_caseBytes(inputs.caseBytes),
      m_resultBytes(inputs.resultBytes),
      m_releasedBaseline(std::min(inputs.releasedBaselineBytes, baselineBytes)),
      m_allowsSequential(inputs.allowsSequential),
      m_allowsRelease(inputs.allowsRelease && inputs.allowsSequential),
      m_allowsInterfaceWithoutFactor(inputs.allowsRelease && inputs.allowsSequential &&
                                     inputs.interface.factorBytes > 0),
      m_baseline(baselineBytes)
{
    m_releaseOrder.resize(m_units.size());
    std::iota(m_releaseOrder.begin(), m_releaseOrder.end(), std::size_t{0});
    const auto freed = [this](std::size_t j) { return m_units[j].keptBytes - m_units[j].keptBytesOf(true); };
    std::stable_sort(m_releaseOrder.begin(), m_releaseOrder.end(),
                     [&](std::size_t a, std::size_t b) { return freed(a) > freed(b); });
    for (const UnitFootprint& unit : m_units)
        if (unit.isRecovered)
        {
            m_leastRegionBytes = std::max(m_leastRegionBytes, unit.largestPanelBytes);
            m_largestFactorBytes = std::max(m_largestFactorBytes, unit.factorBytes);
        }
    m_allAtOnceBudget = m_baseline + figuresOf(MemoryStrategy{}).minimumPeak;
    if (m_allowsSequential)
        m_sequentialBudget = m_baseline + figuresOf(releasing(0)).minimumPeak;
    if (m_allowsRelease)
    {
        m_releasedBudget = m_baseline + figuresOf(releasing(m_units.size())).minimumPeak;
        m_withoutFactorBudget = m_baseline + figuresOf(withoutFactor(m_units.size(), 0)).minimumPeak;
        // The least region when that is smaller (a recovery by regions holds its forward factorisation beside the
        // region: on a small unit it may need more than its factor made again whole).
        m_regionsBudget = std::min(
            m_withoutFactorBudget,
            m_baseline +
                figuresOf(withoutFactor(m_units.size(), std::max<std::uint64_t>(m_leastRegionBytes, 1))).minimumPeak);
    }
    if (m_allowsInterfaceWithoutFactor)
    {
        // Step 10.25: the units' least strategy (as m_regionsBudget) with the interface's least region.
        const std::uint64_t least = leastInterfaceRegion();
        const std::uint64_t whole = figuresOf(withInterface(withoutFactor(m_units.size(), 0), least)).minimumPeak;
        const std::uint64_t regions =
            figuresOf(
                withInterface(withoutFactor(m_units.size(), std::max<std::uint64_t>(m_leastRegionBytes, 1)), least))
                .minimumPeak;
        m_interfaceRegionsBudget = m_baseline + std::min(whole, regions);
    }
    choose(mode, limitBytes);
}

MemoryStrategy CondensedMemoryPlan::withInterface(MemoryStrategy units, std::uint64_t regionBytes)
{
    units.isInterfaceWithoutFactor = true;
    units.interfaceRegionBytes = regionBytes;
    return units;
}

std::uint64_t CondensedMemoryPlan::leastInterfaceRegion() const noexcept
{
    return std::max<std::uint64_t>(m_interface.leastRegionBytes, 1);
}

std::uint64_t CondensedMemoryPlan::interfaceBoundFor(const MemoryStrategy& units, std::uint64_t limitBytes) const
{
    const std::uint64_t least = leastInterfaceRegion();
    const std::uint64_t whole = std::max(m_interface.factorBytes, least);
    const auto fits = [&](std::uint64_t bound)
    { return m_baseline + figuresOf(withInterface(units, bound)).minimumPeak <= limitBytes; };
    // The factor made in the solve and released after it, then regions of decreasing powers of two: the larger the
    // bound, the fewer subtrees factorised again (InterfaceEstimate::regionCosts tabulates these bounds).
    if (fits(whole))
        return whole;
    for (std::uint64_t bound = std::bit_floor(whole); bound > least; bound /= 2)
        if (bound < whole && fits(bound))
            return bound;
    return least;
}

MemoryStrategy CondensedMemoryPlan::autoLadder(std::uint64_t limitBytes, const MemoryStrategy& interface) const
{
    const auto with = [&](MemoryStrategy units)
    {
        units.isInterfaceWithoutFactor = interface.isInterfaceWithoutFactor;
        units.interfaceRegionBytes = interface.interfaceRegionBytes;
        return units;
    };
    const auto fits = [&](const MemoryStrategy& strategy)
    { return m_baseline + figuresOf(strategy).minimumPeak <= limitBytes; };
    const std::size_t unitCount = m_units.size();
    for (const MemoryStrategy& strategy : {with(MemoryStrategy{}), with(releasing(0)), with(withoutFactor(0, 0))})
        if (fits(strategy))
            return strategy;
    if (fits(with(withoutFactor(unitCount, 0))))
    {
        std::size_t low = 1;
        std::size_t high = unitCount;
        while (low < high)
        {
            const std::size_t middle = low + (high - low) / 2;
            if (fits(with(withoutFactor(middle, 0))))
                high = middle;
            else
                low = middle + 1;
        }
        return with(withoutFactor(low, 0));
    }
    return with(withoutFactor(unitCount, regionBoundFor(limitBytes, interface)));
}

MemoryStrategy CondensedMemoryPlan::releasing(std::size_t count) const
{
    MemoryStrategy strategy;
    strategy.recovery = RecoveryStrategy::Sequential;
    if (count == 0)
        return strategy;
    strategy.isReleased.assign(m_units.size(), 0);
    for (std::size_t k = 0; k < count && k < m_releaseOrder.size(); ++k)
        strategy.isReleased[m_releaseOrder[k]] = 1;
    strategy.releasedCount = std::min(count, m_units.size());
    return strategy;
}

MemoryStrategy CondensedMemoryPlan::withoutFactor(std::size_t count, std::uint64_t regionBytes) const
{
    MemoryStrategy strategy = releasing(count);
    strategy.isWithoutFactor = true;
    strategy.regionBytes = regionBytes;
    return strategy;
}

std::uint64_t CondensedMemoryPlan::regionBoundFor(std::uint64_t limitBytes, const MemoryStrategy& interface) const
{
    const std::uint64_t least = std::max<std::uint64_t>(m_leastRegionBytes, 1);
    const auto budgetWith = [&](std::uint64_t bound)
    {
        MemoryStrategy strategy = withoutFactor(m_units.size(), bound);
        strategy.isInterfaceWithoutFactor = interface.isInterfaceWithoutFactor;
        strategy.interfaceRegionBytes = interface.interfaceRegionBytes;
        return m_baseline + figuresOf(strategy).minimumPeak;
    };
    // Every recovered unit's factor made again whole when that fits; otherwise the largest power of two that fits
    // (each region is held in a pass: the figures grow with the bound); when none fits, the smaller need of the whole
    // factors and of the least region (the refusal names it).
    if (budgetWith(0) <= limitBytes)
        return 0;
    for (std::uint64_t bound = std::bit_floor(std::max(m_largestFactorBytes, least)); bound > least; bound /= 2)
        if (budgetWith(bound) <= limitBytes)
            return bound;
    return budgetWith(least) < budgetWith(0) ? least : 0;
}

CondensedMemoryPlan::Figures CondensedMemoryPlan::figuresOf(const MemoryStrategy& strategy) const
{
    Figures figures;
    const bool isSequential = strategy.recovery == RecoveryStrategy::Sequential;
    for (std::size_t j = 0; j < m_units.size(); ++j)
    {
        const UnitFootprint& unit = m_units[j];
        const bool isReleased = j < strategy.isReleased.size() && strategy.isReleased[j] != 0;
        // One unit at a time: the units before it keep their data, it holds its equations and its condensation, and
        // when released the system without equations made beside them.
        const std::uint64_t condensation =
            unit.condensationBytes(strategy.isWithoutFactor) + (isReleased ? unit.keptBytesOf(true) : 0);
        figures.condensationPeak = std::max(figures.condensationPeak, figures.kept + unit.keptBytes + condensation);
        figures.kept += unit.keptBytesOf(isReleased);
        if (!unit.isRecovered)
            continue;
        if (isSequential)
        {
            figures.recovery = std::max(
                figures.recovery, unit.passPeakBytes(isReleased, strategy.isWithoutFactor ? strategy.regionBytes : 0));
            continue;
        }
        figures.recovery += unit.recoveryBytes;
        figures.refactorisation = std::max(figures.refactorisation, unit.factorisationBytes);
    }
    figures.laterPhase =
        m_interface.peakBytes(figures.recovery + figures.refactorisation + m_caseBytes + m_resultBytes,
                              strategy.isInterfaceWithoutFactor, strategy.interfaceRegionBytes, m_resultBytes);
    const std::uint64_t later = figures.kept + figures.laterPhase;
    figures.minimumPeak =
        std::max(figures.condensationPeak, later > m_releasedBaseline ? later - m_releasedBaseline : 0);
    return figures;
}

std::string CondensedMemoryPlan::interfaceText(const MemoryStrategy& strategy) const
{
    if (!strategy.isInterfaceWithoutFactor)
        return {};
    if (strategy.interfaceRegionBytes >= m_interface.factorBytes)
        return base::concatenate({"; the interface problem without its factor (its factor of ",
                                  base::formatBytes(m_interface.factorBytes),
                                  " made in its solve and released after it)"});
    return base::concatenate({"; the interface problem without its factor (",
                              base::formatBytes(m_interface.factorBytes),
                              "), its solve by regions of its elimination tree of at most ",
                              base::formatBytes(strategy.interfaceRegionBytes), " factorised again"});
}

std::string CondensedMemoryPlan::regionText(std::uint64_t regionBytes)
{
    return regionBytes == 0 ? std::string(", their interiors factorised again whole at their recovery")
                            : base::concatenate({", the units whose factor is above ", base::formatBytes(regionBytes),
                                                 " recovered by regions of their elimination tree (none above it)"});
}

void CondensedMemoryPlan::choose(MemoryMode mode, std::uint64_t limitBytes)
{
    m_mode = mode;
    const auto fits = [&](std::uint64_t budget) { return budget <= limitBytes; };
    const auto take = [&](MemoryStrategy strategy, std::string reason)
    {
        m_figures = figuresOf(strategy);
        m_strategy = std::move(strategy);
        m_reason = std::move(reason);
    };
    const std::size_t unitCount = m_units.size();
    const std::string allAtOnce = base::formatBytes(m_allAtOnceBudget);
    if (mode == MemoryMode::Speed || !m_allowsSequential)
    {
        take(MemoryStrategy{},
             mode == MemoryMode::Speed
                 ? "speed: every unit's equations kept and every recovery made at once"
                 : base::concatenate({nameOf(mode),
                                      ": every unit's equations kept and every recovery made at once (this "
                                      "analysis keeps its units' systems for later)"}));
        return;
    }
    if (mode == MemoryMode::LowMemory)
    {
        if (!m_allowsRelease)
        {
            take(releasing(0), base::concatenate({"low_memory: the units recovered in sequence (every recovery at once "
                                                  "would need ",
                                                  allAtOnce, ")"}));
            return;
        }
        // Step 10.23: every unit condensed without its factor, its equations released; by regions only when its factor
        // made again does not fit (the largest bound that fits). Step 10.25: the interface problem without its factor
        // too, by regions only when its factor made in its solve does not fit (the largest bound that fits), unless
        // keeping its factor needs less (a small interface, whose solve's fronts and least region outweigh its factor).
        std::uint64_t bound = regionBoundFor(limitBytes, MemoryStrategy{});
        MemoryStrategy strategy = withoutFactor(unitCount, bound);
        if (m_allowsInterfaceWithoutFactor)
        {
            const std::uint64_t unitBound =
                regionBoundFor(limitBytes, withInterface(MemoryStrategy{}, leastInterfaceRegion()));
            MemoryStrategy without = withoutFactor(unitCount, unitBound);
            without = withInterface(without, interfaceBoundFor(without, limitBytes));
            if (figuresOf(without).minimumPeak <= figuresOf(strategy).minimumPeak)
            {
                strategy = std::move(without);
                bound = unitBound;
            }
        }
        std::string reason = base::concatenate(
            {"low_memory: the units recovered in sequence, condensed without their factor, their equations released "
             "after their condensation and built again at their recovery",
             regionText(bound), interfaceText(strategy), " (every recovery at once would need ", allAtOnce, ")"});
        take(std::move(strategy), std::move(reason));
        return;
    }
    // Auto: the first strategy that fits.
    const std::string limit = base::formatBytes(limitBytes);
    if (fits(m_allAtOnceBudget))
    {
        take(MemoryStrategy{}, base::concatenate({"auto: everything kept, every recovery at once (", allAtOnce,
                                                  " fit the budget of ", limit, ")"}));
        return;
    }
    if (fits(m_sequentialBudget) || !m_allowsRelease)
    {
        take(releasing(0),
             base::concatenate({"auto: the units recovered in sequence: every recovery at once needs ", allAtOnce,
                                ", the budget is ", limit, ", in sequence ", base::formatBytes(m_sequentialBudget)}));
        return;
    }
    // In sequence without the units' factors (their condensation's peak is their fronts), then with the
    // equations of the fewest units released besides, those that free the most first (the figures shrink as more units
    // are released: their kept data and the condensation's peak; the largest pass is taken by the first release
    // already), then every unit's released and the recoveries by regions under the largest bound that fits (the least
    // when none fits: requireFits refuses it with its consumers).
    const std::uint64_t withoutFactors = m_baseline + figuresOf(withoutFactor(0, 0)).minimumPeak;
    if (fits(withoutFactors))
    {
        take(withoutFactor(0, 0),
             base::concatenate({"auto: the units recovered in sequence and condensed without their factor (their "
                                "interiors factorised again at their recovery): every recovery at once needs ",
                                allAtOnce, ", in sequence ", base::formatBytes(m_sequentialBudget), ", the budget is ",
                                limit, ", without the factors ", base::formatBytes(withoutFactors)}));
        return;
    }
    if (fits(m_withoutFactorBudget))
    {
        std::size_t low = 1;
        std::size_t high = unitCount;
        while (low < high)
        {
            const std::size_t middle = low + (high - low) / 2;
            if (fits(m_baseline + figuresOf(withoutFactor(middle, 0)).minimumPeak))
                high = middle;
            else
                low = middle + 1;
        }
        MemoryStrategy strategy = withoutFactor(low, 0);
        const std::uint64_t needed = m_baseline + figuresOf(strategy).minimumPeak;
        take(std::move(strategy),
             base::concatenate({"auto: the units recovered in sequence and condensed without their factor, the "
                                "equations of ",
                                unitsText(low), " of ", std::to_string(unitCount),
                                " released after their condensation and built again at their recovery: every recovery "
                                "at once needs ",
                                allAtOnce, ", in sequence ", base::formatBytes(m_sequentialBudget), ", the budget is ",
                                limit, ", with these releases ", base::formatBytes(needed)}));
        return;
    }
    const std::uint64_t bound = regionBoundFor(limitBytes, MemoryStrategy{});
    MemoryStrategy strategy = withoutFactor(unitCount, bound);
    const std::uint64_t needed = m_baseline + figuresOf(strategy).minimumPeak;
    if (fits(needed) || !m_allowsInterfaceWithoutFactor)
    {
        take(std::move(strategy),
             base::concatenate({"auto: the units recovered in sequence, condensed without their factor, every unit's "
                                "equations released",
                                regionText(bound), ": every recovery at once needs ", allAtOnce,
                                ", every unit's equations released ", base::formatBytes(m_withoutFactorBudget),
                                ", the budget is ", limit, ", by these regions ", base::formatBytes(needed)}));
        return;
    }
    // Step 10.25: no strategy fits with the interface's factor kept: the same ladder with the interface problem
    // without its factor, the units' strategy the first that fits beside the interface's least region, then the
    // interface's largest bound that fits beside it.
    const MemoryStrategy units = autoLadder(limitBytes, withInterface(MemoryStrategy{}, leastInterfaceRegion()));
    MemoryStrategy chosen = withInterface(units, interfaceBoundFor(units, limitBytes));
    const std::uint64_t chosenNeed = m_baseline + figuresOf(chosen).minimumPeak;
    if (!fits(chosenNeed) && needed <= chosenNeed)
    {
        // Neither fits, and without the interface's factor needs no less: the refusal names the smaller need.
        take(std::move(strategy),
             base::concatenate({"auto: the units recovered in sequence, condensed without their factor, every unit's "
                                "equations released",
                                regionText(bound), ": every recovery at once needs ", allAtOnce,
                                ", every unit's equations released ", base::formatBytes(m_withoutFactorBudget),
                                ", the budget is ", limit, ", by these regions ", base::formatBytes(needed),
                                ", with the interface problem without its factor ", base::formatBytes(chosenNeed)}));
        return;
    }
    std::string unitsDescription;
    if (chosen.recovery == RecoveryStrategy::AllAtOnce)
        unitsDescription = "every recovery at once";
    else
    {
        unitsDescription = "the units recovered in sequence";
        if (chosen.isWithoutFactor)
            unitsDescription.append(", condensed without their factor");
        if (chosen.releasedCount > 0)
            unitsDescription.append(base::concatenate({", the equations of ", unitsText(chosen.releasedCount), " of ",
                                                       std::to_string(unitCount), " released"}));
        if (chosen.isWithoutFactor)
            unitsDescription.append(regionText(chosen.regionBytes));
    }
    std::string reason = base::concatenate(
        {"auto: ", unitsDescription, interfaceText(chosen),
         ": with the interface's factor kept, even every unit's equations released and their recoveries by regions "
         "need ",
         base::formatBytes(needed), ", the budget is ", limit, ", this strategy ", base::formatBytes(chosenNeed)});
    take(std::move(chosen), std::move(reason));
}

std::vector<base::MemoryConsumer> CondensedMemoryPlan::consumers() const
{
    std::vector<base::MemoryConsumer> list; // bounded by the units
    if (m_baseline > 0)
        list.push_back({"held before the condensation (the model, its DOF sets and loads)", m_baseline});
    const Figures& figures = m_figures;
    const std::uint64_t later = figures.kept + figures.laterPhase;
    const bool isLaterPhase = (later > m_releasedBaseline ? later - m_releasedBaseline : 0) >= figures.condensationPeak;
    const std::size_t released = m_strategy.releasedCount;
    if (isLaterPhase)
    {
        std::uint64_t keptWithEquations = 0;
        std::uint64_t keptReleased = 0;
        for (std::size_t j = 0; j < m_units.size(); ++j)
        {
            if (isReleased(j))
                keptReleased += m_units[j].keptBytesOf(true);
            else
                keptWithEquations += m_units[j].keptBytes;
        }
        if (m_units.size() > released)
            list.push_back({base::concatenate({"what ", unitsText(m_units.size() - released),
                                               " keep (their equations, patterns and K*)"}),
                            keptWithEquations});
        if (released > 0)
            list.push_back({base::concatenate({"what ", unitsText(released),
                                               " whose equations are released keep (their boundary data, patterns and "
                                               "K*)"}),
                            keptReleased});
        if (m_strategy.isInterfaceWithoutFactor)
        {
            list.push_back({"the interface problem without its factor (its matrix)",
                            std::max(m_interface.problemWithoutFactorBytes, m_interface.heldWithoutFactorBytes)});
            list.push_back({m_strategy.interfaceRegionBytes >= m_interface.factorBytes
                                ? "the solve of the interface problem (its factor made in it, its fronts)"
                                : "the solve of the interface problem by regions (its fronts, a region of its panels)",
                            m_interface.solveBytes(m_strategy.interfaceRegionBytes)});
        }
        else
            list.push_back(
                {"the interface problem and its factor", std::max(m_interface.problemBytes, m_interface.heldBytes)});
        list.push_back({"the boundary basis of the interface", m_interface.basisBytes});
        list.push_back({"the constraint set's relations, DOF sets and force vectors", m_interface.modelBytes});
        if (figures.recovery + figures.refactorisation > 0)
            list.push_back({m_strategy.recovery == RecoveryStrategy::AllAtOnce
                                ? "the recoveries of the recovered units, all at once"
                            : m_strategy.regionBytes > 0
                                ? "the recovery of the largest unit in sequence (its equations built again when "
                                  "released, its regions or its factor, and its recovery)"
                                : "the recovery of the largest unit in sequence (its equations built again when "
                                  "released, its factor and its recovery)",
                            figures.recovery + figures.refactorisation});
        list.push_back({"one load case of a batch", m_caseBytes});
        list.push_back({"the result tables", m_resultBytes});
    }
    else
    {
        // The unit whose condensation makes the peak, and what the units before it keep.
        std::uint64_t kept = 0;
        std::size_t peakUnit = m_units.size();
        for (std::size_t j = 0; j < m_units.size(); ++j)
        {
            const std::uint64_t condensation = m_units[j].condensationBytes(m_strategy.isWithoutFactor) +
                                               (isReleased(j) ? m_units[j].keptBytesOf(true) : 0);
            if (kept + m_units[j].keptBytes + condensation == figures.condensationPeak)
            {
                peakUnit = j;
                break;
            }
            kept += m_units[j].keptBytesOf(isReleased(j));
        }
        if (peakUnit < m_units.size())
        {
            if (peakUnit > 0)
                list.push_back({base::concatenate({"what the ", unitsText(peakUnit), " condensed before unit ",
                                                   std::to_string(m_units[peakUnit].unitId), " keep"}),
                                kept});
            list.push_back(
                {base::concatenate({"unit ", std::to_string(m_units[peakUnit].unitId), ": its equations and patterns"}),
                 m_units[peakUnit].keptBytes});
            list.push_back({base::concatenate({"unit ", std::to_string(m_units[peakUnit].unitId),
                                               m_strategy.isWithoutFactor
                                                   ? ": its condensation without its factor (its fronts, K*)"
                                                   : ": its condensation (the interior factor, the fronts, K*)"}),
                            m_units[peakUnit].condensationBytes(m_strategy.isWithoutFactor)});
        }
    }
    std::stable_sort(list.begin(), list.end(),
                     [](const base::MemoryConsumer& a, const base::MemoryConsumer& b) { return a.bytes > b.bytes; });
    return list;
}

void CondensedMemoryPlan::requireFits(const base::MemoryBudget& budget) const
{
    const std::uint64_t needed = minimumBudgetBytes();
    if (needed <= budget.limit())
        return;
    std::vector<base::MemoryConsumer> largest = consumers();
    if (largest.size() > LISTED_CONSUMERS)
        largest.resize(LISTED_CONSUMERS);
    const std::uint64_t peak = minimumPeakBytes();
    const std::uint64_t available = budget.available();
    std::string strategy = m_strategy.releasedCount == 0
                               ? std::string(nameOf(m_strategy.recovery))
                               : base::concatenate({nameOf(m_strategy.recovery), ", the equations of ",
                                                    unitsText(m_strategy.releasedCount), " released"});
    if (m_strategy.isWithoutFactor)
        strategy.append(", the units condensed without their factor").append(regionText(m_strategy.regionBytes));
    strategy.append(interfaceText(m_strategy));
    std::string message = base::concatenate(
        {"memory budget exceeded: the condensed analysis needs at least ", base::formatBytes(needed), " (",
         base::formatBytes(m_baseline), " held before it and ", base::formatBytes(peak),
         " estimated before any unit is condensed, with one unit condensed at a time and every factor released; memory "
         "mode \"",
         nameOf(m_mode), "\": the recoveries ", strategy, "), ", base::formatBytes(available), " of the budget of ",
         base::formatBytes(budget.limit()), " are left"});
    message.append("; largest consumers:");
    for (const base::MemoryConsumer& consumer : largest)
        message.append(" ").append(consumer.label).append(" (").append(base::formatBytes(consumer.bytes)).append(")");
    message.append(". Raise memory=");
    const std::uint64_t lowMemory =
        m_interfaceRegionsBudget > 0 ? std::min(m_regionsBudget, m_interfaceRegionsBudget) : m_regionsBudget;
    if (m_mode != MemoryMode::LowMemory && m_allowsRelease && lowMemory < needed)
        message.append(", use memory_mode=\"low_memory\" (it needs ").append(base::formatBytes(lowMemory)).append(")");
    message.append(
        ", keep large units full or split them (Solver.units(\"dissection\") with a smaller "
        "max_unit_bytes), recover fewer units, or restrict the requested outputs");
    throw base::MemoryBudgetError(message, base::MemoryShortfall{peak, available, budget.limit(), std::move(largest)});
}

} // namespace oos::reduction
