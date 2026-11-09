#include "oos/design/sensitivity/SolutionReads.hpp"

#include <algorithm>

namespace oos::design
{

base::BudgetedBuffer<std::uint8_t> SolutionReads::ofWeights(const SensitivitySystem& system,
                                                            const LinearisedItems& linearised,
                                                            std::span<const ItemKey> items, base::MemoryBudget& budget)
{
    if (!system.isCondensed())
        return {};
    const auto gSize = static_cast<std::size_t>(system.dofMap().gSetSize());
    base::BudgetedBuffer<std::uint8_t> reads(budget, gSize, "sensitivity solution reads");
    std::fill(reads.data(), reads.data() + gSize, std::uint8_t{0});
    for (const ItemKey& key : items)
        for (const auto& [dof, weight] : linearised.at(key).weights)
            if (weight != 0.0)
                reads[static_cast<std::size_t>(dof)] = 1;
    return reads;
}

base::BudgetedBuffer<std::uint8_t> SolutionReads::ofPseudoLoads(const SensitivitySystem& system,
                                                                const PseudoLoads& loads, std::size_t subcase,
                                                                base::MemoryBudget& budget)
{
    if (!system.isCondensed())
        return {};
    const auto gSize = static_cast<std::size_t>(system.dofMap().gSetSize());
    base::BudgetedBuffer<std::uint8_t> reads(budget, gSize, "sensitivity solution reads");
    std::fill(reads.data(), reads.data() + gSize, std::uint8_t{0});
    loads.markSupport(subcase, reads.span());
    return reads;
}

} // namespace oos::design
