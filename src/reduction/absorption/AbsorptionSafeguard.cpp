#include "oos/reduction/absorption/AbsorptionSafeguard.hpp"

namespace oos::reduction
{

SpiderMeasurement AbsorptionSafeguard::measure(const SpiderWeightMatrix& weights, base::MemoryBudget& budget)
{
    const SpiderChangeOfBasis basis(weights, budget);
    SpiderMeasurement measurement;
    measurement.hasRelation = true;
    measurement.isRankDeficient = basis.isRankDeficient();
    measurement.conditionNumber = basis.conditionNumber();
    measurement.referenceDofCount = static_cast<std::int32_t>(weights.referenceDofCount);
    return measurement;
}

SpiderMeasurement AbsorptionSafeguard::rigidBodyStarMeasurement(std::int32_t referenceDofCount) noexcept
{
    SpiderMeasurement measurement;
    measurement.hasRelation = true;
    measurement.conditionNumber = 1.0;
    measurement.referenceDofCount = referenceDofCount;
    return measurement;
}

SafeguardVerdict AbsorptionSafeguard::judge(const SpiderMeasurement& measurement, double maxCondition) noexcept
{
    if (!measurement.hasRelation)
        return SafeguardVerdict::NoRelation;
    if (measurement.isRankDeficient)
        return SafeguardVerdict::RankDeficient;
    // A NaN condition number fails the comparison and is refused with the ill-conditioned ones.
    if (!(measurement.conditionNumber <= maxCondition))
        return SafeguardVerdict::IllConditioned;
    return SafeguardVerdict::Accepted;
}

} // namespace oos::reduction
