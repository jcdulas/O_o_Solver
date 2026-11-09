#include "oos/reduction/absorption/AbsorptionEligibility.hpp"

#include "oos/input/tables/ContactTable.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <optional>
#include <utility>

namespace oos::reduction
{

namespace
{

// The legs of `spider` in the sorted list `nodes`.
std::vector<base::EntityId> legsIn(const Spider& spider, const std::vector<base::EntityId>& nodes)
{
    std::vector<base::EntityId> found;
    std::set_intersection(spider.legNodeIds.begin(), spider.legNodeIds.end(), nodes.begin(), nodes.end(),
                          std::back_inserter(found));
    return found;
}

} // namespace

std::string_view describeDecision(AbsorptionDecision decision) noexcept
{
    switch (decision)
    {
    case AbsorptionDecision::Absorbed:
        return "absorbed: its reference point stands for its legs on the boundary";
    case AbsorptionDecision::AbsorbedWithCluster:
        return "absorbed with its cluster, which touches one unit only";
    case AbsorptionDecision::PartOfUnit:
        return "part of a unit (an element list holds it)";
    case AbsorptionDecision::NotConnection:
        return "not a connection of the connection system";
    case AbsorptionDecision::NotSpider:
        return "not a spider";
    case AbsorptionDecision::LegsOnSeveralUnits:
        return "kept on its legs: its legs are on several units, or one on none";
    case AbsorptionDecision::LegTouched:
        return "kept on its legs: a leg is touched by another connection";
    case AbsorptionDecision::NoEligibility:
        return "kept on its legs: the piercing table does not say whether another connection touches a leg";
    case AbsorptionDecision::LegRetained:
        return "kept on its legs: a leg is retained";
    case AbsorptionDecision::LegOnSplineNode:
        return "kept on its legs: a leg is a spline node";
    case AbsorptionDecision::LegOnSupport:
        return "kept on its legs: a leg is a FreeBodySupport point";
    case AbsorptionDecision::LegPartiallyConstrained:
        return "kept on its legs: a leg has a DOF constrained in some subcases only";
    case AbsorptionDecision::LegConstrained:
        return "kept on its legs: a leg DOF of the relation is constrained in every subcase";
    case AbsorptionDecision::NotRequested:
        return "kept on its legs: spider absorption not requested";
    case AbsorptionDecision::NoRelation:
        return "kept on its legs: its relation could not be built";
    case AbsorptionDecision::RankDeficient:
        return "kept on its legs: its legs do not fix every reference DOF (rank-deficient weights)";
    case AbsorptionDecision::IllConditioned:
        return "kept on its legs: its change of basis is above SPIDER_ABSORPTION_MAX_CONDITION";
    }
    return "";
}

std::string_view decisionCode(AbsorptionDecision decision) noexcept
{
    switch (decision)
    {
    case AbsorptionDecision::Absorbed:
        return "absorbed";
    case AbsorptionDecision::AbsorbedWithCluster:
        return "absorbedWithCluster";
    case AbsorptionDecision::PartOfUnit:
        return "partOfUnit";
    case AbsorptionDecision::NotConnection:
        return "notConnection";
    case AbsorptionDecision::NotSpider:
        return "notSpider";
    case AbsorptionDecision::LegsOnSeveralUnits:
        return "legsOnSeveralUnits";
    case AbsorptionDecision::LegTouched:
        return "legTouched";
    case AbsorptionDecision::NoEligibility:
        return "noEligibility";
    case AbsorptionDecision::LegRetained:
        return "legRetained";
    case AbsorptionDecision::LegOnSplineNode:
        return "legOnSplineNode";
    case AbsorptionDecision::LegOnSupport:
        return "legOnSupport";
    case AbsorptionDecision::LegPartiallyConstrained:
        return "legPartiallyConstrained";
    case AbsorptionDecision::LegConstrained:
        return "legConstrained";
    case AbsorptionDecision::NotRequested:
        return "notRequested";
    case AbsorptionDecision::NoRelation:
        return "noRelation";
    case AbsorptionDecision::RankDeficient:
        return "rankDeficient";
    case AbsorptionDecision::IllConditioned:
        return "illConditioned";
    }
    return "";
}

AbsorptionDecision AbsorptionEligibility::topological(const Spider& spider, std::vector<base::EntityId>& concerned)
{
    concerned.clear();
    if (!spider.isSpider)
        return AbsorptionDecision::NotSpider;
    if (!spider.hasEligibility)
        return AbsorptionDecision::NoEligibility;
    if (!spider.isEligible)
    {
        concerned = legsIn(spider, spider.blockingNodeIds);
        return AbsorptionDecision::LegTouched;
    }
    return AbsorptionDecision::Absorbed;
}

AbsorptionDecision AbsorptionEligibility::glued(const Spider& spider, const AbsorptionExclusions& exclusions,
                                                std::vector<base::EntityId>& concerned,
                                                std::vector<BlockingConnection>& blocking)
{
    concerned.clear();
    blocking.clear();
    const std::span<const GluedPoint> points = exclusions.gluedPoints;
    for (const base::EntityId leg : spider.legNodeIds)
    {
        const auto isBefore = [](const GluedPoint& point, base::EntityId nodeId) { return point.nodeId < nodeId; };
        const auto first = std::lower_bound(points.begin(), points.end(), leg, isBefore);
        bool isGlued = false;
        for (auto point = first; point != points.end() && point->nodeId == leg; ++point)
        {
            isGlued = true;
            const std::optional<input::ContactTableName> name = input::parseContactTableName(point->table);
            blocking.push_back(name ? BlockingConnection{leg, "contact", name->card, name->id}
                                    : BlockingConnection{leg, "contact", "contact", 0});
        }
        if (isGlued)
            concerned.push_back(leg);
    }
    return concerned.empty() ? AbsorptionDecision::Absorbed : AbsorptionDecision::LegTouched;
}

AbsorptionDecision AbsorptionEligibility::analysis(const Spider& spider, const AbsorptionExclusions& exclusions,
                                                   const SpiderMeasurement& measurement,
                                                   std::vector<base::EntityId>& concerned)
{
    const std::array<std::pair<const std::vector<base::EntityId>*, AbsorptionDecision>, 4> rules{{
        {&exclusions.retainedNodes, AbsorptionDecision::LegRetained},
        {&exclusions.splineNodes, AbsorptionDecision::LegOnSplineNode},
        {&exclusions.supportNodes, AbsorptionDecision::LegOnSupport},
        {&exclusions.partiallyConstrainedNodes, AbsorptionDecision::LegPartiallyConstrained},
    }};
    for (const auto& [nodes, decision] : rules)
    {
        concerned = legsIn(spider, *nodes);
        if (!concerned.empty())
            return decision;
    }
    concerned.clear();
    if (measurement.hasLegConstrainedEverywhere)
        return AbsorptionDecision::LegConstrained;
    return AbsorptionDecision::Absorbed;
}

AbsorptionDecision AbsorptionEligibility::numerical(const SpiderMeasurement& measurement, double maxCondition) noexcept
{
    switch (AbsorptionSafeguard::judge(measurement, maxCondition))
    {
    case SafeguardVerdict::Accepted:
        return AbsorptionDecision::Absorbed;
    case SafeguardVerdict::NoRelation:
        return AbsorptionDecision::NoRelation;
    case SafeguardVerdict::RankDeficient:
        return AbsorptionDecision::RankDeficient;
    case SafeguardVerdict::IllConditioned:
        return AbsorptionDecision::IllConditioned;
    }
    return AbsorptionDecision::NoRelation;
}

} // namespace oos::reduction
