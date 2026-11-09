#include "oos/design/approximation/ConstraintScreening.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <tuple>

namespace oos::design
{

namespace
{

// The RTYPEs a DesignConstrScreen may name: the response types of DesignResp1 (static, dynamic, aeroelastic and the
// others; a type no response of the problem has screens nothing), EQUA for the DesignResp2, DesignResp3, and ALL.
constexpr std::string_view KNOWN_TYPES[] = {"All",
                                            "Equation",
                                            "DesignResp3",
                                            "Weight",
                                            "Volume",
                                            "FractionalMass",
                                            "Displacement",
                                            "Stress",
                                            "Strain",
                                            "Force",
                                            "ConstraintForce",
                                            "StrainEnergy",
                                            "CompositeStress",
                                            "CompositeStrain",
                                            "CompositeFailure",
                                            "CompositeStrengthRatio",
                                            "Eigenvalue",
                                            "Frequency",
                                            "BucklingFactor",
                                            "ComplexEigenvalue",
                                            "TrimVariable",
                                            "StabilityDerivative",
                                            "Flutter",
                                            "Divergence",
                                            "TotalCompliance",
                                            "TotalStrainEnergy",
                                            "NodeForce",
                                            "NodeForcePanel",
                                            "AbsoluteDisplacement",
                                            "FreqDisplacement",
                                            "FreqVelocity",
                                            "FreqAcceleration",
                                            "FreqConstraintForce",
                                            "FreqStress",
                                            "FreqForce",
                                            "SpectralDisplacement",
                                            "SpectralVelocity",
                                            "SpectralAcceleration",
                                            "RootMeanSquareDisplacement",
                                            "RootMeanSquareVelocity",
                                            "RootMeanSquareAcceleration",
                                            "TransientDisplacement",
                                            "TransientVelocity",
                                            "TransientAcceleration",
                                            "TransientConstraintForce",
                                            "TransientForce",
                                            "TransientStress",
                                            "RadiatedPower",
                                            "AcousticPower",
                                            "AcousticFieldPressure",
                                            "AcousticFieldIntensity",
                                            "AcousticFieldVelocity"};

bool isKnownType(std::string_view type)
{
    return std::find(std::begin(KNOWN_TYPES), std::end(KNOWN_TYPES), type) != std::end(KNOWN_TYPES);
}

} // namespace

std::string_view screeningStatusName(ScreeningStatus status) noexcept
{
    switch (status)
    {
    case ScreeningStatus::KeptExplicitly:
        return "kept explicitly";
    case ScreeningStatus::RetainedByScreening:
        return "retained by screening";
    case ScreeningStatus::ScreenedOut:
        return "screened out";
    }
    return "";
}

ConstraintScreening::ConstraintScreening(std::vector<ScreeningRule> rules) : m_rules(std::move(rules))
{
    std::set<std::string> types;
    for (const ScreeningRule& rule : m_rules)
    {
        if (!isKnownType(rule.responseType))
            throw base::OptimisationError(
                base::concatenate({"DesignConstrScreen ", rule.responseType,
                                   ": expected a response type of DesignResp1, EQUA, DesignResp3 or "
                                   "ALL, found '",
                                   rule.responseType, "'; correct the DesignConstrScreen.responseKind"}));
        if (rule.maximumRetained < 1)
            throw base::OptimisationError(base::concatenate(
                {"DesignConstrScreen ", rule.responseType, ": expected DesignConstrScreen.maxRetained > 0, found ",
                 std::to_string(rule.maximumRetained), "; give at least 1"}));
        if (!types.insert(rule.responseType).second)
            throw base::OptimisationError(
                base::concatenate({"DesignConstrScreen ", rule.responseType,
                                   ": expected one DesignConstrScreen per response type, found two; keep one"}));
    }
}

ScreeningRule ConstraintScreening::ruleFor(std::string_view responseType) const
{
    const ScreeningRule* own = nullptr;
    for (const ScreeningRule& rule : m_rules)
    {
        if (rule.responseType == "All")
            return {std::string(responseType), rule.threshold, rule.maximumRetained};
        if (rule.responseType == responseType)
            own = &rule;
    }
    if (own != nullptr)
        return *own;
    return {std::string(responseType), DEFAULT_THRESHOLD, DEFAULT_MAXIMUM_RETAINED};
}

std::vector<ScreeningStatus> ConstraintScreening::screen(std::span<const ScreeningEntry> entries) const
{
    std::vector<ScreeningStatus> statuses(entries.size(), ScreeningStatus::ScreenedOut);
    // The candidates above TRS, grouped by (type, region, subcase) where NSTR applies.
    std::map<std::tuple<std::string, ScreeningRegion, std::int32_t>, std::vector<std::size_t>> groups;
    std::map<std::string, ScreeningRule> rules;
    for (std::size_t k = 0; k < entries.size(); ++k)
    {
        const ScreeningEntry& entry = entries[k];
        if (entry.isExplicit)
        {
            statuses[k] = ScreeningStatus::KeptExplicitly;
            continue;
        }
        auto rule = rules.find(entry.responseType);
        if (rule == rules.end())
            rule = rules.emplace(entry.responseType, ruleFor(entry.responseType)).first;
        if (!(entry.normalised > rule->second.threshold))
            continue;
        if (entry.region.kind == ScreeningRegion::Kind::Unlimited)
            statuses[k] = ScreeningStatus::RetainedByScreening;
        else
            groups[{entry.responseType, entry.region, entry.subcase}].push_back(k);
    }
    for (auto& [key, members] : groups)
    {
        const auto limit = static_cast<std::size_t>(rules.at(std::get<0>(key)).maximumRetained);
        std::stable_sort(members.begin(), members.end(), [&entries](std::size_t a, std::size_t b)
                         { return entries[a].normalised > entries[b].normalised; });
        for (std::size_t m = 0; m < std::min(limit, members.size()); ++m)
            statuses[members[m]] = ScreeningStatus::RetainedByScreening;
    }
    return statuses;
}

} // namespace oos::design
