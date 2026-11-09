#include "oos/design/approximation/ScreeningReport.hpp"

#include <sstream>

namespace oos::design
{

namespace
{

std::string regionText(const ScreeningRegion& region)
{
    switch (region.kind)
    {
    case ScreeningRegion::Kind::Unlimited:
        return "no DesignConstrScreen.maxRetained";
    case ScreeningRegion::Kind::Region:
        return std::string("DesignResp1.regionId ").append(std::to_string(region.id));
    case ScreeningRegion::Kind::Response:
        return std::string("response ").append(std::to_string(region.id));
    case ScreeningRegion::Kind::Property:
        return std::string(region.propertyCard).append(" ").append(std::to_string(region.id));
    }
    return "";
}

} // namespace

std::string ScreeningReport::text() const
{
    std::ostringstream text;
    text << "Constraint screening: " << keptExplicitly << " kept explicitly, " << retainedByScreening
         << " retained by screening, " << screenedOut << " screened out\n";
    if (keptExplicitly > 0)
        text << "  The explicit group is an extension of O_o_Solver: DesignConstrScreen alone would screen these "
                "constraints as "
                "the others.\n";
    for (const ScreeningRule& rule : rules)
        text << "  " << rule.responseType << ": DesignConstrScreen.threshold = " << rule.threshold
             << ", DesignConstrScreen.maxRetained = " << rule.maximumRetained << '\n';
    for (const ScreeningReportRow& row : rows)
        text << "  " << row.label << " (" << row.responseType << ' ' << row.responseId << ") set " << row.setId
             << " subcase " << row.item.subcase << " entity " << row.item.entity << " item " << row.item.component
             << ' ' << (row.side == BoundSide::Upper ? "upper" : "lower") << ' ' << row.bound << ": value " << row.value
             << ", g = " << row.normalised << ", " << regionText(row.region) << ": " << screeningStatusName(row.status)
             << '\n';
    return text.str();
}

} // namespace oos::design
