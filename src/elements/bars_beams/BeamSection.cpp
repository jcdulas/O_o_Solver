#include "oos/elements/bars_beams/BeamSection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>

namespace oos::elements
{

namespace
{

// The interpolated value of one property between two stations.
double interpolate(double atStart, double atEnd, double fraction)
{
    return atStart + (atEnd - atStart) * fraction;
}

std::string stationName(const BeamSection& section, std::size_t index)
{
    if (index == 0)
        return "end A";
    if (index + 1 == section.count)
        return "end B";
    return std::string("the station at BeamProp.stationRatio ")
        .append(std::to_string(section.stations[index].position));
}

} // namespace

BeamStation BeamSection::at(double position) const
{
    std::size_t next = 1;
    while (next + 1 < count && stations[next].position < position)
        ++next;
    const BeamStation& start = stations[next - 1];
    const BeamStation& end = stations[next];
    const double span = end.position - start.position;
    const double fraction = span > 0.0 ? std::clamp((position - start.position) / span, 0.0, 1.0) : 0.0;
    BeamStation result = fraction < 1.0 ? start : end;
    result.position = position;
    result.area = interpolate(start.area, end.area, fraction);
    result.inertia1 = interpolate(start.inertia1, end.inertia1, fraction);
    result.inertia2 = interpolate(start.inertia2, end.inertia2, fraction);
    result.productOfInertia = interpolate(start.productOfInertia, end.productOfInertia, fraction);
    result.torsionalConstant = interpolate(start.torsionalConstant, end.torsionalConstant, fraction);
    result.nonStructuralMass = interpolate(start.nonStructuralMass, end.nonStructuralMass, fraction);
    for (std::size_t k = 0; k < 2; ++k)
    {
        result.neutralAxis[k] = interpolate(start.neutralAxis[k], end.neutralAxis[k], fraction);
        result.massCentre[k] = interpolate(start.massCentre[k], end.massCentre[k], fraction);
    }
    return result;
}

bool BeamSection::isUniform() const
{
    const BeamStation& endA = stations[0];
    for (std::size_t s = 1; s < count; ++s)
    {
        const BeamStation& station = stations[s];
        if (station.area != endA.area || station.inertia1 != endA.inertia1 || station.inertia2 != endA.inertia2 ||
            station.productOfInertia != endA.productOfInertia || station.torsionalConstant != endA.torsionalConstant ||
            station.nonStructuralMass != endA.nonStructuralMass || station.neutralAxis != endA.neutralAxis ||
            station.massCentre != endA.massCentre)
            return false;
    }
    return true;
}

bool BeamSection::hasNeutralAxisOffset() const
{
    return std::any_of(stations.begin(), stations.begin() + static_cast<std::ptrdiff_t>(count),
                       [](const BeamStation& station) { return station.hasNeutralAxisOffset(); });
}

bool BeamSection::hasMassCentreOffset() const
{
    return std::any_of(stations.begin(), stations.begin() + static_cast<std::ptrdiff_t>(count),
                       [](const BeamStation& station) { return station.hasMassCentreOffset(); });
}

std::optional<SectionProblem> findSectionProblem(const BeamSection& section)
{
    struct Property
    {
        const char* field;
        double BeamStation::* member;
    };
    static constexpr Property PROPERTIES[] = {{"BarProp.area", &BeamStation::area},
                                              {"BarProp.bendingInertia1", &BeamStation::inertia1},
                                              {"BarProp.bendingInertia2", &BeamStation::inertia2},
                                              {"BarProp.torsionConstant", &BeamStation::torsionalConstant}};
    for (const Property& property : PROPERTIES)
    {
        std::size_t zeros = 0;
        for (std::size_t s = 0; s < section.count; ++s)
        {
            const double value = section.stations[s].*property.member;
            if (value < 0.0)
                return SectionProblem{property.field,
                                      base::concatenate({"expected a value of ", property.field, " at least 0, found ",
                                                         std::to_string(value), " at ", stationName(section, s),
                                                         "; correct the property"})};
            zeros += value == 0.0 ? 1 : 0;
        }
        if (zeros != 0 && zeros != section.count)
            return SectionProblem{
                property.field,
                base::concatenate({"expected ", property.field,
                                   " either zero at every station or positive at every station, found it zero at ",
                                   std::to_string(zeros), " of the ", std::to_string(section.count),
                                   " stations; a section tapering to zero stiffness has no finite flexibility: give "
                                   "a small positive value"})};
    }
    for (std::size_t s = 0; s < section.count; ++s)
    {
        const BeamStation& station = section.stations[s];
        if (station.productOfInertia == 0.0)
            continue;
        if (station.inertia1 * station.inertia2 <= station.productOfInertia * station.productOfInertia)
            return SectionProblem{
                "BarProp.productInertia12",
                base::concatenate({"expected BarProp.bendingInertia1·BarProp.bendingInertia2 > "
                                   "BarProp.productInertia12² (BarProp), found BarProp.bendingInertia1 ",
                                   std::to_string(station.inertia1), ", BarProp.bendingInertia2 ",
                                   std::to_string(station.inertia2), " and BarProp.productInertia12 ",
                                   std::to_string(station.productOfInertia), " at ", stationName(section, s),
                                   "; correct the inertias"})};
    }
    return std::nullopt;
}

} // namespace oos::elements
