#include "BeamSizing.hpp"

#include "oos/elements/core/PropertyLookup.hpp"

#include <algorithm>
#include <numeric>

namespace oos::elements::beam_sizing
{

Direction zeroDirection(const BeamData& beam)
{
    Direction result;
    result.section.count = beam.section.count;
    for (std::size_t k = 0; k < beam.section.count; ++k)
        result.section.stations[k].position = beam.section.stations[k].position;
    return result;
}

namespace
{

bool sectionComponent(BeamStation& station, std::string_view name)
{
    if (name == "area") station.area = 1.0;
    else if (name == "bendingInertia1") station.inertia1 = 1.0;
    else if (name == "bendingInertia2") station.inertia2 = 1.0;
    else if (name == "productInertia12") station.productOfInertia = 1.0;
    else if (name == "torsionConstant") station.torsionalConstant = 1.0;
    else if (name == "nonStructMass") station.nonStructuralMass = 1.0;
    else return false;
    return true;
}

bool hasRegularSection(const BeamData& beam, const Direction& direction)
{
    for (std::size_t s = 0; s < beam.section.count; ++s)
    {
        const auto& value = beam.section.stations[s];
        const auto& rate = direction.section.stations[s];
        if ((rate.area != 0.0 && value.area <= 0.0) ||
            (rate.inertia1 != 0.0 && value.inertia1 <= 0.0) ||
            (rate.inertia2 != 0.0 && value.inertia2 <= 0.0) ||
            (rate.torsionalConstant != 0.0 && value.torsionalConstant <= 0.0))
            return false;
        if (rate.productOfInertia != 0.0 && (value.inertia1 <= 0.0 || value.inertia2 <= 0.0))
            return false;
    }
    return true;
}

bool shearDirection(const BeamData& beam, const DesignedField& field, Direction& into)
{
    if (field.item < 0 || field.item > 1)
        return false;
    const double value = field.item == 0 ? beam.section.shearFactor1 : beam.section.shearFactor2;
    if (value == 0.0)
        return false; // The zero/infinite-shear branch has no two-sided derivative at its boundary.
    if (value > 0.0)
        (field.item == 0 ? into.section.shearFactor1 : into.section.shearFactor2) = 1.0;
    return true;
}

bool barDirection(const BeamData& beam, const DesignedField& field, Direction& into)
{
    if (field.column == "shearFactor")
        return shearDirection(beam, field, into);
    if (field.column.starts_with("stressPoint") && field.column.size() == 12 && field.item >= 0 && field.item < 2)
    {
        const char point = field.column.back();
        if (point < 'C' || point > 'F')
            return false;
        for (std::size_t s = 0; s < into.section.count; ++s)
            into.section.stations[s].recoveryPoints[static_cast<std::size_t>(point - 'C')]
                                                  [static_cast<std::size_t>(field.item)] = 1.0;
        return true;
    }
    bool known = false;
    for (std::size_t s = 0; s < into.section.count; ++s)
        known = sectionComponent(into.section.stations[s], field.column);
    return known;
}

bool beamDirection(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                    const DesignedField& field, Direction& into)
{
    if (field.column == "warpingCoefficient")
        return true; // This formulation has no warping DOFs.
    if (field.column == "shearFactor")
        return shearDirection(beam, field, into);
    if ((field.column == "neutralAxis" || field.column == "nsmCenter") && field.item >= 0 && field.item < 4)
    {
        const auto component = static_cast<std::size_t>(field.item % 2);
        for (std::size_t s = 0; s < into.section.count; ++s)
        {
            auto& rate = into.section.stations[s];
            const double value = field.item < 2 ? 1.0 - rate.position : rate.position;
            if (field.column == "nsmCenter") rate.massCentre[component] = value;
            else
            {
                rate.neutralAxis[component] = value;
                for (auto& point : rate.recoveryPoints)
                    point[component] = -value; // Card points are measured from the shear centre.
            }
        }
        return true;
    }
    if ((field.column != "stationSection" && field.column != "stationStress") || field.item < 0)
        return false;
    const PropertyRows properties(view, "BeamProp");
    const auto property = resolveProperty(view, properties, row);
    if (!property)
        return false;
    const auto& table = property.value->table->reader();
    const auto positions = table.lists<double>("stationRatio")[property.value->row];
    const auto outputs = table.lists<std::uint8_t>("stationOutput")[property.value->row];
    const std::size_t width = field.column == "stationSection" ? 6 : 8;
    const auto source = static_cast<std::size_t>(field.item) / width;
    const auto component = static_cast<std::size_t>(field.item) % width;
    if (source >= positions.size() || positions.size() > BEAM_MAX_STATIONS || outputs.size() != positions.size())
        return false;
    std::vector<std::size_t> order(positions.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return positions[a] < positions[b]; });
    const auto found = std::find(order.begin(), order.end(), source);
    const auto selected = static_cast<std::size_t>(found - order.begin());
    static constexpr std::array<std::string_view, 6> names{
        "area", "bendingInertia1", "bendingInertia2", "productInertia12", "torsionConstant", "nonStructMass"};
    if (field.column == "stationSection")
    {
        (void)sectionComponent(into.section.stations[selected], names[component]);
        if (positions.size() == 1)
            (void)sectionComponent(into.section.stations[1], names[component]);
        return true;
    }
    constexpr std::uint8_t YES = 1;
    constexpr std::uint8_t NO = 0;
    const bool takesOwnPoints = outputs[source] == YES || (selected == 0 && outputs[source] == NO);
    if (takesOwnPoints)
    {
        into.section.stations[selected].recoveryPoints[component / 2][component % 2] = 1.0;
        if (selected == 0)
            for (std::size_t s = 1; s < into.section.count; ++s)
                if (positions.size() == 1 || outputs[order[s]] != YES)
                    into.section.stations[s].recoveryPoints[component / 2][component % 2] = 1.0;
    }
    return true;
}

} // namespace

std::optional<Direction> directionOf(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                                     const DesignedField& field)
{
    Direction result = zeroDirection(beam);
    if (field.card == "IsoMat")
    {
        if (field.column == "youngsModulus") result.material.youngModulus = 1.0;
        else if (field.column == "shearModulus") result.material.shearModulus = 1.0;
        else if (field.column == "density") result.material.density = 1.0;
        else if (field.column == "thermalExpansion") result.material.expansion = 1.0;
        else if (field.column == "referenceTemperature") result.material.referenceTemperature = 1.0;
        else if (field.column != "poissonRatio" && field.column != "structuralDamping" &&
                 field.column != "tensionAllowable" && field.column != "compressionAllowable" &&
                 field.column != "shearAllowable")
            return std::nullopt;
        if ((result.material.youngModulus != 0.0 && beam.material.youngModulus <= 0.0) ||
            (result.material.shearModulus != 0.0 && beam.material.shearModulus <= 0.0))
            return std::nullopt;
        return result;
    }
    if (field.card == "BarProp" && barDirection(beam, field, result) && hasRegularSection(beam, result))
        return result;
    if (field.card == "BeamProp" && beamDirection(beam, view, row, field, result) && hasRegularSection(beam, result))
        return result;
    return std::nullopt;
}

Station stationOf(const BeamStation& v, const BeamStation& p, const BeamStation& q)
{
    Station result;
    result.area = {v.area, p.area, q.area};
    result.inertia1 = {v.inertia1, p.inertia1, q.inertia1};
    result.inertia2 = {v.inertia2, p.inertia2, q.inertia2};
    result.productInertia = {v.productOfInertia, p.productOfInertia, q.productOfInertia};
    result.torsion = {v.torsionalConstant, p.torsionalConstant, q.torsionalConstant};
    result.nsm = {v.nonStructuralMass, p.nonStructuralMass, q.nonStructuralMass};
    for (std::size_t k = 0; k < 2; ++k)
    {
        result.neutral[k] = {v.neutralAxis[k], p.neutralAxis[k], q.neutralAxis[k]};
        result.centre[k] = {v.massCentre[k], p.massCentre[k], q.massCentre[k]};
        for (std::size_t r = 0; r < 4; ++r)
            result.points[r][k] = {v.recoveryPoints[r][k], p.recoveryPoints[r][k], q.recoveryPoints[r][k]};
    }
    return result;
}

} // namespace oos::elements::beam_sizing
