#include "oos/elements/bars_beams/TabulatedSection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader's code of SO = NO on a BeamShapeProp station (BeamStressOutput).
constexpr std::uint8_t STRESS_OUTPUT_NO = 0;

// The SO of the BeamShapeProp station at X/XB `position`: NO only when the card gives that station with SO = NO (the
// card's X/XB and the table's are the same values: the table is computed from the card).
bool hasStationOutput(double position, std::span<const double> stationPositions,
                      std::span<const std::uint8_t> stationOutputs)
{
    for (std::size_t s = 0; s < stationPositions.size() && s < stationOutputs.size(); ++s)
    {
        if (stationPositions[s] == position)
            return stationOutputs[s] != STRESS_OUTPUT_NO;
    }
    return true;
}

} // namespace

TabulatedSectionColumns::TabulatedSectionColumns(const ElementBatchView& view)
{
    if (!view.snapshot().hasTable(input::TableRole::Sections))
        return;
    table = &view.snapshot().sections();
    propIds = table->propIds();
    positions = table->positions();
    areas = table->areas();
    inertias1 = table->inertias1();
    inertias2 = table->inertias2();
    productsOfInertia = table->productsOfInertia();
    torsionalConstants = table->torsionalConstants();
    nonStructuralMasses = table->nonStructuralMasses();
    shearFactors1 = table->shearFactors1();
    shearFactors2 = table->shearFactors2();
    neutralAxes1 = table->neutralAxes1();
    neutralAxes2 = table->neutralAxes2();
    recoveryPoints = table->recoveryPoints();
}

Resolution<BeamSection> resolveTabulatedSection(const ElementBatchView& view, std::int64_t row,
                                                const TabulatedSectionColumns& columns, std::string_view propertyCard,
                                                std::int32_t propId, std::span<const double> stationPositions,
                                                std::span<const std::uint8_t> stationOutputs)
{
    using Result = Resolution<BeamSection>;
    const std::string property = std::string(propertyCard).append(" ").append(std::to_string(propId));
    if (columns.table == nullptr)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "propId",
            base::concatenate({"expected the section properties of ", property,
                               " in the snapshot's sectionsTable, found no such table; O_o_Reader does not give it "
                               "yet: hand the table over to the snapshot, or give "
                               "the section as a BarProp or BeamProp"})));
    const std::optional<input::SectionRows> rows = columns.table->findProperty(propId);
    if (!rows)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_PROPERTY, "propId",
            base::concatenate({"expected the section properties of ", property,
                               " in sectionsTable, found no row for it; the table does not come from the same "
                               "model: take the snapshot again"})));
    const bool isBar = propertyCard == "BarShapeProp";
    if ((isBar && rows->count() != 1) || rows->count() > static_cast<std::int64_t>(BEAM_MAX_STATIONS))
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "propId",
            base::concatenate({"expected ", isBar ? "one row" : "at most 11 rows (end A and ten stations)", " for ",
                               property, " in sectionsTable, found ", std::to_string(rows->count()),
                               "; the table does not follow the sections table contract"})));

    BeamSection section;
    for (std::int64_t r = rows->first; r < rows->last; ++r)
    {
        const auto t = static_cast<std::size_t>(r);
        BeamStation station;
        station.position = columns.positions[t];
        station.area = columns.areas[t];
        station.inertia1 = columns.inertias1[t];
        station.inertia2 = columns.inertias2[t];
        station.productOfInertia = columns.productsOfInertia[t];
        station.torsionalConstant = columns.torsionalConstants[t];
        station.nonStructuralMass = columns.nonStructuralMasses[t];
        const std::span<const double> points = columns.recoveryPoints[r];
        for (std::size_t point = 0; point < 4; ++point)
            station.recoveryPoints[point] = {points[2 * point], points[2 * point + 1]};
        // The table's recovery points are from the neutral axis already; a BarShapeProp keeps its axis there.
        if (!isBar && !columns.neutralAxes1.empty())
        {
            station.neutralAxis = {columns.neutralAxes1[t], columns.neutralAxes2[t]};
            if (!std::isfinite(station.neutralAxis[0]) || !std::isfinite(station.neutralAxis[1]))
                return Result::failure(elementProblem(
                    view, row, problem_code::INVALID_FIELD, "propId",
                    base::concatenate({"expected the shear centre of ", property, " at BeamShapeProp.stationRatio ",
                                       std::to_string(station.position),
                                       " (its neutral axis n1, n2 in sectionsTable), found none: O_o_Reader gives no "
                                       "shear centre for these dimensions; correct the dimensions of the section"})));
        }
        station.hasOutput = hasStationOutput(station.position, stationPositions, stationOutputs);
        section.stations[section.count++] = station;
    }
    if (section.count == 1)
    {
        section.stations[0].position = 0.0;
        BeamStation endB = section.stations[0];
        endB.position = 1.0;
        section.stations[section.count++] = endB;
    }
    // The ends always have their row (as for a BeamProp).
    section.stations[0].hasOutput = true;
    section.stations[section.count - 1].hasOutput = true;
    if (section.stations[0].position != 0.0 || section.stations[section.count - 1].position != 1.0)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "BeamShapeProp.stationRatio",
            base::concatenate(
                {property, ": expected stations from BeamShapeProp.stationRatio 0.0 (end A) to 1.0 (end B), found ",
                 std::to_string(section.stations[0].position), " to ",
                 std::to_string(section.stations[section.count - 1].position), "; correct the stations"})));
    // A BeamShapeProp has one pair of shear factors for the element; the table gives them per station, those of end A
    // rule.
    const auto first = static_cast<std::size_t>(rows->first);
    section.shearFactor1 = columns.shearFactors1[first];
    section.shearFactor2 = columns.shearFactors2[first];
    return Result::success(section);
}

} // namespace oos::elements
