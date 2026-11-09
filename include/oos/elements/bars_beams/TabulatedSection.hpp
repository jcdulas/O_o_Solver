#pragma once

#include "oos/elements/bars_beams/BeamSection.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/input/tables/SectionTable.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace oos::elements
{

// The section of a BarShapeProp or BeamShapeProp (a section type of the standard section library and its dimensions),
// taken from the sections table of the snapshot, where the section properties computed from the dimensions are given
// per station: area, I1, I2, I12, J, NSM, the shear factors K1, K2 and the recovery points C, D, E, F. The element
// never computes a section from its dimensions: the formulas (and the shear factors, which are not those of the
// thin-walled rules) belong to O_o_Reader. A BeamShapeProp station's neutral axis from its shear centre (the element
// axis of a Beam2 on a BeamShapeProp) is the table's n1, n2 (O_o_Solver's snapshot adds them from O_o_Reader's
// centroid and shear centre of the section); a BarShapeProp's axis is its neutral axis (a BarShapeProp makes no
// difference between the shear centre and the neutral axis).
//
// The columns of the sections table, read once per facet call; `table` is nullptr when the snapshot has no sections
// table.
struct TabulatedSectionColumns
{
    const input::SectionTable* table = nullptr;
    std::span<const std::int32_t> propIds;
    std::span<const double> positions;
    std::span<const double> areas;
    std::span<const double> inertias1;
    std::span<const double> inertias2;
    std::span<const double> productsOfInertia;
    std::span<const double> torsionalConstants;
    std::span<const double> nonStructuralMasses;
    std::span<const double> shearFactors1;
    std::span<const double> shearFactors2;
    std::span<const double> neutralAxes1; // empty: no n1, n2 columns
    std::span<const double> neutralAxes2;
    base::FixedListView<double> recoveryPoints;

    explicit TabulatedSectionColumns(const ElementBatchView& view);
};

// The section of property `propId` (a `propertyCard`, "BarShapeProp" or "BeamShapeProp") for the element of batch row
// `row`, from the sections table: a BarShapeProp has one row (a uniform section, its two stations equal); a
// BeamShapeProp has one row per station, from X/XB 0 to 1 (or one row, a uniform section). `stationPositions` and
// `stationOutputs` are the X/XB and SO of the BeamShapeProp card (empty for a BarShapeProp; O_o_Reader codes SO as 0
// NO, 1 YES): a station whose X/XB the card gives with SO = NO has no output; the ends always have one.
//
// Problems on the field PID: no sections table, no row for the property, a BarShapeProp with
// more than one row, BeamShapeProp stations that do not run from X/XB 0 to 1 or exceed BEAM_MAX_STATIONS.
[[nodiscard]] Resolution<BeamSection> resolveTabulatedSection(const ElementBatchView& view, std::int64_t row,
                                                              const TabulatedSectionColumns& columns,
                                                              std::string_view propertyCard, std::int32_t propId,
                                                              std::span<const double> stationPositions,
                                                              std::span<const std::uint8_t> stationOutputs);

} // namespace oos::elements
