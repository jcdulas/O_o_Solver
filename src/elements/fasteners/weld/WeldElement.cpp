#include "oos/elements/fasteners/weld/WeldElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/bars_beams/BeamFlexibility.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader gives a blank Weld MCID as −1 (the weld system from the geometry).
constexpr std::int32_t GEOMETRIC_SYSTEM = -1;

void setBlock(base::MatN<12>& t, std::size_t blockRow, std::size_t blockColumn, const base::Mat3& block)
{
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            t(3 * blockRow + i, 3 * blockColumn + j) = block(i, j);
}

} // namespace

WeldElement::Columns::Columns(const ElementBatchView& view)
    : pwid(view.batch().propertyIds()),
      type(view.columns().texts("kind")),
      gs(view.columns().values<std::int32_t>("spotNode")),
      ga(view.columns().values<std::int32_t>("nodeA")),
      gb(view.columns().values<std::int32_t>("nodeB")),
      matFrameId(view.columns().values<std::int32_t>("matFrameId")),
      ida(view.columns().values<std::int32_t>("patchIdA")),
      idb(view.columns().values<std::int32_t>("patchIdB")),
      weldProp(view.propertyTable("WeldProp")),
      materials(view)
{
    if (view.columns().hasColumn("patchShapes"))
        sptyp = view.columns().texts("patchShapes", base::NullPolicy::Allow);
    if (weldProp == nullptr)
        return;
    const base::ContractReader& reader = weldProp->reader();
    weldPropIds = weldProp->ids();
    matId = reader.values<std::int32_t>("matId");
    d = reader.values<double>("diameter");
    ldmin = reader.values<double>("minLengthRatio");
    ldmax = reader.values<double>("maxLengthRatio");
}

Resolution<WeldElement::Weld> WeldElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                           std::int64_t row)
{
    using Result = Resolution<Weld>;
    const auto r = static_cast<std::size_t>(row);
    Weld weld;
    // The WeldProp by its ID (O_o_Reader gives no propIdRow for Weld: its property column is PWID).
    const std::int32_t propId = columns.pwid[r];
    const auto found = std::lower_bound(columns.weldPropIds.begin(), columns.weldPropIds.end(), propId);
    if (found == columns.weldPropIds.end() || *found != propId)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_PROPERTY, "propId",
            base::concatenate({"expected the ID of a WeldProp, found ", std::to_string(propId),
                               ", which the model does not define; define the property or correct the card"})));
    const auto p = static_cast<std::size_t>(found - columns.weldPropIds.begin());
    if (columns.matFrameId[r] != GEOMETRIC_SYSTEM)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "matFrameId",
            base::concatenate({"expected Weld.matFrameId blank (the weld system from the geometry), found ",
                               std::to_string(columns.matFrameId[r]),
                               "; a weld system from a coordinate system is not supported: leave "
                               "Weld.matFrameId blank"})));
    Resolution<IsotropicMaterial> material =
        resolveIsotropicMaterial(view, columns.materials, row, columns.matId[p], "WeldProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    weld.material = *material.value;

    // AlignedNodes connects GA and GB directly; a point-to-patch connection (ElemIdPatch without SHIDB, NodeIdPatch
    // with the one face shape QuadPatch or TriaPatch) connects GS directly, as side B, to the element or face side A
    // pierces.
    const std::string_view type = columns.type[row];
    const std::string_view sptyp = columns.sptyp ? (*columns.sptyp)[row] : std::string_view();
    std::array<std::int32_t, 2> directGrids{0, 0};
    if (type == "AlignedNodes")
        directGrids = {columns.ga[r], columns.gb[r]};
    else if ((type == "ElemIdPatch" && columns.idb[r] == 0) ||
             (type == "NodeIdPatch" && (sptyp == "QuadPatch" || sptyp == "TriaPatch")))
        directGrids = {0, columns.gs[r]};
    Resolution<FastenerConnection> connection = resolveFastenerConnection(view, row, directGrids);
    if (!connection)
        return Result::failure(std::move(connection.problem));
    weld.connection = std::move(*connection.value);

    // The beam: a circle of diameter D, Cowper's shear coefficient, the length of weldBeamLength (WeldElement.hpp).
    weld.diameter = columns.d[p];
    const double diameter = weld.diameter;
    if (diameter <= 0.0)
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, "propId",
                           base::concatenate({"WeldProp ", std::to_string(propId),
                                              ": expected a diameter WeldProp.diameter above 0, found ",
                                              std::to_string(diameter), "; correct WeldProp.diameter"})));
    const double poissonRatio = weld.material.shearModulus > 0.0
                                    ? weld.material.youngModulus / (2.0 * weld.material.shearModulus) - 1.0
                                    : weld.material.poissonRatio;
    weld.shearCoefficient = weldShearCoefficient(poissonRatio);
    BeamStation station = weldStation(diameter);
    weld.section.count = 2;
    weld.section.stations[0] = station;
    station.position = 1.0;
    weld.section.stations[1] = station;
    weld.section.shearFactor1 = weld.shearCoefficient;
    weld.section.shearFactor2 = weld.shearCoefficient;
    const base::Vec3 pointA = weld.connection.pointA;
    const base::Vec3 pointB = weld.connection.pointB;
    weld.distance = base::norm(pointB - pointA);
    const std::optional<double> thicknessA = weldSideThickness(view.snapshot(), type, columns.ida[r]);
    const std::optional<double> thicknessB = weldSideThickness(view.snapshot(), type, columns.idb[r]);
    weld.beamLength =
        weldBeamLength(weld.distance, thicknessA, thicknessB, diameter, columns.ldmin[p], columns.ldmax[p]);
    weld.beamStiffness = BeamFlexibility(weld.section, weld.material, weld.beamLength).stiffness();

    // d = T·w: the end DOFs of the beam in the weld system from the motions of A and B, through rigid arms from A and
    // B to the beam ends A' and B' (zero when the beam length is |AB|).
    const base::Mat3 axes = fastenerAxes(pointA, pointB);
    const base::Vec3 xAxis{axes(0, 0), axes(1, 0), axes(2, 0)};
    const base::Vec3 middle = 0.5 * (pointA + pointB);
    const base::Vec3 endA = middle - (0.5 * weld.beamLength) * xAxis;
    const base::Vec3 endB = middle + (0.5 * weld.beamLength) * xAxis;
    const base::Mat3 toWeld = axes.transposed();
    setBlock(weld.toBeamEnds, 0, 0, toWeld);
    setBlock(weld.toBeamEnds, 0, 1, -1.0 * (toWeld * crossProductMatrix(endA - pointA))); // θ_A × (A' − A)
    setBlock(weld.toBeamEnds, 1, 1, toWeld);
    setBlock(weld.toBeamEnds, 2, 2, toWeld);
    setBlock(weld.toBeamEnds, 2, 3, -1.0 * (toWeld * crossProductMatrix(endB - pointB)));
    setBlock(weld.toBeamEnds, 3, 3, toWeld);
    return Result::success(std::move(weld));
}

BeamStation WeldElement::weldStation(double diameter)
{
    const double area = std::numbers::pi * diameter * diameter / 4.0;
    const double inertia = std::numbers::pi * std::pow(diameter, 4) / 64.0;
    BeamStation station;
    station.area = area;
    station.inertia1 = inertia;
    station.inertia2 = inertia;
    station.torsionalConstant = 2.0 * inertia;
    const double radius = diameter / 2.0;
    station.recoveryPoints = {{{radius, 0.0}, {0.0, radius}, {-radius, 0.0}, {0.0, -radius}}};
    return station;
}

double WeldElement::diameterOf(const ElementBatchView& view, const Columns& columns, std::int64_t row)
{
    const std::int32_t propId = columns.pwid[static_cast<std::size_t>(row)];
    const auto found = std::lower_bound(columns.weldPropIds.begin(), columns.weldPropIds.end(), propId);
    if (found == columns.weldPropIds.end() || *found != propId)
        throw std::invalid_argument(base::concatenate(
            {"WeldElement: element ", std::to_string(view.elementId(row)), " names WeldProp ",
             std::to_string(propId),
             ", which the model does not define; check() refuses it: check the batch before recovering it"}));
    return columns.d[static_cast<std::size_t>(found - columns.weldPropIds.begin())];
}

void WeldElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
