#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"

#include <cmath>
#include <numbers>
#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader gives MCID −1 when field 7 holds THETA.
constexpr std::int32_t NO_MAT_FRAME_ID = -1;

// An MCID whose x axis projects on the element plane to less than this has no direction there.
constexpr double MATERIAL_FRAME_MIN_PROJECTION = 1.0e-8;

} // namespace

ShellTria3DrillingElement::Columns::Columns(const ElementBatchView& view)
    : connectivity(view),
      sections(view),
      theta(view.columns().values<double>("materialAngle")),
      matFrameId(view.columns().values<std::int32_t>("matFrameId")),
      offset(view.columns().values<double>("offset")),
      thicknessFlag(view.columns().values<std::int32_t>("thicknessFlag")),
      cornerThickness(view.columns().fixedLists<double>("cornerThickness"))
{
}

Resolution<ShellTria3DrillingElement::Data> ShellTria3DrillingElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                             std::int64_t row) const
{
    using Result = Resolution<Data>;
    const auto r = static_cast<std::size_t>(row);
    Resolution<ShellCorners<3>> corners = resolveShellCorners<3>(view, columns.connectivity, row);
    if (!corners)
        return Result::failure(std::move(corners.problem));
    Resolution<ShellSection> section = resolveShellSection(view, columns.sections, row);
    if (!section)
        return Result::failure(std::move(section.problem));

    Data data;
    data.section = *section.value;
    data.gridIds = corners.value->ids;
    const ShellTria3DrillingGeometryResult geometry = buildShellTria3DrillingGeometry(corners.value->positions);
    if (geometry.isDegenerate)
        return Result::failure(elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node3",
                                              "expected a triangle, found ShellTria3.node1, ShellTria3.node2 and "
                                              "ShellTria3.node3 on a line (or coincident); correct their positions"));
    data.geometry = geometry.geometry;
    // A membrane-bending coupling is refused: the drilling membrane has none.
    if (data.section.hasCoupling)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "ShellProp.couplingMatId",
            base::concatenate(
                {"expected no membrane-bending coupling on the ", view.card(),
                 " (ShellProp.couplingMatId blank, a laminate "
                 "with B = 0), found one; the coupling is supported on the ShellQuad4 and ShellTria3 only"})));

    Resolution<std::array<double, 3>> thickness =
        resolveCornerThicknesses<3>(view, row, data.section, columns.thicknessFlag[r], columns.cornerThickness[row]);
    if (!thickness)
        return Result::failure(std::move(thickness.problem));
    data.thickness = *thickness.value;

    // The material axis, with the triangle's x axis along G1→G2: THETA from that edge, or the projection of MCID's x.
    if (columns.matFrameId[r] == NO_MAT_FRAME_ID)
    {
        data.materialAngle = columns.theta[r] * std::numbers::pi / 180.0;
    }
    else
    {
        Resolution<input::CoordinateFrame> system =
            resolveRectangularFrame(view, row, columns.matFrameId[r], "matFrameId");
        if (!system)
            return Result::failure(std::move(system.problem));
        const base::Mat3& e = data.geometry.axes;
        const base::Mat3& axes = system.value->axes;
        const double alongX = axes(0, 0) * e(0, 0) + axes(1, 0) * e(1, 0) + axes(2, 0) * e(2, 0);
        const double alongY = axes(0, 0) * e(0, 1) + axes(1, 0) * e(1, 1) + axes(2, 0) * e(2, 1);
        if (std::hypot(alongX, alongY) <= MATERIAL_FRAME_MIN_PROJECTION)
            return Result::failure(elementProblem(
                view, row, problem_code::DEGENERATE_GEOMETRY, "matFrameId",
                base::concatenate({"expected the x axis of coordinate system ", std::to_string(columns.matFrameId[r]),
                                   " to cross the element plane at an angle, found it normal to the element; give "
                                   "another ShellTria3.matFrameId or a ShellTria3.materialAngle"})));
        data.materialAngle = std::atan2(alongY, alongX);
    }
    // The section in the element frame: the material matrices turned by the material angle.
    data.section = data.section.rotatedToElementFrame(data.materialAngle);
    if (std::optional<std::string> problem = offsetSectionProblem(data.section, columns.offset[r]))
        return Result::failure(elementProblem(view, row, problem_code::INVALID_FIELD, "offset", std::move(*problem)));
    if (std::optional<std::string> problem = offsetDefinitionProblem(view.snapshot(), columns.offset[r]))
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, "offset", std::move(*problem)));
    data.frames =
        ShellNodeFrames<3>::build(corners.value->rows, corners.value->cdFrames, data.geometry.axes, columns.offset[r]);
    return Result::success(data);
}

void ShellTria3DrillingElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    checkRows(view, {0, view.rowCount()}, sink);
}

void ShellTria3DrillingElement::checkRows(const ElementBatchView& view, ElementRange range, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        auto resolution = resolve(view, columns, row);
        if (!resolution)
            sink.report(std::move(resolution.problem));
    }
}

} // namespace oos::elements
