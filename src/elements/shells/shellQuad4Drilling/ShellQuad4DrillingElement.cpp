#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingBending.hpp"

#include <cmath>
#include <numbers>
#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader gives MCID −1 when field 7 holds THETA (ShellQuad4Card::NO_MAT_FRAME_ID).
constexpr std::int32_t NO_MAT_FRAME_ID = -1;

// An MCID whose x axis projects on the element plane to less than this (relative to a unit axis) has no direction
// there.
constexpr double MATERIAL_FRAME_MIN_PROJECTION = 1.0e-8;

} // namespace

ShellQuad4DrillingElement::Columns::Columns(const ElementBatchView& view)
    : connectivity(view),
      sections(view),
      theta(view.columns().values<double>("materialAngle")),
      matFrameId(view.columns().values<std::int32_t>("matFrameId")),
      offset(view.columns().values<double>("offset")),
      thicknessFlag(view.columns().values<std::int32_t>("thicknessFlag")),
      cornerThickness(view.columns().fixedLists<double>("cornerThickness"))
{
}

Resolution<ShellQuad4DrillingElement::Data> ShellQuad4DrillingElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                             std::int64_t row) const
{
    using Result = Resolution<Data>;
    const auto r = static_cast<std::size_t>(row);
    Resolution<ShellCorners<4>> corners = resolveShellCorners<4>(view, columns.connectivity, row);
    if (!corners)
        return Result::failure(std::move(corners.problem));
    Resolution<ShellSection> section = resolveShellSection(view, columns.sections, row);
    if (!section)
        return Result::failure(std::move(section.problem));

    Data data;
    data.section = *section.value;
    data.gridIds = corners.value->ids;
    const ShellQuad4DrillingGeometryResult geometry = buildShellQuad4DrillingGeometry(corners.value->positions);
    switch (geometry.problem)
    {
    case QuadGeometryProblem::None:
        break;
    case QuadGeometryProblem::DegenerateNormal:
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
                           "expected a quadrilateral, found diagonals ShellQuad4.node1-ShellQuad4.node3 and "
                           "ShellQuad4.node2-ShellQuad4.node4 parallel or of zero length; the four grid "
                           "points do not span a plane: correct their positions or the order of "
                           "ShellQuad4.node1-ShellQuad4.node4"));
    case QuadGeometryProblem::FoldedCorner:
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY,
                           std::string("node").append(std::to_string(geometry.corner + 1)),
                           base::concatenate({"expected a convex quadrilateral, found the corner ShellQuad4.node",
                                              std::to_string(geometry.corner + 1),
                                              " re-entrant or collapsed (Jacobian not positive there); reorder "
                                              "ShellQuad4.node1-ShellQuad4.node4 around the "
                                              "element, or use a ShellTria3 for a triangle"})));
    }
    // A warped element is accepted: it is formed flat in its mean plane with the rigid offsets of ShellQuad4DrillingWarpOffsets
    // (ShellQuad4DrillingPlate). A membrane-bending coupling is not: the drilling membrane has none.
    data.geometry = geometry.geometry;
    if (data.section.hasCoupling)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "ShellProp.couplingMatId",
            base::concatenate(
                {"expected no membrane-bending coupling on the ", view.card(),
                 " (ShellProp.couplingMatId blank, a laminate "
                 "with B = 0), found one; the coupling is supported on the ShellQuad4 and ShellTria3 only"})));
    if (data.section.hasBending && !ShellQuad4DrillingBending::hasEdgeShearFrame(data.geometry))
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
                           "expected a quadrilateral whose edges ShellQuad4.node1-ShellQuad4.node2 and "
                           "ShellQuad4.node3-ShellQuad4.node4 run across the element y axis and "
                           "ShellQuad4.node2-ShellQuad4.node3, ShellQuad4.node4-ShellQuad4.node1 "
                           "across its x axis, found an edge along the other axis (the edge shear cannot be formed); "
                           "the element is too distorted: remesh it"));

    Resolution<std::array<double, 4>> thickness =
        resolveCornerThicknesses<4>(view, row, data.section, columns.thicknessFlag[r], columns.cornerThickness[row]);
    if (!thickness)
        return Result::failure(std::move(thickness.problem));
    data.thickness = *thickness.value;

    // The material axis.
    if (columns.matFrameId[r] == NO_MAT_FRAME_ID)
    {
        data.materialAngle = data.geometry.edgeAngle + columns.theta[r] * std::numbers::pi / 180.0;
    }
    else
    {
        Resolution<input::CoordinateFrame> system =
            resolveRectangularFrame(view, row, columns.matFrameId[r], "matFrameId");
        if (!system)
            return Result::failure(std::move(system.problem));
        const base::Vec3 axis{system.value->axes(0, 0), system.value->axes(1, 0), system.value->axes(2, 0)};
        const base::Mat3& e = data.geometry.axes;
        const double alongX = axis[0] * e(0, 0) + axis[1] * e(1, 0) + axis[2] * e(2, 0);
        const double alongY = axis[0] * e(0, 1) + axis[1] * e(1, 1) + axis[2] * e(2, 1);
        if (std::hypot(alongX, alongY) <= MATERIAL_FRAME_MIN_PROJECTION)
            return Result::failure(elementProblem(
                view, row, problem_code::DEGENERATE_GEOMETRY, "matFrameId",
                base::concatenate({"expected the x axis of coordinate system ", std::to_string(columns.matFrameId[r]),
                                   " to cross the element plane at an angle, found it normal to the element; give "
                                   "another ShellQuad4.matFrameId or a ShellQuad4.materialAngle"})));
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
        ShellNodeFrames<4>::build(corners.value->rows, corners.value->cdFrames, data.geometry.axes, columns.offset[r]);
    return Result::success(data);
}

void ShellQuad4DrillingElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    checkRows(view, {0, view.rowCount()}, sink);
}

void ShellQuad4DrillingElement::checkRows(const ElementBatchView& view, ElementRange range, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        Resolution<Data> resolution = resolve(view, columns, row);
        if (!resolution)
        {
            sink.report(std::move(resolution.problem));
            continue;
        }
        const double warp = resolution.value->geometry.warpRatio;
        if (warp > QUAD_WARP_REPORT_RATIO)
        {
            base::Diagnostic warning = elementProblem(
                view, row, "warped-element", "node4",
                base::concatenate({"expected a warp (largest height off the mean plane over the mean diagonal) of "
                                   "at most 0.05, found ",
                                   std::to_string(warp),
                                   "; the element is accepted, but its stiffness is less accurate: flatten it or "
                                   "split it into flatter elements"}));
            warning.severity = base::Severity::Warning;
            sink.report(std::move(warning));
        }
    }
}

} // namespace oos::elements
