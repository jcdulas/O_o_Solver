#include "oos/elements/shells/HigherOrderGeometry.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/shells/HigherOrderShapes.hpp"
#include "oos/elements/shells/MindlinShell.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

// O_o_Reader gives 0 for a blank edge grid point.
constexpr std::int32_t BLANK_GRID = 0;

// The element frame of the corners, its x axis along the edge G1→G2 (projected on the mean plane of the corners of
// a quadrilateral, whose normal is along the cross product of its diagonals): the reference of THETA and the x axis
// of the results, as for the ShellQuad4 and the ShellTria3. `edgeAngle` is the angle of the edge G1→G2 from the x
// axis: zero.
struct CornerFrame
{
    bool isDegenerate = false;
    base::Mat3 axes = base::Mat3::identity();
    base::Vec3 centre;
    double edgeAngle = 0.0;
};

template <std::size_t C>
[[nodiscard]] CornerFrame cornerFrameOf(const std::array<base::Vec3, C>& corners)
{
    CornerFrame frame;
    if constexpr (C == 4)
    {
        const ShellQuad4DrillingGeometryResult geometry = buildShellQuad4DrillingGeometry(corners);
        frame.isDegenerate = geometry.problem == QuadGeometryProblem::DegenerateNormal;
        frame.centre = geometry.geometry.centre;
        // The frame of the bisector of the diagonals, turned about its normal by the angle of the edge G1→G2.
        const base::Mat3& bisector = geometry.geometry.axes;
        const double cosine = std::cos(geometry.geometry.edgeAngle);
        const double sine = std::sin(geometry.geometry.edgeAngle);
        for (std::size_t r = 0; r < 3; ++r)
        {
            frame.axes(r, 0) = cosine * bisector(r, 0) + sine * bisector(r, 1);
            frame.axes(r, 1) = -sine * bisector(r, 0) + cosine * bisector(r, 1);
            frame.axes(r, 2) = bisector(r, 2);
        }
    }
    else
    {
        const ShellTria3DrillingGeometryResult geometry = buildShellTria3DrillingGeometry(corners);
        frame.isDegenerate = geometry.isDegenerate;
        frame.axes = geometry.geometry.axes;
        frame.centre = geometry.geometry.centre;
    }
    return frame;
}

// The problem of a curved element whose surface turns too far from the plane of its corners, or nothing: its normal
// at a node or an integration point more than 45° from the element normal (HIGHER_ORDER_SHELL_MIN_NORMAL_COSINE).
template <typename Shape>
[[nodiscard]] std::optional<std::string> curvatureProblem(const ElementBatchView& view,
                                                          const std::array<base::Vec3, Shape::NODES>& position)
{
    const double cosine = MindlinShell<Shape>::smallestNormalCosine(position);
    if (cosine >= HIGHER_ORDER_SHELL_MIN_NORMAL_COSINE)
        return std::nullopt;
    const double degrees = std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / std::numbers::pi;
    return base::concatenate({"expected a curved ", view.card(),
                              " whose surface normal stays within 45 degrees of the normal of its "
                              "corners' plane, found it turned by ",
                              std::to_string(static_cast<int>(std::lround(degrees))),
                              " degrees; one quadratic element cannot follow so curved a surface: refine the mesh"});
}

} // namespace

template <typename Shape>
Resolution<HigherOrderShellGeometry<Shape::NODES>> resolveHigherOrderShellGeometry(
    const ElementBatchView& view, const ShellConnectivity& connectivity, std::int64_t row)
{
    constexpr std::size_t N = Shape::NODES;
    constexpr std::size_t C = Shape::CORNERS;
    using Result = Resolution<HigherOrderShellGeometry<N>>;
    const std::span<const std::int32_t> ids = connectivity.ids[row];
    for (std::size_t i = C; i < N && i < ids.size(); ++i)
        if (ids[i] == BLANK_GRID)
            return Result::failure(elementProblem(
                view, row, problem_code::UNSUPPORTED_FIELD, std::string("node").append(std::to_string(i + 1)),
                base::concatenate({"expected every edge grid point of the ", view.card(), ", found ", view.card(),
                                   ".node", std::to_string(i + 1),
                                   " blank; an element without its edge nodes is not supported: give the edge nodes "
                                   "or use a ShellQuad4 or ShellTria3"})));
    Resolution<ShellCorners<N>> nodes = resolveShellCorners<N>(view, connectivity, row);
    if (!nodes)
        return Result::failure(std::move(nodes.problem));
    std::array<base::Vec3, C> corners{};
    for (std::size_t i = 0; i < C; ++i)
        corners[i] = nodes.value->positions[i];
    const CornerFrame frame = cornerFrameOf<C>(corners);
    if (frame.isDegenerate)
        return Result::failure(elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
                                              "expected corner grid points that span a plane, found them on a line "
                                              "or coincident; correct their positions or their order"));
    HigherOrderShellGeometry<N> geometry;
    geometry.nodes = *nodes.value;
    geometry.axes = frame.axes;
    geometry.edgeAngle = frame.edgeAngle;
    std::array<double, N> x{};
    std::array<double, N> y{};
    std::array<double, N> z{};
    double size = 0.0;
    double largestHeight = 0.0;
    for (std::size_t i = 0; i < N; ++i)
    {
        const base::Vec3 d = nodes.value->positions[i] - frame.centre;
        x[i] = d[0] * frame.axes(0, 0) + d[1] * frame.axes(1, 0) + d[2] * frame.axes(2, 0);
        y[i] = d[0] * frame.axes(0, 1) + d[1] * frame.axes(1, 1) + d[2] * frame.axes(2, 1);
        z[i] = d[0] * frame.axes(0, 2) + d[1] * frame.axes(1, 2) + d[2] * frame.axes(2, 2);
        largestHeight = std::max(largestHeight, std::abs(z[i]));
        size = std::max(size, std::hypot(x[i], y[i]));
    }
    if (!MindlinShell<Shape>::isValidShape(x, y))
        return Result::failure(elementProblem(
            view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
            base::concatenate({"expected an element whose edge nodes lie near the middle of their edges and whose "
                               "corners turn counter-clockwise, found a map with a Jacobian that is not positive"
                               "; move the edge nodes or reorder the grid points"})));
    // A flat element keeps its nodes projected on the plane of its corners and takes n as its own normal at every
    // node: the Mindlin plate exactly (MindlinShell); a curved one its nodes as given and the normals of its surface.
    geometry.isCurved = largestHeight > HIGHER_ORDER_SHELL_FLATNESS * size;
    for (std::size_t i = 0; i < N; ++i)
        geometry.position[i] = base::Vec3{x[i], y[i], geometry.isCurved ? z[i] : 0.0};
    geometry.ownDirector.fill(base::Vec3{0.0, 0.0, 1.0});
    if (geometry.isCurved)
    {
        if (std::optional<std::string> problem = curvatureProblem<Shape>(view, geometry.position))
            return Result::failure(
                elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, "node1", std::move(*problem)));
        geometry.ownDirector = MindlinShell<Shape>::nodalDirectors(geometry.position);
    }
    return Result::success(std::move(geometry));
}

template Resolution<HigherOrderShellGeometry<SerendipityQuadShape::NODES>>
resolveHigherOrderShellGeometry<SerendipityQuadShape>(const ElementBatchView&, const ShellConnectivity&, std::int64_t);
template Resolution<HigherOrderShellGeometry<QuadraticTriangleShape::NODES>>
resolveHigherOrderShellGeometry<QuadraticTriangleShape>(const ElementBatchView&, const ShellConnectivity&,
                                                        std::int64_t);

} // namespace oos::elements
