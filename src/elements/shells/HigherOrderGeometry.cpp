#include "oos/elements/shells/HigherOrderGeometry.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/shells/HigherOrderShapes.hpp"
#include "oos/elements/shells/MindlinShell.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <span>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

// O_o_Reader gives 0 for a blank edge grid point.
constexpr std::int32_t BLANK_GRID = 0;
constexpr std::string_view QUADRILATERAL_CARD = "ShellQuad8";
constexpr std::string_view TRIANGLE_CARD = "ShellTria6";

// The element frame of the corners: the ShellQuad4 frame or the ShellTria3 frame, and the angle of the
// edge G1→G2 from its x axis (the reference of THETA).
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
        frame.axes = geometry.geometry.axes;
        frame.centre = geometry.geometry.centre;
        frame.edgeAngle = geometry.geometry.edgeAngle;
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

// v in basic from the element frame of `axes`: E v.
[[nodiscard]] base::Vec3 toBasic(const base::Mat3& axes, const base::Vec3& v)
{
    return {axes(0, 0) * v[0] + axes(0, 1) * v[1] + axes(0, 2) * v[2],
            axes(1, 0) * v[0] + axes(1, 1) * v[1] + axes(1, 2) * v[2],
            axes(2, 0) * v[0] + axes(2, 1) * v[1] + axes(2, 2) * v[2]};
}

// One node of one element at a grid point, with the element's own normal there (basic).
struct NodeNormal
{
    std::int32_t gridRow = 0;
    std::size_t slot = 0; // the node's index in the card's table (row · N + node)
    bool isTriangle = false;
    bool isCurved = false; // the element is curved
    base::Vec3 normal;
};

// The own normals of every node of the elements of the batch of `card`, appended to `normals`.
template <typename Shape>
void collectNodeNormals(const SnapshotIndex& index, std::string_view card, bool isTriangle,
                        std::vector<NodeNormal>& normals)
{
    const input::ElementBatch* batch = index.snapshot().findElementBatch(card);
    if (batch == nullptr)
        return;
    const ElementBatchView view(*batch, index);
    const ShellConnectivity connectivity(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        const Resolution<HigherOrderShellGeometry<Shape::NODES>> geometry =
            resolveHigherOrderShellGeometry<Shape>(view, connectivity, row);
        if (!geometry)
            continue; // check() reports it
        for (std::size_t node = 0; node < Shape::NODES; ++node)
            normals.push_back({geometry.value->nodes.rows[node], static_cast<std::size_t>(row) * Shape::NODES + node,
                               isTriangle, geometry.value->isCurved,
                               toBasic(geometry.value->axes, geometry.value->ownDirector[node])});
    }
}

// The smooth groups of the normals of one grid point (HigherOrderShellDirectors): `group[i]` the group of normal i,
// the connected sets of the relation "angle in direction below the sharp-edge angle" (a small union-find).
[[nodiscard]] std::vector<std::size_t> smoothGroups(std::span<const NodeNormal> normals)
{
    const double sameDirectionCosine = std::cos(HIGHER_ORDER_SHELL_SHARP_EDGE_DEGREES * std::numbers::pi / 180.0);
    std::vector<std::size_t> parent(normals.size()); // bounded by the elements at one grid point
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    const auto root = [&](std::size_t i)
    {
        while (parent[i] != i)
            i = parent[i];
        return i;
    };
    for (std::size_t i = 0; i < normals.size(); ++i)
        for (std::size_t j = i + 1; j < normals.size(); ++j)
            if (std::abs(base::dot(normals[i].normal, normals[j].normal)) >= sameDirectionCosine)
                parent[root(j)] = root(i);
    std::vector<std::size_t> group(normals.size());
    for (std::size_t i = 0; i < normals.size(); ++i)
        group[i] = root(i);
    return group;
}

// The shared director of the members of one smooth group, or nothing when its elements are all flat or its normals
// agree to round-off.
[[nodiscard]] std::optional<base::Vec3> groupDirector(std::span<const NodeNormal> normals,
                                                      std::span<const std::size_t> members)
{
    const base::Vec3& first = normals[members[0]].normal;
    const bool hasCurved =
        std::any_of(members.begin(), members.end(), [&](std::size_t m) { return normals[m].isCurved; });
    if (!hasCurved)
        return std::nullopt;
    bool isSame = true;
    base::Vec3 sum{0.0, 0.0, 0.0};
    for (const std::size_t m : members)
    {
        const base::Vec3& normal = normals[m].normal;
        isSame = isSame && base::norm(base::cross(normal, first)) <= HIGHER_ORDER_SHELL_SAME_NORMAL_SINE;
        sum = sum + (base::dot(normal, first) >= 0.0 ? normal : -normal);
    }
    if (isSame)
        return std::nullopt;
    return base::normalized(sum);
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

HigherOrderShellDirectors HigherOrderShellDirectors::build(const SnapshotIndex& index)
{
    HigherOrderShellDirectors directors;
    std::vector<NodeNormal> normals; // bounded by the nodes of the higher-order shells
    collectNodeNormals<SerendipityQuadShape>(index, QUADRILATERAL_CARD, false, normals);
    collectNodeNormals<QuadraticTriangleShape>(index, TRIANGLE_CARD, true, normals);
    if (const input::ElementBatch* batch = index.snapshot().findElementBatch(QUADRILATERAL_CARD))
        directors.m_quadrilaterals.assign(static_cast<std::size_t>(batch->rowCount()) * SerendipityQuadShape::NODES,
                                          base::Vec3{});
    if (const input::ElementBatch* batch = index.snapshot().findElementBatch(TRIANGLE_CARD))
        directors.m_triangles.assign(static_cast<std::size_t>(batch->rowCount()) * QuadraticTriangleShape::NODES,
                                     base::Vec3{});
    // Grid point by grid point, in a fixed order (the stable sort keeps the card, row and node order): the directors
    // do not depend on the threads.
    std::stable_sort(normals.begin(), normals.end(),
                     [](const NodeNormal& a, const NodeNormal& b) { return a.gridRow < b.gridRow; });
    std::size_t first = 0;
    while (first < normals.size())
    {
        std::size_t last = first + 1;
        while (last < normals.size() && normals[last].gridRow == normals[first].gridRow)
            ++last;
        const std::span<const NodeNormal> atGrid(normals.data() + first, last - first);
        const std::vector<std::size_t> group = smoothGroups(atGrid);
        for (std::size_t g = 0; g < atGrid.size(); ++g)
        {
            if (group[g] != g)
                continue;                     // the group is handled from its root
            std::vector<std::size_t> members; // bounded by the elements at one grid point
            for (std::size_t m = 0; m < atGrid.size(); ++m)
                if (group[m] == g)
                    members.push_back(m);
            if (members.size() < 2)
                continue;
            const std::optional<base::Vec3> shared = groupDirector(atGrid, members);
            if (!shared)
                continue;
            for (const std::size_t m : members)
                (atGrid[m].isTriangle ? directors.m_triangles : directors.m_quadrilaterals)[atGrid[m].slot] = *shared;
        }
        first = last;
    }
    return directors;
}

std::optional<base::Vec3> HigherOrderShellDirectors::sharedDirector(std::string_view card, std::int64_t row,
                                                                    std::size_t node) const
{
    const bool isTriangle = card == TRIANGLE_CARD;
    const std::vector<base::Vec3>& table = isTriangle ? m_triangles : m_quadrilaterals;
    const std::size_t nodes = isTriangle ? QuadraticTriangleShape::NODES : SerendipityQuadShape::NODES;
    const std::size_t slot = static_cast<std::size_t>(row) * nodes + node;
    if (slot >= table.size() || table[slot] == base::Vec3{})
        return std::nullopt;
    return table[slot];
}

const HigherOrderShellDirectors& higherOrderShellDirectors(const SnapshotIndex& index)
{
    return index.computedOnce<HigherOrderShellDirectors>([&] { return HigherOrderShellDirectors::build(index); });
}

template Resolution<HigherOrderShellGeometry<SerendipityQuadShape::NODES>>
resolveHigherOrderShellGeometry<SerendipityQuadShape>(const ElementBatchView&, const ShellConnectivity&, std::int64_t);
template Resolution<HigherOrderShellGeometry<QuadraticTriangleShape::NODES>>
resolveHigherOrderShellGeometry<QuadraticTriangleShape>(const ElementBatchView&, const ShellConnectivity&,
                                                        std::int64_t);

} // namespace oos::elements
