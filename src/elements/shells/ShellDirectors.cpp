#include "oos/elements/shells/ShellDirectors.hpp"

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/shells/HigherOrderGeometry.hpp"
#include "oos/elements/shells/HigherOrderShapes.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"
#include "oos/elements/shells/shellTria3/ShellTria3.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <span>

namespace oos::elements
{

namespace
{

// The own normal of one grid point of one element, in basic.
struct NodeNormal
{
    std::int32_t gridRow = 0;
    std::size_t table = 0; // the card: its place in DIRECTOR_SHELL_CARDS
    std::size_t slot = 0;  // the node's index in the card's table (row · N + node)
    base::Vec3 normal;
};

// The place of `card` in DIRECTOR_SHELL_CARDS (their number for another card).
[[nodiscard]] std::size_t tableOf(std::string_view card)
{
    std::size_t table = 0;
    while (table < DIRECTOR_SHELL_CARDS.size() && DIRECTOR_SHELL_CARDS[table].card != card)
        ++table;
    return table;
}

// v in basic from the element frame of `axes`: E v.
[[nodiscard]] base::Vec3 toBasic(const base::Mat3& axes, const base::Vec3& v)
{
    return {axes(0, 0) * v[0] + axes(0, 1) * v[1] + axes(0, 2) * v[2],
            axes(1, 0) * v[0] + axes(1, 1) * v[1] + axes(1, 2) * v[2],
            axes(2, 0) * v[0] + axes(2, 1) * v[1] + axes(2, 2) * v[2]};
}

// The own normals of the corners of the ShellQuad4 or ShellTria3 of the batch of `Shell`, appended to `normals`; an
// element whose geometry is refused has none (its check reports it).
template <typename Shell>
void collectCornerNormals(const SnapshotIndex& index, std::vector<NodeNormal>& normals)
{
    const input::ElementBatch* batch = index.snapshot().findElementBatch(Shell::CARD);
    if (batch == nullptr)
        return;
    const ElementBatchView view(*batch, index);
    const ShellConnectivity connectivity(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        const Resolution<ShellCorners<Shell::NODES>> corners =
            resolveShellCorners<Shell::NODES>(view, connectivity, row);
        if (!corners)
            continue;
        const auto built = Shell::buildGeometry(corners.value->positions);
        if (!built.problem.empty())
            continue;
        const base::Mat3& axes = built.geometry.axes;
        for (std::size_t corner = 0; corner < Shell::NODES; ++corner)
        {
            const Vector3<double>& n = built.geometry.normal[corner];
            normals.push_back({corners.value->rows[corner], tableOf(Shell::CARD),
                               static_cast<std::size_t>(row) * Shell::NODES + corner,
                               toBasic(axes, base::Vec3{n[0], n[1], n[2]})});
        }
    }
}

// The own normals of every grid point of the ShellQuad8 or ShellTria6 of the batch of `card`, appended to `normals`.
template <typename Shape>
void collectNodeNormals(const SnapshotIndex& index, std::string_view card, std::vector<NodeNormal>& normals)
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
            normals.push_back({geometry.value->nodes.rows[node], tableOf(card),
                               static_cast<std::size_t>(row) * Shape::NODES + node,
                               toBasic(geometry.value->axes, geometry.value->ownDirector[node])});
    }
}

// The smooth groups of the normals of one grid point: `group[i]` the group of normal i, the connected sets of the
// relation "angle in direction of at most the shell normal angle" (a small union-find).
[[nodiscard]] std::vector<std::size_t> smoothGroups(std::span<const NodeNormal> normals, double cosine)
{
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
            if (std::abs(base::dot(normals[i].normal, normals[j].normal)) >= cosine)
                parent[root(j)] = root(i);
    std::vector<std::size_t> group(normals.size());
    for (std::size_t i = 0; i < normals.size(); ++i)
        group[i] = root(i);
    return group;
}

// The normal the members of one smooth group share (on the side of the first member), or nothing when their normals
// agree to round-off or when one of them is farther than the angle from their mean.
[[nodiscard]] std::optional<base::Vec3> groupNormal(std::span<const NodeNormal> normals,
                                                    std::span<const std::size_t> members, double cosine)
{
    const base::Vec3& first = normals[members.front()].normal;
    base::Vec3 sum;
    for (const std::size_t m : members)
        sum = sum + (base::dot(normals[m].normal, first) >= 0.0 ? normals[m].normal : (-1.0) * normals[m].normal);
    const double length = base::norm(sum);
    if (!(length > 0.0))
        return std::nullopt;
    const base::Vec3 mean = sum * (1.0 / length);
    bool isSame = true;
    for (const std::size_t m : members)
    {
        if (std::abs(base::dot(mean, normals[m].normal)) < cosine)
            return std::nullopt;
        isSame = isSame && base::norm(base::cross(mean, normals[m].normal)) <= SAME_SHELL_NORMAL_SINE;
    }
    if (isSame)
        return std::nullopt;
    return mean;
}

// The axes of a coordinate system at the point `position` (basic), columns in basic: its own axes for a rectangular
// system; the radial, tangential and axial directions for a cylindrical one; the radial, polar and azimuthal
// directions for a spherical one. On the axis of a curvilinear system (no radial direction) the axes of the system.
[[nodiscard]] base::Mat3 axesAt(const input::CoordinateFrame& frame, const base::Vec3& position)
{
    if (frame.kind == input::CoordinateFrame::Kind::Rectangular)
        return frame.axes;
    const base::Vec3 fromOrigin = position + (-1.0) * frame.origin;
    base::Vec3 local;
    for (std::size_t c = 0; c < 3; ++c)
        local[c] =
            frame.axes(0, c) * fromOrigin[0] + frame.axes(1, c) * fromOrigin[1] + frame.axes(2, c) * fromOrigin[2];
    const double radius = std::hypot(local[0], local[1]);
    if (!(radius > 0.0))
        return frame.axes;
    const double cosine = local[0] / radius;
    const double sine = local[1] / radius;
    // the local components of the three directions
    std::array<base::Vec3, 3> directions{base::Vec3{cosine, sine, 0.0}, base::Vec3{-sine, cosine, 0.0},
                                         base::Vec3{0.0, 0.0, 1.0}};
    if (frame.kind == input::CoordinateFrame::Kind::Spherical)
    {
        const double distance = std::hypot(radius, local[2]);
        const double polarCosine = local[2] / distance;
        const double polarSine = radius / distance;
        directions = {base::Vec3{polarSine * cosine, polarSine * sine, polarCosine},
                      base::Vec3{polarCosine * cosine, polarCosine * sine, -polarSine}, base::Vec3{-sine, cosine, 0.0}};
    }
    base::Mat3 axes;
    for (std::size_t column = 0; column < 3; ++column)
        for (std::size_t r = 0; r < 3; ++r)
            axes(r, column) = frame.axes(r, 0) * directions[column][0] + frame.axes(r, 1) * directions[column][1] +
                              frame.axes(r, 2) * directions[column][2];
    return axes;
}

// A normal a ShellNormal card gives at a grid point, in basic: a unit vector, or the zero vector (the grid point has
// no shared normal).
struct GivenNormal
{
    std::int32_t gridRow = 0;
    base::Vec3 normal;
};

// The normals of the ShellNormal cards, sorted by grid point (the last card of a grid point is kept); a card whose
// grid point or coordinate system the model does not define gives none.
[[nodiscard]] std::vector<GivenNormal> givenNormalsOf(const SnapshotIndex& index)
{
    std::vector<GivenNormal> given; // bounded by the ShellNormal cards
    const base::ContractReader* table = index.snapshot().shellNormals();
    if (table == nullptr)
        return given;
    const std::span<const std::int32_t> nodeIds = table->values<std::int32_t>("nodeId");
    const bool hasFrames = table->hasColumn("frameId");
    const base::FixedListView<double> vectors = table->fixedLists<double>("vector");
    const base::FixedListView<double> positions = index.snapshot().nodes().positionsInBasic();
    for (std::int64_t row = 0; row < table->rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        const std::optional<std::int32_t> gridRow = index.points().findGrid(nodeIds[r]);
        if (!gridRow)
            continue;
        const std::int32_t frameId = hasFrames ? table->values<std::int32_t>("frameId")[r] : 0;
        const std::optional<input::CoordinateFrame> frame = index.findCoordinateFrame(frameId);
        if (!frame)
            continue;
        const std::span<const double> x = positions[*gridRow];
        const base::Mat3 axes = axesAt(*frame, base::Vec3{x[0], x[1], x[2]});
        const std::span<const double> v = vectors[row];
        base::Vec3 normal;
        for (std::size_t k = 0; k < 3; ++k)
            normal[k] = axes(k, 0) * v[0] + axes(k, 1) * v[1] + axes(k, 2) * v[2];
        const double length = base::norm(normal);
        given.push_back({*gridRow, length > 0.0 ? normal * (1.0 / length) : base::Vec3{}});
    }
    std::stable_sort(given.begin(), given.end(),
                     [](const GivenNormal& a, const GivenNormal& b) { return a.gridRow < b.gridRow; });
    return given;
}

// The value of the real or integer PARAM `name`, or `fallback`.
[[nodiscard]] double parameterOr(const input::ModelSnapshot& snapshot, std::string_view name, double fallback)
{
    if (!snapshot.hasTable(input::TableRole::Params))
        return fallback;
    const std::optional<input::ParamValue> value = snapshot.params().find(name);
    if (!value)
        return fallback;
    if (value->kind == input::ParamValue::Kind::Real)
        return value->real;
    if (value->kind == input::ParamValue::Kind::Integer)
        return static_cast<double>(value->integer);
    return fallback;
}

} // namespace

ShellDirectors ShellDirectors::build(const SnapshotIndex& index)
{
    ShellDirectors directors;
    const input::ModelSnapshot& snapshot = index.snapshot();
    const double stiffness = parameterOr(snapshot, "ShellDrillingStiffness", DEFAULT_SHELL_DRILLING_STIFFNESS);
    directors.m_drillingFactor = stiffness > 0.0 ? stiffness * SHELL_DRILLING_STIFFNESS_SCALE : 0.0;
    for (std::size_t table = 0; table < DIRECTOR_SHELL_CARDS.size(); ++table)
        if (const input::ElementBatch* batch = snapshot.findElementBatch(DIRECTOR_SHELL_CARDS[table].card))
            directors.m_normals[table].assign(
                static_cast<std::size_t>(batch->rowCount()) * DIRECTOR_SHELL_CARDS[table].nodes, base::Vec3{});
    const double angle = parameterOr(snapshot, "ShellNormalAngle", DEFAULT_SHELL_NORMAL_ANGLE_DEGREES);
    const bool sharesNormals = angle > 0.0;
    const std::vector<GivenNormal> given = givenNormalsOf(index);
    const double degrees = sharesNormals ? std::min(angle, 90.0) : DEFAULT_SHELL_NORMAL_ANGLE_DEGREES;
    const double cosine = std::cos(degrees * std::numbers::pi / 180.0);

    directors.m_hasOneSmoothSide.assign(static_cast<std::size_t>(snapshot.nodes().rowCount()), 0);
    std::vector<NodeNormal> normals; // bounded by the grid points of the elements
    collectCornerNormals<ShellQuad4>(index, normals);
    collectCornerNormals<ShellTria3>(index, normals);
    collectNodeNormals<SerendipityQuadShape>(index, DIRECTOR_SHELL_CARDS[2].card, normals);
    collectNodeNormals<QuadraticTriangleShape>(index, DIRECTOR_SHELL_CARDS[3].card, normals);
    // Grid point by grid point, in a fixed order (the stable sort keeps the card, row and node order): the normals
    // do not depend on the threads.
    std::stable_sort(normals.begin(), normals.end(),
                     [](const NodeNormal& a, const NodeNormal& b) { return a.gridRow < b.gridRow; });
    const auto slotOf = [&](const NodeNormal& node) -> base::Vec3&
    { return directors.m_normals[node.table][node.slot]; };
    // `normal` on the side of the own normal of `node`.
    const auto onSideOf = [](const NodeNormal& node, const base::Vec3& normal)
    { return base::dot(node.normal, normal) >= 0.0 ? normal : (-1.0) * normal; };
    std::size_t nextGiven = 0;
    std::size_t first = 0;
    while (first < normals.size())
    {
        std::size_t last = first + 1;
        while (last < normals.size() && normals[last].gridRow == normals[first].gridRow)
            ++last;
        const std::span<const NodeNormal> atGrid(normals.data() + first, last - first);
        const std::vector<std::size_t> group = smoothGroups(atGrid, cosine);
        const bool isOneGroup =
            std::all_of(group.begin(), group.end(), [&](std::size_t root) { return root == group.front(); });
        directors.m_hasOneSmoothSide[static_cast<std::size_t>(atGrid.front().gridRow)] = isOneGroup ? 1 : 0;
        if (sharesNormals)
        {
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
                const std::optional<base::Vec3> shared = groupNormal(atGrid, members, cosine);
                if (!shared)
                    continue;
                for (const std::size_t m : members)
                    slotOf(atGrid[m]) = onSideOf(atGrid[m], *shared);
            }
        }
        // The normal a ShellNormal card gives this grid point (the last card): the elements whose own normal is
        // within the angle of it, on either side, take it; a zero vector leaves every element its own.
        while (nextGiven < given.size() && given[nextGiven].gridRow < atGrid.front().gridRow)
            ++nextGiven;
        while (nextGiven < given.size() && given[nextGiven].gridRow == atGrid.front().gridRow)
        {
            const base::Vec3& normal = given[nextGiven].normal;
            for (const NodeNormal& node : atGrid)
            {
                if (normal == base::Vec3{})
                    slotOf(node) = base::Vec3{};
                else if (std::abs(base::dot(node.normal, normal)) >= cosine)
                    slotOf(node) = onSideOf(node, normal);
            }
            ++nextGiven;
        }
        first = last;
    }
    return directors;
}

std::optional<base::Vec3> ShellDirectors::sharedNormal(std::string_view card, std::int64_t row,
                                                       std::size_t node) const
{
    const std::size_t which = tableOf(card);
    if (which >= DIRECTOR_SHELL_CARDS.size())
        return std::nullopt;
    const std::vector<base::Vec3>& table = m_normals[which];
    const std::size_t slot = static_cast<std::size_t>(row) * DIRECTOR_SHELL_CARDS[which].nodes + node;
    if (slot >= table.size() || table[slot] == base::Vec3{})
        return std::nullopt;
    return table[slot];
}

const ShellDirectors& shellDirectors(const SnapshotIndex& index)
{
    return index.computedOnce<ShellDirectors>([&] { return ShellDirectors::build(index); });
}

} // namespace oos::elements
