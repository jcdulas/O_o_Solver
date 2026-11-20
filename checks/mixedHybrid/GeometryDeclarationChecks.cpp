// The states of the edges of a mesh read (smooth, sharp, raw), the fold angle that decides them where nothing is
// declared, and the mesh lines declared (docs/mixed-hybrid-formulation-spec.md section 4.4). None of it is in the
// reference: the expected values are those of the geometry (a circle, a plane, a right angle).

#include "CheckRunner.hpp"
#include "GeometryCheckMeshes.hpp"
#include "oos/elements/mixedHybrid/geometry/ElementSurface.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/SolidSkin.hpp"
#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <set>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
namespace geo = oos::checks::geometry;

namespace
{

using geo::CellPattern;
using mh::EdgeState;
using mh::EdgeStateOrigin;
using mh::Vector3;

constexpr double PI = std::numbers::pi;
constexpr double DEGREE = PI / 180.0;
// The fold angle of the checks: the default of the model parameter.
constexpr double FOLD_ANGLE = 20.0 * DEGREE;
// The points per edge of the geometry report: 2k + 4 at k = 4.
constexpr std::size_t REPORT_POINTS = 12;
// The tolerance of an angle computed from the coordinates, and that of the angle of two normals that agree (the arc
// cosine of 1 less the round-off: about the square root of the round-off).
constexpr double ANGLE_TOLERANCE = 1e-12;
constexpr double FLAT_ANGLE_TOLERANCE = 1e-7;

// A mesh, its topology and the surfaces of its elements.
struct ReadMesh
{
    mh::MeshReadingInput input;
    mh::Topology topology;
    std::vector<mh::ElementSurface> surfaces;
};

ReadMesh read(mh::MeshReadingInput input)
{
    ReadMesh out{std::move(input), {}, {}};
    out.topology = mh::Topology::fromMesh(out.input);
    out.surfaces = geo::surfacesOf(out.input, out.topology);
    return out;
}

// The point at τ of the conic of an edge of grids xa, xb and bulge b, with the given weight or that of its control
// polygon (Surface.hpp, "edge conics").
Vector3 conicPoint(const Vector3& xa, const Vector3& xb, const Vector3& bulge, double weight, double tau)
{
    const Vector3 chord = mh::sub3(xb, xa);
    const double w = std::isfinite(weight) ? weight : mh::controlPolygonWeight(chord, bulge);
    const Vector3 control = mh::add3(mh::add3(xa, mh::scl3(chord, 0.5)), mh::scl3(bulge, 2.0));
    const double b0 = (1.0 - tau) * (1.0 - tau);
    const double b1 = 2.0 * w * tau * (1.0 - tau);
    const double b2 = tau * tau;
    Vector3 out{};
    for (std::size_t c = 0; c < 3; ++c)
        out[c] = (b0 * xa[c] + b1 * control[c] + b2 * xb[c]) / (b0 + b1 + b2);
    return out;
}

// The largest distance between the bulges of the same edges of two readings of one mesh.
double bulgeDifference(const mh::Topology& first, const mh::Topology& second)
{
    double out = 0.0;
    for (std::size_t e = 0; e < first.edges.size(); ++e)
    {
        const std::size_t other = second.indexOf(first.edges[e].lower, first.edges[e].higher);
        out = std::fmax(out, mh::norm3(mh::sub3(first.bulges[e], second.bulges[other])));
    }
    return out;
}

// A square tube along x of side 1 and length 3, `perSide` quadrilaterals per side and `along` along it: grid (i, j)
// has index i · 4 perSide + j, and its four folds run along the grids j = c · perSide.
mh::MeshReadingInput squareTube(std::size_t perSide, std::size_t along)
{
    const std::size_t around = 4 * perSide;
    const auto corner = [&](std::size_t j) -> std::array<double, 2>
    {
        const std::size_t side = j / perSide;
        const double f = static_cast<double>(j % perSide) / static_cast<double>(perSide);
        switch (side)
        {
        case 0:
            return {-0.5 + f, -0.5};
        case 1:
            return {0.5, -0.5 + f};
        case 2:
            return {0.5 - f, 0.5};
        default:
            return {-0.5, 0.5 - f};
        }
    };
    const auto id = [&](std::size_t i, std::size_t j) { return i * around + j % around; };
    mh::MeshReadingInput input;
    for (std::size_t i = 0; i <= along; ++i)
        for (std::size_t j = 0; j < around; ++j)
        {
            const double x = 3.0 * static_cast<double>(i) / static_cast<double>(along);
            input.grids.push_back({x, corner(j)[0], corner(j)[1]});
        }
    for (std::size_t i = 0; i < along; ++i)
        for (std::size_t j = 0; j < around; ++j)
            input.shells.push_back(
                mh::ShellCorners::quadrilateral(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)));
    return input;
}

// A quarter of a cylinder of radius r along x, from x = 0 to x = 1, closed at x = 1 by a flat flange that goes in to
// the radius r / 2: `cells` quadrilaterals round the quarter, two along the cylinder and two across the flange. Grid
// (i, j) has index i (cells + 1) + j; the rim is the row i = 2.
constexpr std::size_t FLANGE_RIM_ROW = 2;
constexpr std::size_t FLANGE_ROWS = 4;
mh::MeshReadingInput flangedCylinder(double r, std::size_t cells)
{
    return geo::cellMesh(
        FLANGE_ROWS, cells,
        [&](double s, double t)
        {
            const double theta = t * 0.5 * PI;
            const double x = std::fmin(2.0 * s, 1.0);
            const double radius = s <= 0.5 ? r : r * (1.0 - 0.5 * (2.0 * s - 1.0));
            return Vector3{x, radius * std::sin(theta), radius * std::cos(theta)};
        },
        CellPattern::Quadrilaterals);
}

// The edges of the rim of a flanged cylinder.
std::vector<mh::EdgeKey> rimEdges(std::size_t cells)
{
    std::vector<mh::EdgeKey> out;
    for (std::size_t j = 0; j < cells; ++j)
        out.push_back(mh::EdgeKey::of(FLANGE_RIM_ROW * (cells + 1) + j, FLANGE_RIM_ROW * (cells + 1) + j + 1));
    return out;
}

// How far the cylinder and the flange of a flanged cylinder read are from a cylinder of radius r and from the plane
// x = 1: the largest distance over the corners and the middle of the elements and the quarter points of their edges.
double flangedCylinderError(const ReadMesh& mesh, double r, std::size_t cells)
{
    double out = 0.0;
    for (std::size_t ie = 0; ie < mesh.surfaces.size(); ++ie)
    {
        const bool isCylinder = ie / cells < FLANGE_RIM_ROW;
        for (const double u : {-1.0, -0.5, 0.0, 0.5, 1.0})
            for (const double v : {-1.0, -0.5, 0.0, 0.5, 1.0})
            {
                const Vector3 x = mesh.surfaces[ie].patch.evaluate(u, v);
                out = std::fmax(out, isCylinder ? std::abs(std::hypot(x[1], x[2]) - r) : std::abs(x[0] - 1.0));
            }
    }
    return out;
}

} // namespace

CHECK_CASE("declarations: the angle between two shells at a grid of their edge")
{
    // two unit squares along the edge 1-4, the second turned up about it by the angle
    const auto angleOf = [](double fold, bool isSecondReversed)
    {
        const std::vector<Vector3> grids{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0 + std::cos(fold), 0.0, std::sin(fold)},
                                         {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1.0 + std::cos(fold), 1.0, std::sin(fold)}};
        const mh::ShellCorners first = mh::ShellCorners::quadrilateral(0, 1, 4, 3);
        const mh::ShellCorners second = isSecondReversed ? mh::ShellCorners::quadrilateral(1, 4, 5, 2)
                                                         : mh::ShellCorners::quadrilateral(1, 2, 5, 4);
        return std::array<double, 2>{mh::cornerAngle(grids, first, second, 1, 4),
                                     mh::cornerAngle(grids, first, second, 4, 1)};
    };
    for (const double fold : {0.0, 10.0 * DEGREE, 90.0 * DEGREE, 170.0 * DEGREE})
        for (const bool isReversed : {false, true})
        {
            const std::array<double, 2> angles = angleOf(fold, isReversed);
            const std::string which = std::to_string(fold / DEGREE) + (isReversed ? " degrees, reversed" : " degrees");
            runner.expectNear(angles[0], fold, 1e-7, which + ": at the first grid");
            runner.expectNear(angles[1], fold, 1e-7, which + ": at the second grid");
        }
}

CHECK_CASE("declarations: the fold angle makes the folds of a square tube sharp and leaves its faces smooth")
{
    const std::size_t perSide = 2;
    const std::size_t along = 3;
    mh::MeshReadingInput input = squareTube(perSide, along);
    input.foldAngle = FOLD_ANGLE;
    const ReadMesh mesh = read(input);
    for (const Vector3& b : mesh.topology.bulges)
        runner.expect(mh::norm3(b) == 0.0, "flat faces, straight edges");
    std::size_t folds = 0;
    for (const mh::EdgeGeometry& e : mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS))
    {
        runner.expect(!e.raw, "the angle makes no edge raw");
        if (e.sharp)
        {
            ++folds;
            const bool isFold = e.grids.lower % perSide == 0 && e.grids.higher % perSide == 0;
            runner.expect(isFold, "the sharp edges run along the corners of the tube");
            runner.expect(e.origin == EdgeStateOrigin::Angle, "sharp by the angle");
            runner.expectNear(e.angle, 0.5 * PI, ANGLE_TOLERANCE, "an angle of 90 degrees");
            runner.expect(std::abs(e.kink - 0.5 * PI) < 1e-12, "a fold of 90 degrees " + std::to_string(e.kink));
        }
        else
        {
            runner.expect(e.kink < 1e-7, "no kink elsewhere " + std::to_string(e.kink));
            if (e.elements == 2)
                runner.expect(e.origin == EdgeStateOrigin::Angle && e.angle < FLAT_ANGLE_TOLERANCE,
                              "smooth by the angle");
        }
    }
    runner.expect(folds == 4 * along, "the folds are the sharp edges: " + std::to_string(folds));

    // no angle and no declaration: every edge smooth, the folds rounded (the reading of the reference)
    const ReadMesh rounded = read(squareTube(perSide, along));
    double largest = 0.0;
    for (std::size_t e = 0; e < rounded.topology.edges.size(); ++e)
    {
        const mh::EdgeReading& state = rounded.topology.states[e];
        runner.expect(state.state == EdgeState::Smooth, "without an angle every edge is smooth");
        if (rounded.topology.edgeElements[e].size() == 2)
            runner.expect(state.origin == EdgeStateOrigin::Unmeasured, "and nothing decided it");
        largest = std::fmax(largest, mh::norm3(rounded.topology.bulges[e]));
    }
    runner.expect(largest > 1e-2, "the folds are rounded: " + std::to_string(largest));

    // a zero angle: every edge of two shells is sharp
    mh::MeshReadingInput faceted = squareTube(perSide, along);
    faceted.foldAngle = 0.0;
    const mh::Topology facets = mh::Topology::fromMesh(faceted);
    for (std::size_t e = 0; e < facets.edges.size(); ++e)
        if (facets.edgeElements[e].size() == 2)
            runner.expect(facets.states[e].state == EdgeState::Sharp, "a zero angle: every edge sharp");
}

CHECK_CASE("declarations: a sharp edge keeps its line, a raw edge is straight")
{
    const double r = 1.5;
    const std::size_t cells = 6; // 15 degrees each: under the fold angle
    // by the angle, and declared sharp without an angle: the same reading
    mh::MeshReadingInput byAngle = flangedCylinder(r, cells);
    byAngle.foldAngle = FOLD_ANGLE;
    mh::MeshReadingInput declared = flangedCylinder(r, cells);
    for (const mh::EdgeKey& edge : rimEdges(cells))
        declared.sharpSides[edge].insert(mh::MeshReadingInput::WHOLE_EDGE);
    const ReadMesh angled = read(byAngle);
    const ReadMesh sharp = read(declared);
    runner.expect(bulgeDifference(angled.topology, sharp.topology) == 0.0, "by the angle as declared sharp");
    for (const mh::EdgeKey& edge : rimEdges(cells))
    {
        const std::size_t e = angled.topology.indexOf(edge.lower, edge.higher);
        runner.expect(angled.topology.states[e].state == EdgeState::Sharp &&
                          angled.topology.states[e].origin == EdgeStateOrigin::Angle,
                      "the rim is sharp by the angle");
        runner.expectNear(angled.topology.states[e].angle, 0.5 * PI, 1e-9, "an angle of 90 degrees");
        runner.expect(sharp.topology.states[e].origin == EdgeStateOrigin::Declared, "or declared");
        runner.expect(angled.topology.partners[e].empty(), "no shell goes on across the rim");
        // the rim is the circle of the cylinder
        const Vector3& xa = angled.input.grids[edge.lower];
        const Vector3& xb = angled.input.grids[edge.higher];
        for (const double tau : {0.25, 0.5, 0.75})
        {
            const Vector3 x = conicPoint(xa, xb, angled.topology.bulges[e], angled.topology.weights[e], tau);
            runner.expect(std::abs(std::hypot(x[1], x[2]) - r) < 1e-12 && std::abs(x[0] - 1.0) < 1e-12,
                          "the rim keeps its circle");
        }
    }
    const double error = flangedCylinderError(angled, r, cells);
    runner.expect(error < 1e-12, "the cylinder is round and the flange flat: " + std::to_string(error));
    double rimKink = 0.0;
    double otherKink = 0.0;
    for (const mh::EdgeGeometry& e : mh::geometryReport(angled.input, angled.topology, angled.surfaces, REPORT_POINTS))
        (e.sharp ? rimKink : otherKink) = std::fmax(e.sharp ? rimKink : otherKink, e.kink);
    runner.expectNear(rimKink, 0.5 * PI, 1e-7, "a fold of 90 degrees along the rim");
    runner.expect(otherKink < 1e-7, "no kink elsewhere: " + std::to_string(otherKink));

    // declared raw: the rim is a polygon, and the state says where it comes from
    mh::MeshReadingInput rawRim = flangedCylinder(r, cells);
    rawRim.foldAngle = FOLD_ANGLE;
    for (const mh::EdgeKey& edge : rimEdges(cells))
        rawRim.rawEdges.insert(edge);
    const mh::Topology straight = mh::Topology::fromMesh(rawRim);
    for (const mh::EdgeKey& edge : rimEdges(cells))
    {
        const std::size_t e = straight.indexOf(edge.lower, edge.higher);
        runner.expect(mh::norm3(straight.bulges[e]) == 0.0, "a raw rim is straight");
        const mh::EdgeReading& state = straight.states[e];
        runner.expect(state.state == EdgeState::Raw && state.origin == EdgeStateOrigin::Declared, "raw, declared");
    }

    // no angle and no declaration: the fold is rounded, the flange is no longer flat
    const ReadMesh rounded = read(flangedCylinder(r, cells));
    runner.expect(flangedCylinderError(rounded, r, cells) > 1e-3, "without an angle the fold is rounded");
}

CHECK_CASE("declarations: a curved mesh coarser than the fold angle is faceted unless declared smooth")
{
    // three cells of 30 degrees round a quarter of a cylinder
    const auto quarter = []
    {
        return geo::cellMesh(
            2, 3,
            [](double s, double t)
            {
                const double theta = t * 0.5 * PI;
                return Vector3{2.0 * s, std::sin(theta), std::cos(theta)};
            },
            CellPattern::Quadrilaterals);
    };
    mh::MeshReadingInput coarse = quarter();
    coarse.foldAngle = FOLD_ANGLE;
    const mh::Topology faceted = mh::Topology::fromMesh(coarse);
    std::vector<mh::EdgeKey> generators;
    for (std::size_t e = 0; e < faceted.edges.size(); ++e)
    {
        runner.expect(mh::norm3(faceted.bulges[e]) == 0.0, "facets: every edge straight");
        if (faceted.edgeElements[e].size() != 2)
            continue;
        const bool isGenerator = coarse.grids[faceted.edges[e].lower][0] != coarse.grids[faceted.edges[e].higher][0];
        if (isGenerator)
        {
            generators.push_back(faceted.edges[e]);
            runner.expect(faceted.states[e].state == EdgeState::Sharp, "a generator between two facets is sharp");
            runner.expectNear(faceted.states[e].angle, 30.0 * DEGREE, 1e-9, "by its 30 degrees");
        }
        else
        {
            runner.expect(faceted.states[e].state == EdgeState::Smooth, "a hoop between two facets is smooth");
        }
    }
    runner.expect(generators.size() == 4, "two generators inside, in two rows: " + std::to_string(generators.size()));

    // declared smooth, the generators give back the reading without an angle: the exact cylinder
    mh::MeshReadingInput smooth = quarter();
    smooth.foldAngle = FOLD_ANGLE;
    for (const mh::EdgeKey& edge : generators)
        smooth.smoothSides[edge].insert(mh::MeshReadingInput::WHOLE_EDGE);
    const mh::Topology round = mh::Topology::fromMesh(smooth);
    const mh::Topology reference = mh::Topology::fromMesh(quarter());
    runner.expect(bulgeDifference(round, reference) == 0.0, "declared smooth: the reading without an angle");
    for (const mh::EdgeKey& edge : generators)
        runner.expect(round.states[round.indexOf(edge.lower, edge.higher)].origin == EdgeStateOrigin::Declared,
                      "smooth, declared");
}

CHECK_CASE("declarations: the sides of a junction of three shells")
{
    // two shells of a skin in the plane z = 0 along the edge 1-4, the second turned up by `fold`, and a rib on that
    // edge
    const auto junction = [](double fold)
    {
        mh::MeshReadingInput input;
        input.grids = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0 + std::cos(fold), 0.0, std::sin(fold)},
                       {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1.0 + std::cos(fold), 1.0, std::sin(fold)},
                       {1.0, 0.0, -1.0}, {1.0, 1.0, -1.0}};
        input.shells = {mh::ShellCorners::quadrilateral(0, 1, 4, 3), mh::ShellCorners::quadrilateral(1, 2, 5, 4),
                        mh::ShellCorners::quadrilateral(1, 4, 7, 6)};
        return input;
    };
    const auto edgeOf = [](const mh::Topology& topology) { return topology.indexOf(1, 4); };
    const mh::EdgeKey edge = mh::EdgeKey::of(1, 4);

    // the skin goes through, the rib ends: by the geometry of the junction, with or without an angle
    for (const bool hasAngle : {false, true})
    {
        mh::MeshReadingInput input = junction(0.0);
        if (hasAngle)
            input.foldAngle = FOLD_ANGLE;
        const mh::Topology topology = mh::Topology::fromMesh(input);
        const std::map<std::size_t, std::size_t>& partners = topology.partners[edgeOf(topology)];
        runner.expect(partners.size() == 2 && partners.at(0) == 1 && partners.at(1) == 0, "the skin goes through");
        runner.expect(!partners.contains(2), "the rib ends at the junction");
        runner.expect(topology.states[edgeOf(topology)].state == EdgeState::Sharp, "one side ends: sharp");
        runner.expect(topology.states[edgeOf(topology)].origin == EdgeStateOrigin::Junction, "by the junction");
    }
    // a skin that folds by more than the angle: no side goes through
    mh::MeshReadingInput folded = junction(30.0 * DEGREE);
    folded.foldAngle = FOLD_ANGLE;
    const mh::Topology foldedRead = mh::Topology::fromMesh(folded);
    runner.expect(foldedRead.partners[edgeOf(foldedRead)].empty(), "a folded skin does not go through");
    runner.expect(foldedRead.states[edgeOf(foldedRead)].origin == EdgeStateOrigin::Angle, "by the angle");
    runner.expectNear(foldedRead.states[edgeOf(foldedRead)].angle, 30.0 * DEGREE, 1e-9, "of 30 degrees");
    // the same skin declared smooth, by the edge or by the side of one of its shells
    for (const std::size_t side : {mh::MeshReadingInput::WHOLE_EDGE, std::size_t{1}})
    {
        mh::MeshReadingInput smooth = junction(30.0 * DEGREE);
        smooth.foldAngle = FOLD_ANGLE;
        smooth.smoothSides[edge].insert(side);
        const mh::Topology topology = mh::Topology::fromMesh(smooth);
        runner.expect(topology.partners[edgeOf(topology)].size() == 2, "declared smooth: the skin goes through");
        runner.expect(topology.states[edgeOf(topology)].origin == EdgeStateOrigin::Declared, "declared");
    }
    // a flat skin with one of its sides declared sharp: nothing goes through
    mh::MeshReadingInput sharp = junction(0.0);
    sharp.foldAngle = FOLD_ANGLE;
    sharp.sharpSides[edge].insert(0);
    const mh::Topology sharpRead = mh::Topology::fromMesh(sharp);
    runner.expect(sharpRead.partners[edgeOf(sharpRead)].empty(), "a sharp side goes on into no shell");
    // the side of the rib declared sharp changes nothing: it faces no shell
    mh::MeshReadingInput rib = junction(0.0);
    rib.sharpSides[edge].insert(2);
    const mh::Topology ribRead = mh::Topology::fromMesh(rib);
    runner.expect(ribRead.partners[edgeOf(ribRead)].size() == 2, "the rib declared sharp: the skin still goes through");
}

CHECK_CASE("declarations: an edge declared smooth or sharp is not raw with its element")
{
    mh::MeshReadingInput input = squareTube(2, 3);
    input.rawElements.push_back(0);
    const mh::ShellCorners& shell = input.shells[0];
    const mh::EdgeKey kept = mh::EdgeKey::of(shell.at(0), shell.at(1));
    input.smoothSides[kept].insert(mh::MeshReadingInput::WHOLE_EDGE);
    const mh::Topology topology = mh::Topology::fromMesh(input);
    for (std::size_t k = 0; k < shell.count; ++k)
    {
        const mh::EdgeKey edge = mh::EdgeKey::of(shell.at(k), shell.at(k + 1));
        const mh::EdgeReading& state = topology.states[topology.indexOf(edge.lower, edge.higher)];
        if (edge == kept)
            runner.expect(state.state == EdgeState::Smooth && state.origin == EdgeStateOrigin::Declared,
                          "the edge declared smooth stays smooth");
        else
            runner.expect(state.state == EdgeState::Raw && state.origin == EdgeStateOrigin::RawElement,
                          "the other edges of the raw element are raw");
    }
}

CHECK_CASE("declarations: a declared line end keeps a corner of the outline, a continuation carries a line on")
{
    // The plate of "geometry: a rounded corner of the outline": its outline turns at grid 3 between the edges 1-3
    // and 3-4, and the reading takes the line round the corner because a triangle has that grid.
    //   5 - 6 - 7
    //   |   |   |
    //   2 - 3 - 4
    //   | / |
    //   0 - 1
    const auto plate = []
    {
        mh::MeshReadingInput input;
        constexpr std::array<std::array<double, 2>, 8> LATTICE{
            {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}}};
        for (const std::array<double, 2>& at : LATTICE)
            input.grids.push_back({at[0], at[1], 0.0});
        input.shells = {mh::ShellCorners::triangle(0, 1, 3), mh::ShellCorners::triangle(0, 3, 2),
                        mh::ShellCorners::quadrilateral(2, 3, 6, 5), mh::ShellCorners::quadrilateral(3, 4, 7, 6)};
        return input;
    };
    const auto largestTurn = [](const mh::MeshReadingInput& input, const mh::Topology& topology)
    {
        double out = 0.0;
        for (std::size_t e = 0; e < topology.edges.size(); ++e)
            out = std::fmax(out, mh::outlineTurn(input, topology, e));
        return out;
    };
    const mh::MeshReadingInput rounded = plate();
    const mh::Topology roundedRead = mh::Topology::fromMesh(rounded);
    runner.expectNear(largestTurn(rounded, roundedRead), 0.25 * PI, 1e-14,
                      "without a declaration the corner is rounded");
    runner.expect(mh::degenerateCorners(rounded, roundedRead).size() == 1, "and closes a triangle");

    mh::MeshReadingInput ended = plate();
    ended.declaredLines[{3, 1}] = {};
    ended.declaredLines[{3, 4}] = {};
    const mh::Topology endedRead = mh::Topology::fromMesh(ended);
    runner.expect(largestTurn(ended, endedRead) == 0.0, "the line ended at the corner: a straight outline");
    runner.expect(mh::degenerateCorners(ended, endedRead).empty(), "and sound corners");
    for (const Vector3& b : endedRead.bulges)
        runner.expect(mh::norm3(b) == 0.0, "a flat plate with straight edges");

    // a quarter of a cylinder whose hoops are cut by raw generators: each hoop edge is a line of one edge, straight;
    // declared to go on across the generators, the hoops are circles again
    const std::size_t cells = 4;
    const auto quarter = [&]
    {
        mh::MeshReadingInput input = geo::cellMesh(
            1, cells,
            [](double s, double t)
            {
                const double theta = t * 0.5 * PI;
                return Vector3{s, std::sin(theta), std::cos(theta)};
            },
            CellPattern::Quadrilaterals);
        for (std::size_t j = 1; j < cells; ++j)
            input.sharpSides[mh::EdgeKey::of(j, cells + 1 + j)].insert(mh::MeshReadingInput::WHOLE_EDGE);
        return input;
    };
    const mh::Topology cut = mh::Topology::fromMesh(quarter());
    for (const Vector3& b : cut.bulges)
        runner.expect(mh::norm3(b) == 0.0, "hoops cut by sharp generators are straight");
    mh::MeshReadingInput carried = quarter();
    for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t j = 1; j < cells; ++j)
        {
            const std::size_t g = row * (cells + 1) + j;
            carried.declaredLines[{g, g - 1}] = {g + 1};
            carried.declaredLines[{g, g + 1}] = {g - 1};
        }
    const mh::Topology carriedRead = mh::Topology::fromMesh(carried);
    for (std::size_t e = 0; e < carriedRead.edges.size(); ++e)
    {
        const Vector3& xa = carried.grids[carriedRead.edges[e].lower];
        const Vector3& xb = carried.grids[carriedRead.edges[e].higher];
        if (xa[0] != xb[0])
            continue; // a generator
        for (const double tau : {0.25, 0.5, 0.75})
        {
            const Vector3 x = conicPoint(xa, xb, carriedRead.bulges[e], carriedRead.weights[e], tau);
            runner.expect(std::abs(std::hypot(x[1], x[2]) - 1.0) < 1e-12, "a hoop declared to go on is a circle");
        }
    }
}

CHECK_CASE("declarations: the skin of solids by the fold angle is the skin with its edges declared sharp")
{
    // a quarter of a thick tube of tetrahedra, three cells of 30 degrees round it: its hoops turn by more than the
    // fold angle of the checks, so the angle here is 40 degrees
    const std::size_t nt = 3;
    const auto tube = [](double s, double t, double w)
    {
        const double r = 1.0 + s;
        const double theta = t * 0.5 * PI;
        return Vector3{r * std::cos(theta), r * std::sin(theta), 0.5 * w};
    };
    const geo::TetMesh tets = geo::tetMesh(1, nt, 1, tube);
    std::vector<mh::SolidCell> solids;
    for (const auto& t : tets.tets)
        solids.push_back({{t[0], t[1], t[2], t[3]}, {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}});
    std::set<mh::EdgeKey> creases;
    for (const auto& [a, b] : geo::tetMeshCreases(1, nt, 1))
        creases.insert(mh::EdgeKey::of(a, b));
    const std::vector<mh::SkinFace> faces = mh::skinFaces(tets.grids, solids);
    const mh::SkinReading declared = mh::readSolidSkin(tets.grids, faces, creases);

    mh::SkinDeclarations byAngle;
    byAngle.foldAngle = 40.0 * DEGREE;
    const mh::SkinReading angled = mh::readSolidSkin(tets.grids, faces, byAngle);
    std::set<mh::EdgeKey> sharp;
    for (const auto& [edge, state] : angled.states)
        if (state.state == EdgeState::Sharp)
        {
            sharp.insert(edge);
            runner.expect(state.origin == EdgeStateOrigin::Angle, "sharp by the angle");
            // 90 degrees between the faces of the tube; between their facets, half a cell (15 degrees) less or more
            // where a curved face meets a flat end
            runner.expect(std::abs(state.angle - 0.5 * PI) < 15.0 * DEGREE + 1e-9,
                          "the edges of the block: about 90 degrees, found " + std::to_string(state.angle / DEGREE));
        }
    runner.expect(sharp == creases, "the sharp edges are the edges of the block");
    runner.expect(angled.conics.size() == declared.conics.size(), "the same edges");
    for (const auto& [edge, conic] : declared.conics)
    {
        const auto found = angled.conics.find(edge);
        runner.expect(found != angled.conics.end() && mh::norm3(mh::sub3(found->second.bulge, conic.bulge)) == 0.0 &&
                          found->second.weight == conic.weight,
                      "the same conic");
    }

    // at 20 degrees the hoops of 30 degrees are sharp too, unless declared smooth
    mh::SkinDeclarations coarse;
    coarse.foldAngle = FOLD_ANGLE;
    const mh::SkinReading faceted = mh::readSolidSkin(tets.grids, faces, coarse);
    std::set<mh::EdgeKey> extra;
    for (const auto& [edge, state] : faceted.states)
        if (state.state == EdgeState::Sharp && !creases.contains(edge))
            extra.insert(edge);
    runner.expect(!extra.empty(), "a coarse curved skin is faceted by the angle");
    coarse.smoothEdges = extra;
    const mh::SkinReading smoothed = mh::readSolidSkin(tets.grids, faces, coarse);
    for (const auto& [edge, conic] : declared.conics)
        runner.expect(mh::norm3(mh::sub3(smoothed.conics.at(edge).bulge, conic.bulge)) == 0.0,
                      "declared smooth: the skin with its edges declared sharp");
}

CHECK_CASE("declarations: the reading by the fold angle does not depend on the corner or the sense of the elements")
{
    // A spherical zone meshed coarsely (three cells of 24 degrees along the meridians, of 30 degrees along the
    // parallels) and distorted, in triangles and in a mix of both shapes: some of its edges are above the fold angle,
    // others below. The same elements written from another corner, or turned over, are the same mesh.
    const auto zone = [](double s, double t)
    {
        // the grids inside the zone moved along it, by a tenth of a cell
        const bool isInside = s > 0.0 && s < 1.0 && t > 0.0 && t < 1.0;
        const double shift = isInside ? 0.1 / 3.0 : 0.0;
        const double phi = (18.0 + 72.0 * (s + shift * std::sin(7.0 * t))) * DEGREE;
        const double theta = 90.0 * (t + shift * std::cos(5.0 * s)) * DEGREE;
        return Vector3{std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi)};
    };
    for (const CellPattern pattern : {CellPattern::AlternateDiagonals, CellPattern::Mixed})
    {
        mh::MeshReadingInput base = geo::cellMesh(3, 3, zone, pattern);
        base.foldAngle = FOLD_ANGLE;
        const mh::Topology reference = mh::Topology::fromMesh(base);
        std::size_t sharp = 0;
        std::size_t smooth = 0;
        for (std::size_t e = 0; e < reference.edges.size(); ++e)
            if (reference.edgeElements[e].size() == 2)
                (reference.states[e].state == EdgeState::Sharp ? sharp : smooth) += 1;
        runner.expect(sharp > 0 && smooth > 0,
                      "edges on both sides of the fold angle: " + std::to_string(sharp) + " sharp, " +
                          std::to_string(smooth) + " smooth");
        // turn 1, 2: the elements start one or two corners later; turn 3: they are turned over
        for (std::size_t turn = 1; turn <= 3; ++turn)
        {
            mh::MeshReadingInput variant = base;
            for (mh::ShellCorners& shell : variant.shells)
            {
                const mh::ShellCorners written = shell;
                for (std::size_t k = 0; k < shell.count; ++k)
                    shell.grids[k] = turn == 3 ? written.at(shell.count - k) : written.at(k + turn);
            }
            const mh::Topology read = mh::Topology::fromMesh(variant);
            double worst = 0.0;
            bool isSameState = true;
            for (std::size_t e = 0; e < reference.edges.size(); ++e)
            {
                const std::size_t other = read.indexOf(reference.edges[e].lower, reference.edges[e].higher);
                isSameState = isSameState && read.states[other].state == reference.states[e].state;
                worst = std::fmax(worst, mh::norm3(mh::sub3(read.bulges[other], reference.bulges[e])));
            }
            const std::string which = "turn " + std::to_string(turn);
            runner.expect(isSameState, which + ": the same states");
            runner.expect(worst < 1e-10, which + ": the same curves, within " + std::to_string(worst));
        }
    }
}
