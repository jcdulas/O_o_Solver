#include "CheckRunner.hpp"
#include "GeometryCheckMeshes.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/geometry/EdgeSupport.hpp"
#include "oos/elements/mixedHybrid/geometry/ElementSurface.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/SolidSkin.hpp"
#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
namespace geo = oos::checks::geometry;

namespace
{

using geo::CellPattern;
using mh::Vector3;

constexpr double PI = std::numbers::pi;
constexpr double DEGREE = PI / 180.0;
// The points per edge of the geometry report: 2k + 4 at k = 4.
constexpr std::size_t REPORT_POINTS = 12;
// The tolerance of the values computed by the reference crate (absolute on quantities of order 1).
constexpr double REFERENCE_TOLERANCE = 1e-12;

double distance(const Vector3& a, const Vector3& b)
{
    return mh::norm3(mh::sub3(a, b));
}

double relativeDifference(const Vector3& value, const Vector3& expected)
{
    double scale = 1.0;
    for (const double e : expected)
        scale = std::fmax(scale, std::abs(e));
    return distance(value, expected) / scale;
}

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

// The points of the reference domain of an element at which the checks look: a 7 × 7 grid of the square, or the
// points of a 6 × 6 grid of the square inside the triangle.
std::vector<mh::ReferencePoint> samplePoints(std::size_t cornerCount)
{
    std::vector<mh::ReferencePoint> out;
    for (int i = 0; i <= 6; ++i)
        for (int j = 0; j <= 6; ++j)
        {
            const double u = -1.0 + i / 3.0;
            const double v = -1.0 + j / 3.0;
            if (cornerCount == 4 || u + v <= 1e-12)
                out.push_back({u, v});
        }
    return out;
}

// The largest |dist(X)| and the largest angle between the normal of the patches and normal(X), over the sample points
// of every element.
std::array<double, 2> surfaceError(const ReadMesh& mesh, const std::function<double(const Vector3&)>& dist,
                                   const std::function<Vector3(const Vector3&)>& normal)
{
    std::array<double, 2> out{0.0, 0.0};
    for (const mh::ElementSurface& s : mesh.surfaces)
        for (const mh::ReferencePoint& p : samplePoints(s.patch.cornerCount()))
        {
            const mh::SurfacePointFrame<double> f = mh::frame(s.patch, p[0], p[1]);
            out[0] = std::fmax(out[0], std::abs(dist(f.position)));
            const double c = std::fmin(std::abs(mh::dot3(f.normal, normal(f.position))), 1.0);
            out[1] = std::fmax(out[1], mh::norm3(mh::cross3(f.normal, normal(f.position))) + (1.0 - c));
        }
    return out;
}

// The largest angle between the normal of each patch at its corners and normal(X).
double gridNormalError(const ReadMesh& mesh, const std::function<Vector3(const Vector3&)>& normal)
{
    double out = 0.0;
    for (const mh::ElementSurface& s : mesh.surfaces)
        for (std::size_t k = 0; k < s.patch.cornerCount(); ++k)
        {
            const mh::ReferencePoint c = mh::referenceCorner(s.patch.cornerCount(), k);
            const mh::SurfacePointFrame<double> f = mh::frame(s.patch, c[0], c[1]);
            out = std::fmax(out, mh::norm3(mh::cross3(f.normal, normal(f.position))));
        }
    return out;
}

Vector3 cylinderPoint(double radius, double length, double angle, double s, double t)
{
    const double theta = t * angle;
    return {s * length, radius * std::sin(theta), radius * std::cos(theta)};
}

// The point of the spherical zone of the reference tests (colatitude 18 to 90 degrees, a quarter of a turn).
Vector3 spherePoint(double radius, double s, double t)
{
    const double phi = (18.0 + 72.0 * s) * DEGREE;
    const double theta = 90.0 * t * DEGREE;
    return {radius * std::sin(phi) * std::cos(theta), radius * std::sin(phi) * std::sin(theta), radius * std::cos(phi)};
}

// The ellipsoid of semi-axes (1, 0.7, 0.5) on the same zone.
Vector3 ellipsoidPoint(double s, double t)
{
    const Vector3 x = spherePoint(1.0, s, t);
    return {x[0], 0.7 * x[1], 0.5 * x[2]};
}

// The sphere of radius 10 of the reference test geometry_report (tests/tdnns.rs).
Vector3 reportSpherePoint(double x, double y)
{
    const double phi = 0.3 + x;
    const double theta = y * PI / 2.0;
    return {10.0 * std::sin(phi) * std::cos(theta), 10.0 * std::sin(phi) * std::sin(theta), 10.0 * std::cos(phi)};
}

// The mesh of the reference test geometry_report: quadrilaterals, or each cut along G1-G3.
mh::MeshReadingInput reportSphere(bool triangles)
{
    return geo::cellMesh(4, 4, reportSpherePoint, triangles ? CellPattern::FirstDiagonal : CellPattern::Quadrilaterals);
}

// A flat lx × ly mesh of n × n cells, its interior grids moved by `distort` of a cell (the plate mesh of the
// reference, src/bench.rs::plate_mesh).
mh::MeshReadingInput flatMesh(double lx, double ly, std::size_t n, double distort, CellPattern pattern)
{
    geo::QuadMesh mesh = geo::mappedMesh(n, n, [&](double s, double t) { return Vector3{s * lx, t * ly, 0.0}; });
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 1; j < n; ++j)
        {
            const std::size_t g = i * (n + 1) + j;
            const double a = std::sin(1.7 * static_cast<double>(i) + 0.9 * static_cast<double>(j));
            const double b = std::cos(2.3 * static_cast<double>(i) - 1.3 * static_cast<double>(j));
            mesh.grids[g][0] += distort * a * lx / static_cast<double>(n);
            mesh.grids[g][1] += distort * b * ly / static_cast<double>(n);
        }
    mh::MeshReadingInput input;
    input.grids = mesh.grids;
    input.shells = geo::cellShells(n, mesh.quads, pattern);
    return input;
}

// The point of the conic of an edge of grids xa, xb and bulge b at τ, with the weight of its control polygon or the
// given one.
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

// A probe value of the reference: an element point (its patch, W and normal).
struct ElementProbe
{
    std::size_t element;
    double u;
    double v;
    Vector3 point;
    double weight;
    Vector3 normal;
};

// A probe value of the reference: the bulge and the weight of an edge (NaN: that of its control polygon).
struct EdgeProbe
{
    std::size_t edge;
    Vector3 bulge;
    double weight;
};

// A probe value of the reference: the three moments of degree 4 of an edge.
struct MomentProbe
{
    std::size_t edge;
    std::array<Vector3, 3> position;
    std::array<double, 3> weight;
};

void compareWithReference(oos::checks::CheckRunner& runner, const std::string& name, const ReadMesh& mesh,
                          std::size_t edgeCount, const std::vector<EdgeProbe>& edges,
                          const std::vector<ElementProbe>& points, const std::vector<MomentProbe>& moments,
                          double patchKink)
{
    runner.expect(mesh.topology.edges.size() == edgeCount, name + ": edge count");
    if (mesh.topology.edges.size() != edgeCount)
        return;
    for (const EdgeProbe& e : edges)
    {
        runner.expect(relativeDifference(mesh.topology.bulges[e.edge], e.bulge) < REFERENCE_TOLERANCE,
                      name + ": bulge of edge " + std::to_string(e.edge));
        const double w = mesh.topology.weights[e.edge];
        runner.expect(std::isnan(e.weight) ? std::isnan(w) : std::abs(w - e.weight) < REFERENCE_TOLERANCE,
                      name + ": weight of edge " + std::to_string(e.edge));
    }
    for (const ElementProbe& p : points)
    {
        const mh::Patch& patch = mesh.surfaces[p.element].patch;
        const mh::SurfacePointFrame<double> f = mh::frame(patch, p.u, p.v);
        runner.expect(relativeDifference(f.position, p.point) < REFERENCE_TOLERANCE,
                      name + ": point of element " + std::to_string(p.element));
        runner.expect(distance(f.normal, p.normal) < REFERENCE_TOLERANCE,
                      name + ": normal of element " + std::to_string(p.element));
        runner.expectNear(mh::weightPolynomial(patch).evaluate(p.u, p.v), p.weight, REFERENCE_TOLERANCE,
                          name + ": W of element " + std::to_string(p.element));
    }
    std::vector<mh::Polynomial2> weights;
    for (const mh::ElementSurface& s : mesh.surfaces)
        weights.push_back(mh::weightPolynomial(s.patch));
    const std::vector<mh::EdgeMoments> all = mh::edgeMoments(mesh.input, mesh.topology, mesh.surfaces, weights, 4);
    for (const MomentProbe& m : moments)
        for (std::size_t l = 0; l < 3; ++l)
        {
            runner.expect(relativeDifference(all[m.edge].position[l], m.position[l]) < REFERENCE_TOLERANCE,
                          name + ": moment " + std::to_string(l) + " of edge " + std::to_string(m.edge));
            runner.expect(std::abs(all[m.edge].weight[l] - m.weight[l]) < REFERENCE_TOLERANCE,
                          name + ": weight moment " + std::to_string(l) + " of edge " + std::to_string(m.edge));
        }
    const double kink = geo::largestKink(mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS));
    // absolute: the angle of two unit normals 1e-3 apart is the arc cosine of a number 1e-6 from 1, which amplifies
    // the round-off of their dot product by 1e3
    runner.expect(std::abs(kink - patchKink) < REFERENCE_TOLERANCE, name + ": largest kink " + std::to_string(kink));
}

} // namespace

// A cylinder meshed along its generators and circles is read exactly: the generators are straight, the circles are
// circular arcs, the patches are on the cylinder with its normals, the report has no kink.
CHECK_CASE("geometry: cylinder in quadrilaterals is read exactly")
{
    const double r = 1.5;
    const ReadMesh mesh = read(geo::cellMesh(
        4, 4, [&](double s, double t) { return cylinderPoint(r, 2.0, 0.5 * PI, s, t); }, CellPattern::Quadrilaterals));
    const auto dist = [&](const Vector3& x) { return std::hypot(x[1], x[2]) - r; };
    const auto normal = [](const Vector3& x) { return mh::unit3(Vector3{0.0, x[1], x[2]}); };
    const std::array<double, 2> error = surfaceError(mesh, dist, normal);
    runner.expect(error[0] < 1e-12 * r, "points on the cylinder: " + std::to_string(error[0]));
    runner.expect(error[1] < 1e-12, "normals of the cylinder: " + std::to_string(error[1]));
    runner.expect(gridNormalError(mesh, normal) < 1e-14, "normals at the grids");
    for (std::size_t e = 0; e < mesh.topology.edges.size(); ++e)
    {
        const Vector3 chord =
            mh::sub3(mesh.input.grids[mesh.topology.edges[e].higher], mesh.input.grids[mesh.topology.edges[e].lower]);
        // a generator (along x) is straight
        if (std::abs(chord[0]) > 0.0)
            runner.expect(mh::norm3(mesh.topology.bulges[e]) < 1e-15, "straight generator");
    }
    const std::vector<mh::EdgeGeometry> report =
        mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS);
    runner.expect(geo::largestKink(report) < 1e-7, "kinks " + std::to_string(geo::largestKink(report)));
}

// A sphere meshed along its parallels and meridians is read exactly (documents 25, 3.2 and 34, section 9: kinks of
// 2.6e-8 rad, the round-off of the angle of two unit normals).
CHECK_CASE("geometry: sphere in quadrilaterals is read exactly")
{
    for (const double radius : {1.0, 10.0})
    {
        const ReadMesh mesh = read(geo::cellMesh(
            4, 4, [&](double s, double t) { return spherePoint(radius, s, t); }, CellPattern::Quadrilaterals));
        const auto dist = [&](const Vector3& x) { return mh::norm3(x) - radius; };
        const auto normal = [](const Vector3& x) { return mh::unit3(x); };
        const std::array<double, 2> error = surfaceError(mesh, dist, normal);
        runner.expect(error[0] < 1e-12 * radius, "points on the sphere: " + std::to_string(error[0]));
        runner.expect(error[1] < 1e-12, "normals of the sphere: " + std::to_string(error[1]));
        runner.expect(gridNormalError(mesh, normal) < 1e-13, "normals at the grids");
        const double kink =
            geo::largestKink(mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS));
        runner.expect(kink < 1e-7, "kinks " + std::to_string(kink));
    }
    // the mesh of the reference test geometry_report (tests/tdnns.rs)
    const ReadMesh mesh = read(reportSphere(false));
    const double kink = geo::largestKink(mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS));
    runner.expect(kink < 1e-7, "kinks of the reference test " + std::to_string(kink));
}

// A cylinder meshed with triangles is reproduced to round-off whatever the diagonals (document 26, 3.2), with one
// diagonal, the other, alternate ones, and with quadrilaterals and triangles.
CHECK_CASE("geometry: cylinder in triangles is read exactly")
{
    for (const CellPattern pattern :
         {CellPattern::FirstDiagonal, CellPattern::SecondDiagonal, CellPattern::AlternateDiagonals, CellPattern::Mixed})
    {
        const ReadMesh mesh = read(
            geo::cellMesh(4, 4, [](double s, double t) { return cylinderPoint(1.0, 1.0, 0.5 * PI, s, t); }, pattern));
        const auto dist = [](const Vector3& x) { return std::hypot(x[1], x[2]) - 1.0; };
        const auto normal = [](const Vector3& x) { return mh::unit3(Vector3{0.0, x[1], x[2]}); };
        const std::array<double, 2> error = surfaceError(mesh, dist, normal);
        const std::string which = "pattern " + std::to_string(static_cast<int>(pattern));
        runner.expect(error[0] < 1e-12, which + ": points on the cylinder " + std::to_string(error[0]));
        runner.expect(error[1] < 1e-12, which + ": normals of the cylinder " + std::to_string(error[1]));
    }
}

// Triangles on a sphere: their edges are arcs of great circles and the normals at the grids are exact; the fitted
// interiors leave kinks of about 1.7e-3 rad between them (document 34, section 9).
CHECK_CASE("geometry: triangles on a sphere")
{
    const ReadMesh mesh = read(reportSphere(true));
    const auto normal = [](const Vector3& x) { return mh::unit3(x); };
    runner.expect(gridNormalError(mesh, normal) < 1e-13, "normals at the grids");
    double worstEdge = 0.0;
    for (std::size_t e = 0; e < mesh.topology.edges.size(); ++e)
        for (const double tau : {0.1, 0.35, 0.5, 0.8})
        {
            const Vector3 x = conicPoint(mesh.input.grids[mesh.topology.edges[e].lower],
                                         mesh.input.grids[mesh.topology.edges[e].higher], mesh.topology.bulges[e],
                                         mesh.topology.weights[e], tau);
            worstEdge = std::fmax(worstEdge, std::abs(mh::norm3(x) - 10.0));
        }
    runner.expect(worstEdge < 1e-12 * 10.0, "edges on the sphere: " + std::to_string(worstEdge));
    const double kink = geo::largestKink(mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS));
    runner.expect(kink > 1.5e-3 && kink < 1.9e-3, "kinks of the fitted interiors: " + std::to_string(kink));
}

// A flat mesh is read flat, with every pattern, distorted or not: the patches are in the plane with its normal, and
// the edges are straight (document 25, 3.2: a distorted flat mesh has straight interior edges).
CHECK_CASE("geometry: flat mesh is read flat")
{
    for (const double distort : {0.0, 0.25})
        for (const CellPattern pattern : {CellPattern::Quadrilaterals, CellPattern::FirstDiagonal,
                                          CellPattern::AlternateDiagonals, CellPattern::Mixed})
        {
            const ReadMesh mesh = read(flatMesh(3.0, 2.0, 4, distort, pattern));
            const auto dist = [](const Vector3& x) { return x[2]; };
            const auto normal = [](const Vector3&) { return Vector3{0.0, 0.0, 1.0}; };
            const std::array<double, 2> error = surfaceError(mesh, dist, normal);
            const std::string which =
                "distortion " + std::to_string(distort) + ", pattern " + std::to_string(static_cast<int>(pattern));
            runner.expect(error[0] == 0.0, which + ": in the plane " + std::to_string(error[0]));
            runner.expect(error[1] < 1e-15, which + ": normal of the plane " + std::to_string(error[1]));
            double worstBulge = 0.0;
            for (const Vector3& b : mesh.topology.bulges)
                worstBulge = std::fmax(worstBulge, mh::norm3(b));
            runner.expect(worstBulge < 1e-12, which + ": straight edges " + std::to_string(worstBulge));
        }
}

// A flat annular sector meshed along its circles and radii: read flat, its radii straight and its circles exact arcs
// (the reading of tests/tria.rs::annular and tests/plates.rs), but where two triangles share a circle edge.
CHECK_CASE("geometry: annulus has circular edges")
{
    for (const CellPattern pattern : {CellPattern::Quadrilaterals, CellPattern::FirstDiagonal})
    {
        const ReadMesh mesh = read(geo::cellMesh(
            2, 2,
            [](double s, double t)
            {
                const double r = 1.0 + s;
                const double theta = 0.9 * t;
                return Vector3{r * std::cos(theta), r * std::sin(theta), 0.0};
            },
            pattern));
        for (std::size_t e = 0; e < mesh.topology.edges.size(); ++e)
        {
            const Vector3& xa = mesh.input.grids[mesh.topology.edges[e].lower];
            const Vector3& xb = mesh.input.grids[mesh.topology.edges[e].higher];
            const double ra = mh::norm3(xa);
            const double rb = mh::norm3(xb);
            if (std::abs(ra - rb) > 1e-12)
            {
                // a radius, or a diagonal of a triangle: in the plane
                runner.expect(std::abs(mesh.topology.bulges[e][2]) < 1e-15, "edge in the plane");
                continue;
            }
            if (pattern != CellPattern::Quadrilaterals && mesh.topology.edgeElements[e].size() == 2)
            {
                // an edge of two triangles is a plane section of the surface through its chord (document 26, 3.2,
                // rule 3): on a plane, the chord
                runner.expect(mh::norm3(mesh.topology.bulges[e]) < 1e-15, "edge of two triangles straight");
                continue;
            }
            for (const double tau : {0.2, 0.5, 0.7})
            {
                const Vector3 x = conicPoint(xa, xb, mesh.topology.bulges[e], mesh.topology.weights[e], tau);
                runner.expect(std::abs(mh::norm3(x) - ra) < 1e-12 && x[2] == 0.0, "circular edge");
            }
        }
    }
}

// A raw edge is straight and a raw element is a facet (the bilinear patch of its grids), on a curved mesh.
CHECK_CASE("geometry: raw edges are straight and raw elements are facets")
{
    mh::MeshReadingInput input = geo::cellMesh(
        4, 4, [](double s, double t) { return cylinderPoint(1.0, 2.0, 0.5 * PI, s, t); }, CellPattern::Quadrilaterals);
    // a hoop edge of the middle row raw, and an element raw
    input.rawEdges.insert(mh::EdgeKey::of(1 * 5 + 2, 1 * 5 + 3));
    input.rawElements.push_back(10);
    const ReadMesh mesh = read(input);
    const std::size_t rawEdge = mesh.topology.indexOf(7, 8);
    runner.expect(mh::norm3(mesh.topology.bulges[rawEdge]) == 0.0, "raw edge straight");
    const mh::ShellCorners& facet = mesh.input.shells[10];
    for (std::size_t k = 0; k < 4; ++k)
        runner.expect(mh::norm3(mesh.topology.bulges[mesh.topology.indexOf(facet.at(k), facet.at(k + 1))]) == 0.0,
                      "edges of the raw element straight");
    const mh::RationalPatch* patch = mesh.surfaces[10].patch.rational();
    runner.expect(patch != nullptr, "the facet is a rational patch");
    if (patch != nullptr)
        for (const mh::ReferencePoint& p : samplePoints(4))
        {
            Vector3 bilinear{};
            for (std::size_t k = 0; k < 4; ++k)
            {
                const double nk =
                    0.25 * (1.0 + mh::REFERENCE_SQUARE[k][0] * p[0]) * (1.0 + mh::REFERENCE_SQUARE[k][1] * p[1]);
                bilinear = mh::add3(bilinear, mh::scl3(mesh.input.grids[facet.grids[k]], nk));
            }
            runner.expect(distance(patch->evaluate(p[0], p[1]), bilinear) < 1e-14, "bilinear facet");
        }
    // the edges of the element that share the raw edge are straight there too
    for (const std::size_t ie : mesh.topology.edgeElements[rawEdge])
    {
        const mh::Patch& p = mesh.surfaces[ie].patch;
        for (std::size_t k = 0; k < 4; ++k)
        {
            const mh::ShellCorners& shell = mesh.input.shells[ie];
            if (mh::EdgeKey::of(shell.at(k), shell.at(k + 1)) != mh::EdgeKey::of(7, 8))
                continue;
            const Vector3 mid = mh::edgeEvaluation(p, k, 0.5).position;
            const Vector3 chordMid =
                mh::scl3(mh::add3(mesh.input.grids[shell.at(k)], mesh.input.grids[shell.at(k + 1)]), 0.5);
            runner.expect(distance(mid, chordMid) < 1e-15, "straight edge of the neighbour");
        }
    }
}

// A box beam with its four folds declared raw keeps its folds: kinks of 90 degrees along them, none elsewhere, and its
// faces flat.
CHECK_CASE("geometry: box beam keeps its declared folds")
{
    const std::size_t perSide = 2;
    const std::size_t along = 3;
    const std::size_t around = 4 * perSide;
    const auto corner = [&](double p)
    {
        const std::size_t side = std::min<std::size_t>(static_cast<std::size_t>(std::floor(p)), 3);
        const double f = p - static_cast<double>(side);
        switch (side)
        {
        case 0:
            return std::array<double, 2>{-0.5 + f, -0.5};
        case 1:
            return std::array<double, 2>{0.5, -0.5 + f};
        case 2:
            return std::array<double, 2>{0.5 - f, 0.5};
        default:
            return std::array<double, 2>{-0.5, 0.5 - f};
        }
    };
    const auto id = [&](std::size_t i, std::size_t j) { return i * around + j % around; };
    mh::MeshReadingInput input;
    for (std::size_t i = 0; i <= along; ++i)
        for (std::size_t j = 0; j < around; ++j)
        {
            const std::array<double, 2> yz = corner(static_cast<double>(j) / static_cast<double>(perSide));
            input.grids.push_back({3.0 * static_cast<double>(i) / static_cast<double>(along), yz[0], yz[1]});
        }
    for (std::size_t i = 0; i < along; ++i)
        for (std::size_t j = 0; j < around; ++j)
            input.shells.push_back(
                mh::ShellCorners::quadrilateral(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)));
    for (std::size_t i = 0; i < along; ++i)
        for (std::size_t c = 0; c < 4; ++c)
            input.rawEdges.insert(mh::EdgeKey::of(id(i, c * perSide), id(i + 1, c * perSide)));
    const ReadMesh mesh = read(input);
    for (const Vector3& b : mesh.topology.bulges)
        runner.expect(mh::norm3(b) == 0.0, "flat faces, straight edges");
    const std::vector<mh::EdgeGeometry> report =
        mh::geometryReport(mesh.input, mesh.topology, mesh.surfaces, REPORT_POINTS);
    std::size_t folds = 0;
    for (const mh::EdgeGeometry& e : report)
    {
        if (e.raw)
        {
            ++folds;
            runner.expect(std::abs(e.kink - 0.5 * PI) < 1e-12, "a fold of 90 degrees " + std::to_string(e.kink));
        }
        else
        {
            runner.expect(e.kink < 1e-7, "no kink elsewhere " + std::to_string(e.kink));
        }
        runner.expect(!e.sharp, "no sharp edge");
    }
    runner.expect(folds == 4 * along, "the folds are the raw edges");
}

// The edge moments of a straight edge are those of its chord: ∫ P_l(2τ − 1) X dτ = (X_a + X_b)/2, (X_b − X_a)/6, 0
// and ∫ P_l W dτ = 1, 0, 0.
CHECK_CASE("geometry: edge moments of a straight edge are those of its chord")
{
    for (const CellPattern pattern : {CellPattern::Quadrilaterals, CellPattern::AlternateDiagonals})
    {
        mh::MeshReadingInput input =
            geo::cellMesh(3, 2, [](double s, double t) { return cylinderPoint(1.0, 2.0, 0.5 * PI, s, t); }, pattern);
        for (std::size_t ie = 0; ie < input.shells.size(); ++ie)
            input.rawElements.push_back(ie);
        const ReadMesh mesh = read(input);
        std::vector<mh::Polynomial2> weights;
        for (const mh::ElementSurface& s : mesh.surfaces)
            weights.push_back(mh::weightPolynomial(s.patch));
        const std::vector<mh::EdgeMoments> moments =
            mh::edgeMoments(mesh.input, mesh.topology, mesh.surfaces, weights, 4);
        for (std::size_t e = 0; e < mesh.topology.edges.size(); ++e)
        {
            const Vector3& xa = mesh.input.grids[mesh.topology.edges[e].lower];
            const Vector3& xb = mesh.input.grids[mesh.topology.edges[e].higher];
            const std::array<Vector3, 3> expected{mh::scl3(mh::add3(xa, xb), 0.5),
                                                  mh::scl3(mh::sub3(xb, xa), 1.0 / 6.0), Vector3{}};
            const std::array<double, 3> expectedWeight{1.0, 0.0, 0.0};
            runner.expect(moments[e].position.size() == 3, "three moments at k = 4");
            for (std::size_t l = 0; l < 3; ++l)
            {
                runner.expect(distance(moments[e].position[l], expected[l]) < 1e-14, "moment of the chord");
                runner.expect(std::abs(moments[e].weight[l] - expectedWeight[l]) < 1e-14, "moment of W");
            }
        }
    }
}

// The reading of curved meshes against the values of the reference crate at commit 9f55f0b (the probe of this port):
// bulges and weights of edges, points, denominators and normals of the patches, edge moments at k = 4, and the largest
// kink of the patches. The meshes are 4 × 4 cells of an ellipsoid (quadrilaterals, and every other cell cut in two
// triangles), of a sphere (triangles), of the sphere of the reference test geometry_report, and of a cylinder
// (alternate diagonals).
CHECK_CASE("geometry: reading against the reference")
{
    constexpr double NAN_WEIGHT = std::numeric_limits<double>::quiet_NaN();
    compareWithReference(runner, "ellipsoid, quadrilaterals",
                         read(geo::cellMesh(4, 4, ellipsoidPoint, CellPattern::Quadrilaterals)), 40,
                         {{0, {1.47033694902926905e-3, 4.86403178511643030e-15, 5.21709502144113331e-3}, NAN_WEIGHT},
                          {7, {1.13471894472785785e-3, 1.18883023491465119e-3, 3.33568166220780428e-3}, NAN_WEIGHT},
                          {20, {1.46027370967713693e-3, 5.13914234815926810e-3, 5.54273763133126575e-3}, NAN_WEIGHT},
                          {33, {9.88651792384198473e-3, 5.67030775291526169e-3, 9.76182423499128045e-4}, NAN_WEIGHT}},
                         {{5,
                           0.3,
                           -0.4,
                           {6.37694626249547647e-1, 2.50572366820392745e-1, 3.41142996156192868e-1},
                           9.93290900272480015e-1,
                           {4.01553594934456637e-1, 3.21005369398807028e-1, 8.57735543866726791e-1}},
                          {10,
                           -0.7,
                           0.2,
                           {4.31647038714609577e-1, 4.94953004004851649e-1, 2.80062389039089366e-1},
                           9.92478382043001428e-1,
                           {2.75161216842846112e-1, 6.43994518589870113e-1, 7.13832868934854758e-1}}},
                         {{7,
                           {{{3.20775659483078290e-1, 3.36052337835417436e-1, 4.06398131032330967e-1},
                             {-3.17818766560982804e-2, 1.48651797382136110e-2, -1.73472347597680709e-18},
                             {-9.88703347838326349e-5, -1.03588435311721509e-4, -3.77926768971322165e-4}}},
                           {9.99187329044929351e-1, -1.73472347597680709e-17, 1.62534191014432339e-4}},
                          {20,
                           {{{1.55521217268639655e-1, 5.47300728225421973e-1, 2.97102434194900344e-1},
                             {-5.15995667081822762e-2, 7.18465437878236177e-3, -1.30104260698260532e-17},
                             {-1.44503428818529983e-4, -5.08559059861000279e-4, -6.41961609732666187e-4}}},
                           {9.98408573041862368e-1, 0.0, 3.18285391627825404e-4}}},
                         2.40966467180637363e-2);
    compareWithReference(
        runner, "sphere, triangles",
        read(geo::cellMesh(
            4, 4, [](double s, double t) { return spherePoint(1.0, s, t); }, CellPattern::FirstDiagonal)),
        56,
        {{0, {5.67484766811277800e-3, 3.83992132140978977e-9, 1.10216862829890846e-2}, NAN_WEIGHT},
         {5, {3.19349662379657395e-3, 2.13382622428825953e-3, 5.38995450686983579e-3}, 9.93403499413136881e-1},
         {18, {9.83918483175196153e-3, 2.18849085177872998e-3, 7.46855917052073170e-3}, 9.87602941530663414e-1},
         {40, {3.53241948770563995e-3, 1.61523266586146817e-2, 5.46838198981384159e-3}, 9.82838166445272554e-1}},
        {{9,
          -0.4,
          -0.2,
          {6.28513686896942536e-1, 1.86049334183526510e-1, 7.55185879344303035e-1},
          9.91621042625167926e-1,
          {6.28879861897323589e-1, 1.85729805521754526e-1, 7.54993085160953314e-1}},
         {20,
          0.1,
          -0.6,
          {5.97047178273831913e-1, 7.02937527475869905e-1, 3.86584602891503426e-1},
          9.90037052560321928e-1,
          {5.97232458282558110e-1, 7.02975179290126984e-1, 3.86185561713260350e-1}}},
        {{5,
          {{{4.80395827213676074e-1, 3.20990229489748979e-1, 8.10807700461423475e-1},
            {-2.12359710545909144e-2, 3.17818766560982874e-2, 3.46944695195361419e-18},
            {-2.12195254489829475e-4, -1.41784336117945810e-4, -3.58141217294943512e-4}}},
          {9.97801166471045775e-1, -3.81639164714897561e-17, 4.39766705791180129e-4}},
         {18,
          {{{7.81487854222462697e-1, 1.55599923947559393e-1, 5.90273630249646741e-1},
            {-1.02637919696891098e-2, 5.15995667081822831e-2, -2.08166817117216851e-17},
            {-6.52447151316232557e-4, -1.60244764602479779e-4, -4.97675591434546677e-4}}},
          {9.95867647176888138e-1, -3.81639164714897561e-17, 8.26470564622754089e-4}}},
        1.41341370308883116e-3);
    compareWithReference(
        runner, "ellipsoid, quadrilaterals and triangles",
        read(geo::cellMesh(4, 4, ellipsoidPoint, CellPattern::Mixed)), 48,
        {{0, {1.55846678696615662e-3, -3.95351430046883371e-6, 5.67962474272996464e-3}, NAN_WEIGHT},
         {6, {1.21197739689386948e-3, 1.69506633583658387e-3, 6.35404012785778429e-3}, 9.93681257435371279e-1},
         {19, {5.04300811764245120e-3, 8.42264579933495530e-3, 9.79325038611657608e-3}, 9.80933750556403972e-1},
         {30, {1.14977879504893597e-2, 5.39033019160674309e-3, 6.96589604886615078e-3}, 9.82517437460688425e-1}},
        {{3,
          -0.4,
          -0.2,
          {2.81778919268156958e-1, 2.88346056613262403e-1, 4.33343099318142744e-1},
          9.95777299333155841e-1,
          {1.53608565699794364e-1, 3.18627826362965993e-1, 9.35350584973818067e-1}},
         {7,
          0.2,
          0.1,
          {6.83275814544969440e-1, 3.25433687851903630e-1, 2.81199429151010705e-1},
          9.98651747358774711e-1,
          {4.71957172495201249e-1, 4.57110199246390680e-1, 7.53861189527130326e-1}}},
        {{6,
          {{{3.17202531900035278e-1, 2.22602660457638213e-1, 4.43300850439492744e-1},
            {3.28531542588405001e-2, 2.29972079811883584e-2, -1.18366268266838354e-2},
            {-2.70113798205504074e-5, -1.31085591397095760e-4, -6.56494554393353795e-4}}},
          {9.97893752478457130e-1, -3.12250225675825277e-17, 4.21249504308877920e-4}}},
        2.62790145841775846e-2);
    compareWithReference(runner, "sphere of the reference test, quadrilaterals", read(reportSphere(false)), 40,
                         {{3, {5.62378407085557738e-2, 1.11864020316589305e-2, 0.0}, NAN_WEIGHT}},
                         {{4,
                           0.1,
                           0.2,
                           {6.17066949925477370e0, 1.48224003234221136e0, 7.72824704687220621e0},
                           9.86950400777674486e-1,
                           {6.17066949925484831e-1, 1.48224003234221829e-1, 7.72824704687214759e-1}}},
                         {}, 2.58095682795178505e-8);
    compareWithReference(
        runner, "sphere of the reference test, triangles", read(reportSphere(true)), 56,
        {{3, {2.98386568190322499e-2, 1.23595763374395734e-2, 7.13617580215432229e-2}, 9.92197667229329006e-1}},
        {{4,
          0.1,
          0.2,
          {2.92210522081122770e0, 4.62181726782033309e0, 8.37258152234086239e0},
          9.99785521018330736e-1,
          {2.92208802610506191e-1, 4.62178251673121154e-1, 8.37260580319718062e-1}}},
        {}, 1.70484447238783096e-3);
    compareWithReference(
        runner, "cylinder, alternate diagonals",
        read(geo::cellMesh(
            4, 4, [](double s, double t) { return Vector3{s, std::sin(0.5 * PI * t), std::cos(0.5 * PI * t)}; },
            CellPattern::AlternateDiagonals)),
        56,
        {{0, {0.0, 0.0, 0.0}, 1.0},
         {9, {0.0, 1.61329540868281746e-2, 1.07796952873514713e-2}, 9.80785280403230542e-1},
         {25, {0.0, 1.61329540868281746e-2, 1.07796952873514713e-2}, 9.80785280403230542e-1}},
        {{11,
          -0.3,
          -0.3,
          {3.37703395998539269e-1, 6.19260877693796341e-1, 7.85185306381817827e-1},
          9.91929817769356847e-1,
          {0.0, 6.19260877693796341e-1, 7.85185306381817827e-1}}},
        {{9,
          {{{2.48398773366935915e-1, 8.20818642000126664e-1, 5.48453482190413344e-1},
            {0.0, 3.61287918874565600e-2, -5.40705581369096550e-2},
            {3.20245326612891498e-4, -1.06509703024163654e-3, -7.11675082918744074e-4}}},
          {9.93595093467743662e-1, 0.0, 1.28098130645156599e-3}}},
        2.98023223876953125e-8);
}

// The kinks of the patches are those that the reference reads on the isoparametric map of its elements at k = 4
// (Analysis::geometry_report, commit 9f55f0b): on these meshes the map is the patch.
CHECK_CASE("geometry: report against the reference")
{
    const ReadMesh sphereQuads = read(reportSphere(false));
    const ReadMesh sphereTriangles = read(reportSphere(true));
    const ReadMesh ellipsoid = read(geo::cellMesh(4, 4, ellipsoidPoint, CellPattern::Quadrilaterals));
    const auto kinkOf = [](const ReadMesh& m)
    { return geo::largestKink(mh::geometryReport(m.input, m.topology, m.surfaces, REPORT_POINTS)); };
    runner.expect(std::abs(kinkOf(sphereQuads) - 2.58095682795178505e-8) < 1e-12, "sphere, quadrilaterals");
    runner.expect(std::abs(kinkOf(sphereTriangles) - 1.70484447075978864e-3) < 1e-11, "sphere, triangles");
    runner.expect(std::abs(kinkOf(ellipsoid) - 2.40966467159763921e-2) < 1e-11, "ellipsoid, quadrilaterals");
}

// A strip of triangles one element wide on a cylinder, whose boundary goes on through the inner grids of its two arcs
// (as on the skin of a solid): the reading of the reference test strip_reading (tests/tetra.rs), against the
// reference.
CHECK_CASE("geometry: strip of triangles against the reference")
{
    const geo::QuadMesh cells = geo::mappedMesh(
        4, 1, [](double s, double t) { return Vector3{std::cos(s * 0.5 * PI), std::sin(s * 0.5 * PI), 0.5 * t}; });
    mh::MeshReadingInput input;
    input.grids = cells.grids;
    input.shells = geo::cellShells(1, cells.quads, CellPattern::AlternateDiagonals);
    std::set<std::size_t> through;
    for (std::size_t g = 0; g < input.grids.size(); ++g)
        if (g / 2 > 0 && g / 2 < 4)
            through.insert(g);
    input.throughGrids = through;
    const mh::Topology topology = mh::Topology::fromMesh(input);
    constexpr double NAN_WEIGHT = std::numeric_limits<double>::quiet_NaN();
    const std::vector<EdgeProbe> expected{
        {0, {1.90301168721782772e-2, 3.78532559855658188e-3, 0.0}, NAN_WEIGHT},
        {1, {0.0, 0.0, 0.0}, 1.0},
        {2, {1.90301168721782946e-2, 3.78532559855655933e-3, 0.0}, 9.80785280403230542e-1},
        {5, {1.61329540868281260e-2, 1.07796952873515772e-2, 0.0}, NAN_WEIGHT},
        {6, {1.61329540868281850e-2, 1.07796952873514731e-2, 0.0}, 9.80785280403230431e-1},
        {9, {1.07796952873515269e-2, 1.61329540868281503e-2, 0.0}, NAN_WEIGHT},
        {14, {3.78532559855656670e-3, 1.90301168721783015e-2, 0.0}, 9.80785280403230542e-1},
        {16, {3.78532559855657320e-3, 1.90301168721782980e-2, 0.0}, NAN_WEIGHT}};
    runner.expect(topology.edges.size() == 17, "edge count");
    for (const EdgeProbe& e : expected)
    {
        runner.expect(relativeDifference(topology.bulges[e.edge], e.bulge) < REFERENCE_TOLERANCE,
                      "bulge of edge " + std::to_string(e.edge));
        const double w = topology.weights[e.edge];
        runner.expect(std::isnan(e.weight) ? std::isnan(w) : std::abs(w - e.weight) < REFERENCE_TOLERANCE,
                      "weight of edge " + std::to_string(e.edge));
    }
}

// The common frames of the edges of a warped plate whose external edges hold translations at both grids: the frame
// turns about the edge to the held direction (fixed_normal_direction), against the reference.
CHECK_CASE("geometry: edge frames with held translations against the reference")
{
    mh::MeshReadingInput input =
        geo::cellMesh(2, 2, [](double s, double t) { return Vector3{s, t, 0.1 * s * t}; }, CellPattern::Quadrilaterals);
    for (const auto& [grid, components] : std::vector<std::pair<std::size_t, std::vector<int>>>{
             {0, {2}}, {1, {2}}, {2, {0, 2}}, {5, {0, 2}}, {3, {1, 2}}, {6, {1, 2}}})
        for (const int c : components)
            input.heldTranslations.insert({grid, c});
    const mh::Topology topology = mh::Topology::fromMesh(input);
    const std::vector<std::array<Vector3, 2>> expected{
        {{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}},
        {{{-9.99688424013585819e-1, 1.24649622981939999e-3, -2.49299245963880006e-2},
          {-2.49610675499375503e-2, -4.99220575816337747e-2, 9.98441151632675439e-1}}},
        {{{-1.24649622981939999e-3, 9.99688424013585819e-1, 2.49299245963880006e-2},
          {-4.99220575816337747e-2, -2.49610675499375503e-2, 9.98441151632675439e-1}}},
        {{{-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}},
        {{{-9.97206256315086303e-1, 3.73020372975288233e-3, -7.46040745950576484e-2},
          {-7.46972714766100943e-2, -4.97981040419156057e-2, 9.95962080838311947e-1}}},
        {{{0.0, 1.00000000000000022e0, 0.0}, {-9.95037190209989292e-2, 0.0, 9.95037190209989264e-1}}},
        {{{-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}},
        {{{0.0, 9.97199309888456398e-1, 7.47899482416342354e-2},
          {0.0, -7.47899482416342354e-2, 9.97199309888456398e-1}}},
        {{{-9.99689201594490151e-1, 2.48061836623942981e-3, -2.48061836623942981e-2},
          {-2.49299060441743742e-2, -9.94727934237848721e-2, 9.94727934237848666e-1}}},
        {{{-3.73020372975288233e-3, 9.97206256315086303e-1, 7.46040745950576484e-2},
          {-4.97981040419156057e-2, -7.46972714766100943e-2, 9.95962080838311947e-1}}},
        {{{-9.97213202565625068e-1, 7.42342334416098114e-3, -7.42342334416098149e-2},
          {-7.46044812917431438e-2, -9.92264223121204214e-2, 9.92264223121204103e-1}}},
        {{{-7.42342334416098114e-3, 9.97213202565625068e-1, 7.42342334416098149e-2},
          {-9.92264223121204214e-2, -7.46044812917431438e-2, 9.92264223121204103e-1}}}};
    runner.expect(topology.edges.size() == expected.size(), "edge count");
    for (std::size_t e = 0; e < expected.size() && e < topology.edges.size(); ++e)
    {
        runner.expect(distance(topology.frames[e][1], expected[e][0]) < REFERENCE_TOLERANCE,
                      "b of edge " + std::to_string(e));
        runner.expect(distance(topology.frames[e][2], expected[e][1]) < REFERENCE_TOLERANCE,
                      "n of edge " + std::to_string(e));
        runner.expect(distance(mh::cross3(topology.frames[e][2], topology.frames[e][0]), topology.frames[e][1]) < 1e-15,
                      "b = n x e");
    }
}

// The skin of a tube sector of tetrahedra with the edges of its block declared sharp: its faces, its smooth parts,
// the conics of its edges (exact circular arcs on the bore and the outer face) and the patch of a face, against the
// reference (a replication of src/tdnns/solid.rs::read_skin with the crate at commit 9f55f0b).
CHECK_CASE("geometry: skin of a tube of tetrahedra")
{
    const std::size_t nr = 1;
    const std::size_t nt = 4;
    const std::size_t nz = 1;
    const double ri = 1.0;
    const double ro = 2.0;
    const geo::TetMesh tets =
        geo::tetMesh(nr, nt, nz,
                     [&](double s, double t, double w)
                     {
                         const double r = ri + s * (ro - ri);
                         return Vector3{r * std::cos(t * 0.5 * PI), r * std::sin(t * 0.5 * PI), 0.5 * w};
                     });
    std::vector<mh::SolidCell> solids;
    for (const auto& t : tets.tets)
        solids.push_back({{t[0], t[1], t[2], t[3]}, {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}});
    std::set<mh::EdgeKey> sharp;
    for (const auto& [a, b] : geo::tetMeshCreases(nr, nt, nz))
        sharp.insert(mh::EdgeKey::of(a, b));
    const std::vector<mh::SkinFace> faces = mh::skinFaces(tets.grids, solids);
    runner.expect(faces.size() == 36, "faces of the skin: " + std::to_string(faces.size()));
    const mh::SkinReading reading = mh::readSolidSkin(tets.grids, faces, sharp);
    runner.expect(reading.mesh.grids.size() == 48,
                  "grids of the smooth parts: " + std::to_string(reading.mesh.grids.size()));
    if (faces.size() != 36)
        return;
    runner.expect(
        faces[0].view().size() == 3 && faces[0].grids[0] == 10 && faces[0].grids[1] == 12 && faces[0].grids[2] == 13,
        "first face");
    // every face turns about the outward normal: away from its solid's centroid
    for (const mh::SkinFace& f : faces)
    {
        Vector3 centre{};
        for (const std::size_t g : solids[f.solid].corners)
            centre = mh::add3(centre, mh::scl3(tets.grids[g], 0.25));
        const Vector3& x0 = tets.grids[f.grids[0]];
        const Vector3 n = mh::cross3(mh::sub3(tets.grids[f.grids[1]], x0), mh::sub3(tets.grids[f.grids[2]], x0));
        runner.expect(mh::dot3(n, mh::sub3(x0, centre)) > 0.0, "outward face");
    }
    struct ConicProbe
    {
        std::size_t a;
        std::size_t b;
        Vector3 bulge;
        double weight;
    };
    const std::vector<ConicProbe> expected{
        {0, 2, {1.90301168721782807e-2, 3.78532559855658361e-3, 0.0}, 9.80785280403230542e-1},
        {2, 4, {1.61329540868281746e-2, 1.07796952873514783e-2, 0.0}, 9.80785280403230431e-1},
        {2, 5, {1.61329540868281954e-2, 1.07796952873514332e-2, 0.0}, 9.80785280403230431e-1},
        {3, 5, {1.61329540868281572e-2, 1.07796952873515078e-2, 0.0}, 9.80785280403230431e-1},
        {0, 10, {0.0, 0.0, 0.0}, 1.0},
        {12, 13, {0.0, 0.0, 0.0}, 1.0},
        {13, 15, {3.22659081736563491e-2, 2.15593905747029566e-2, 0.0}, 9.80785280403230431e-1}};
    runner.expect(reading.conics.size() == 54, "conics: " + std::to_string(reading.conics.size()));
    for (const ConicProbe& c : expected)
    {
        const auto it = reading.conics.find(mh::EdgeKey::of(c.a, c.b));
        runner.expect(it != reading.conics.end(), "conic found");
        if (it == reading.conics.end())
            continue;
        runner.expect(relativeDifference(it->second.bulge, c.bulge) < REFERENCE_TOLERANCE, "bulge of a conic");
        runner.expect(std::abs(it->second.weight - c.weight) < REFERENCE_TOLERANCE, "weight of a conic");
    }
    // the hoops of the bore and of the outer face are circular arcs
    for (const auto& [edge, conic] : reading.conics)
    {
        const Vector3& xa = tets.grids[edge.lower];
        const Vector3& xb = tets.grids[edge.higher];
        const double ra = std::hypot(xa[0], xa[1]);
        if (std::abs(ra - std::hypot(xb[0], xb[1])) > 1e-12 || xa[2] != xb[2])
            continue;
        for (const double tau : {0.25, 0.5})
        {
            const Vector3 x = conicPoint(xa, xb, conic.bulge, conic.weight, tau);
            runner.expect(std::abs(std::hypot(x[0], x[1]) - ra) < 1e-12, "circular hoop");
        }
    }
    const std::array<std::pair<std::size_t, Vector3>, 2> facePoints{
        {{0, {1.93511089919809587e0, 5.05317531661764296e-1, 1.50305300987806367e-1}},
         {5, {9.62397512000045308e-1, 2.71645042097445000e-1, 1.75406791997078398e-1}}}};
    for (const auto& [face, point] : facePoints)
        runner.expect(
            relativeDifference(mh::skinFacePatch(reading, face).evaluate(-0.3, -0.4), point) < REFERENCE_TOLERANCE,
            "patch of face " + std::to_string(face));
}

// The supports read along an edge: stated on edges, they decide alone; otherwise an external edge holds what both its
// grids hold, an internal edge nothing.
CHECK_CASE("geometry: edge supports")
{
    runner.expect(mh::EdgeSupport::clamped().components() == std::vector<int>{0, 1, 2, 3, 4, 5}, "clamped");
    runner.expect(mh::EdgeSupport::pinned().components() == std::vector<int>{0, 1, 2}, "pinned");
    runner.expect(mh::EdgeSupport::simple(1).components() == std::vector<int>{0, 1, 2, 4}, "simple");
    runner.expect(mh::EdgeSupport::symmetry(0).components() == std::vector<int>{0, 4, 5}, "symmetry");
    runner.expect(mh::EdgeSupport::antisymmetry(2).components() == std::vector<int>{0, 1, 5}, "antisymmetry");
    const std::array<bool, 6> lower{true, true, false, true, false, false};
    const std::array<bool, 6> higher{true, false, true, true, true, false};
    const mh::EdgeKey edge = mh::EdgeKey::of(4, 2);
    const mh::EdgeSupport external = mh::edgeHeld({}, edge, 1, lower, higher, true);
    runner.expect(external == mh::EdgeSupport{{true, false, false}, {true, false, false}}, "held at both grids");
    runner.expect(mh::edgeHeld({}, edge, 2, lower, higher, true) == mh::EdgeSupport{}, "internal edge free");
    runner.expect(mh::edgeHeld({}, edge, 1, lower, higher, false) == mh::EdgeSupport{}, "grid supports not read");
    std::map<mh::EdgeKey, mh::EdgeSupport> stated;
    mh::addEdgeSupport(stated, 2, 4, mh::EdgeSupport::symmetry(0));
    mh::addEdgeSupport(stated, 4, 2, mh::EdgeSupport::antisymmetry(1));
    runner.expect(
        mh::edgeHeld(stated, edge, 1, lower, higher, true) == mh::EdgeSupport{{true, false, true}, {false, true, true}},
        "stated supports add up");
    runner.expect(mh::edgeHeld(stated, mh::EdgeKey::of(1, 2), 1, lower, higher, true) == mh::EdgeSupport{},
                  "an edge without a stated support is free");
}

// The edges of the line elements that no shell has come after those of the shells, with no shell, straight, and a
// frame whose normal is the orientation of the section across the edge (any direction across it for a rod); a line on
// an edge of a shell adds no edge.
CHECK_CASE("geometry: edges of line elements")
{
    mh::MeshReadingInput input;
    input.grids = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
                   {0.0, 1.0, 0.0}, {1.0, 1.0, 2.0}, {3.0, 1.0, 2.0}};
    input.shells.push_back(mh::ShellCorners::quadrilateral(0, 1, 2, 3));
    input.lines.push_back({{2, 4}, Vector3{1.0, 0.0, 0.3}});
    input.lines.push_back({{4, 5}, std::nullopt});
    input.lines.push_back({{1, 2}, Vector3{0.0, 0.0, 1.0}});
    const mh::Topology topology = mh::Topology::fromMesh(input);
    runner.expect(topology.edges.size() == 6, "four shell edges and two line edges");
    if (topology.edges.size() != 6)
        return;
    runner.expect(topology.edges[4] == mh::EdgeKey::of(2, 4) && topology.edges[5] == mh::EdgeKey::of(4, 5),
                  "line edges last");
    for (std::size_t e = 4; e < 6; ++e)
    {
        runner.expect(topology.edgeElements[e].empty(), "no shell");
        runner.expect(mh::norm3(topology.bulges[e]) == 0.0 && std::isnan(topology.weights[e]), "straight");
        const std::array<Vector3, 3>& f = topology.frames[e];
        runner.expect(std::abs(mh::dot3(f[0], f[2])) < 1e-15 && distance(mh::cross3(f[2], f[0]), f[1]) < 1e-15,
                      "orthonormal frame");
    }
    runner.expect(distance(topology.frames[4][0], {0.0, 0.0, 1.0}) < 1e-15, "e along the line");
    runner.expect(distance(topology.frames[4][2], {1.0, 0.0, 0.0}) < 1e-15, "n along the orientation");
    // a rod along x: n across it from the y axis
    runner.expect(distance(topology.frames[5][2], {0.0, 1.0, 0.0}) < 1e-15, "n of a rod");
}
