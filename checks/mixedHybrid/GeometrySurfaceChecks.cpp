#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/AnalyticSurfaces.hpp"
#include "oos/elements/mixedHybrid/geometry/ElementSurface.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/Patch.hpp"
#include "oos/elements/mixedHybrid/geometry/RationalPatch.hpp"
#include "oos/elements/mixedHybrid/geometry/Surface.hpp"
#include "oos/elements/mixedHybrid/geometry/TrianglePatch.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <string>

namespace mh = oos::elements::mixedHybrid;

namespace
{

using mh::Vector3;

constexpr double NAN_WEIGHT = std::numeric_limits<double>::quiet_NaN();

double distance(const Vector3& a, const Vector3& b)
{
    return mh::norm3(mh::sub3(a, b));
}

// The largest relative difference of two vectors, relative to the largest component of the expected one (at least 1).
double relativeDifference(const Vector3& value, const Vector3& expected)
{
    double scale = 1.0;
    for (const double e : expected)
        scale = std::fmax(scale, std::abs(e));
    return distance(value, expected) / scale;
}

// The patch of the probe P1 of the reference: a distorted, non-planar quadrilateral with curved edges, one edge with
// a given weight.
mh::RationalPatch probeQuadrilateral()
{
    const std::array<Vector3, 4> x{{{0.0, 0.0, 0.0}, {2.0, 0.1, 0.3}, {2.2, 2.5, 0.6}, {-0.1, 2.0, 0.2}}};
    const std::array<Vector3, 4> bulge{{{0.0, -0.05, 0.12}, {0.08, 0.0, 0.1}, {0.0, 0.06, 0.15}, {-0.04, 0.0, 0.09}}};
    return mh::RationalPatch::fromWeightedEdges(x, bulge, {NAN_WEIGHT, 0.93, NAN_WEIGHT, NAN_WEIGHT});
}

// The triangle of the probe P2 of the reference.
mh::TrianglePatch probeTriangle()
{
    const std::array<Vector3, 3> x{{{0.0, 0.0, 0.0}, {2.0, 0.2, 0.3}, {0.3, 1.8, 0.4}}};
    const std::array<Vector3, 3> bulge{{{0.0, -0.05, 0.12}, {0.07, 0.05, 0.1}, {-0.06, 0.0, 0.11}}};
    return mh::TrianglePatch::fromEdges(x, bulge, {NAN_WEIGHT, 0.97, NAN_WEIGHT});
}

// The bulge of the circular arc of half angle α/2 on a chord of a circle of radius r: the control point is where the
// two end tangents meet, at the height (L/2) tan(α/2) above the middle of the chord, L = 2 r sin(α/2).
double arcBulgeLength(double r, double alpha)
{
    const double chord = 2.0 * r * std::sin(0.5 * alpha);
    return 0.25 * chord * std::tan(0.5 * alpha);
}

} // namespace

// Port of the reference test sphere_curvature (src/surface.rs): on a sphere the curvature E^T dn/ds is ±(1/R) I.
CHECK_CASE("geometry: sphere curvature")
{
    mh::SpherePatch s;
    s.centre = {0.0, 0.0, 0.0};
    s.axis = {0.0, 0.0, 1.0};
    s.e1 = {1.0, 0.0, 0.0};
    s.e2 = {0.0, 1.0, 0.0};
    s.radius = 10.0;
    s.phi = {0.5, 0.8, 0.8, 0.5};
    s.theta = {0.0, 0.0, 0.3, 0.3};
    const mh::SurfacePointGeometry g = mh::pointGeometry(s, 0.3, -0.2);
    const mh::PhysicalGeometry p = mh::physicalGeometry(g, {1.0, 0.0, 0.0});
    const double k00 = mh::dot3(p.frame[0], p.normalGradient[0]);
    const double k11 = mh::dot3(p.frame[1], p.normalGradient[1]);
    const double k01 = mh::dot3(p.frame[0], p.normalGradient[1]);
    runner.expect(std::abs(std::abs(k00) - 0.1) < 1e-12, "kappa_00 = 1/R");
    runner.expect(std::abs(k11 - k00) < 1e-12, "kappa_11 = kappa_00");
    runner.expect(std::abs(k01) < 1e-12, "kappa_01 = 0");
    runner.expect(std::abs(mh::norm3(g.normal) - 1.0) < 1e-14, "unit normal");
}

// Port of the reference test arc_edges_reproduce_a_cylinder (src/surface.rs): four grids on a cylinder with the exact
// end slopes; the Coons patch of the edge arcs is the cylinder.
CHECK_CASE("geometry: arc edges reproduce a cylinder")
{
    const double r = 3.0;
    const double alpha = 0.4;
    const auto p = [&](double x, double a) { return Vector3{x, r * std::sin(a), r * std::cos(a)}; };
    const Vector3 b = mh::scl3(Vector3{0.0, std::sin(0.5 * alpha), std::cos(0.5 * alpha)}, arcBulgeLength(r, alpha));
    const mh::CoonsPatch q{{p(0.0, 0.0), p(1.0, 0.0), p(1.0, alpha), p(0.0, alpha)}, {Vector3{}, b, Vector3{}, b}};
    for (const auto& [u, v] : std::array<std::array<double, 2>, 3>{{{0.0, 0.0}, {0.3, -0.7}, {-0.9, 0.55}}})
    {
        const Vector3 x = q.evaluate(u, v);
        const double radius = std::sqrt(x[1] * x[1] + x[2] * x[2]);
        runner.expect(std::abs(radius - r) < 1e-14, "radius " + std::to_string(radius));
        const mh::SurfacePointGeometry g = mh::pointGeometry(q, u, v);
        runner.expect(std::abs(g.normal[1] * x[2] - g.normal[2] * x[1]) < 1e-13 && std::abs(g.normal[0]) < 1e-14,
                      "radial normal");
    }
}

// Port of the reference test offset_keeps_normal (src/surface.rs): the parallel surface keeps the normal and is at the
// offset along it.
CHECK_CASE("geometry: offset keeps the normal")
{
    const mh::CoonsPatch q{{{{0.0, 0.0, 0.0}, {2.0, 0.0, 0.1}, {2.2, 3.0, 0.0}, {0.0, 2.5, 0.2}}},
                           {{{0.0, 0.0, 0.1}, {0.0, 0.0, 0.0}, {0.0, 0.0, -0.05}, {0.0, 0.0, 0.0}}}};
    const mh::OffsetSurface<mh::CoonsPatch> o{q, 0.3};
    const mh::SurfacePointGeometry g0 = mh::pointGeometry(q, 0.2, 0.4);
    const mh::SurfacePointGeometry g1 = mh::pointGeometry(o, 0.2, 0.4);
    for (std::size_t i = 0; i < 3; ++i)
    {
        runner.expect(std::abs(g0.normal[i] - g1.normal[i]) < 1e-12, "normal kept");
        runner.expect(std::abs(g1.position[i] - g0.position[i] - 0.3 * g0.normal[i]) < 1e-13, "offset along it");
    }
}

// The rational patch of four grids of a cylinder and the circular arcs of its hoops is the cylinder (the weight of
// the centre is the product of the mean weights of the opposite edges), and the fit of the centre keeps it.
CHECK_CASE("geometry: rational patch of a cylinder is the cylinder")
{
    const double r = 2.5;
    const double alpha = 0.6;
    const auto p = [&](double x, double a) { return Vector3{x, r * std::sin(a), r * std::cos(a)}; };
    const Vector3 b = mh::scl3(Vector3{0.0, std::sin(0.5 * alpha), std::cos(0.5 * alpha)}, arcBulgeLength(r, alpha));
    mh::RationalPatch patch =
        mh::RationalPatch::fromEdges({p(0.0, 0.0), p(1.5, 0.0), p(1.5, alpha), p(0.0, alpha)}, {Vector3{}, b, {}, b});
    for (const bool fitted : {false, true})
    {
        if (fitted)
            patch.fitCentre();
        double worstRadius = 0.0;
        double worstNormal = 0.0;
        for (int i = 0; i <= 6; ++i)
            for (int j = 0; j <= 6; ++j)
            {
                const double u = -1.0 + i / 3.0;
                const double v = -1.0 + j / 3.0;
                const mh::SurfacePointFrame<double> f = mh::frame(patch, u, v);
                worstRadius = std::fmax(worstRadius, std::abs(std::hypot(f.position[1], f.position[2]) - r));
                const Vector3 radial = mh::unit3(Vector3{0.0, f.position[1], f.position[2]});
                worstNormal = std::fmax(worstNormal, mh::norm3(mh::cross3(f.normal, radial)));
            }
        const std::string which = fitted ? "fitted" : "not fitted";
        runner.expect(worstRadius < 1e-14 * r, which + ": on the cylinder " + std::to_string(worstRadius));
        runner.expect(worstNormal < 1e-14, which + ": normal radial " + std::to_string(worstNormal));
    }
}

// A flat element with straight edges is the bilinear patch, and the fits move nothing; a flat triangle with straight
// edges is the plane triangle.
CHECK_CASE("geometry: flat patches are bilinear and not moved by the fits")
{
    const std::array<Vector3, 4> x{{{0.0, 0.0, 0.0}, {2.0, 0.3, 0.0}, {2.4, 1.9, 0.0}, {-0.2, 1.5, 0.0}}};
    mh::RationalPatch patch = mh::RationalPatch::fromEdges(x, {});
    const mh::RationalPatch before = patch;
    patch.fitCentre();
    runner.expect(distance(patch.controlPoints[1][1], before.controlPoints[1][1]) < 1e-15, "centre not moved");
    for (const auto& [u, v] : std::array<std::array<double, 2>, 3>{{{0.1, 0.2}, {-0.8, 0.6}, {0.9, -0.95}}})
    {
        Vector3 bilinear{};
        for (std::size_t k = 0; k < 4; ++k)
        {
            const double nk = 0.25 * (1.0 + mh::REFERENCE_SQUARE[k][0] * u) * (1.0 + mh::REFERENCE_SQUARE[k][1] * v);
            bilinear = mh::add3(bilinear, mh::scl3(x[k], nk));
        }
        runner.expect(distance(patch.evaluate(u, v), bilinear) < 1e-14, "bilinear patch");
    }
    const std::array<Vector3, 3> t{{{0.0, 0.0, 0.0}, {2.0, 0.3, 0.0}, {0.4, 1.9, 0.0}}};
    for (const bool rational : {false, true})
    {
        mh::TrianglePatch tp = mh::TrianglePatch::fromEdges(t, {}, {NAN_WEIGHT, NAN_WEIGHT, NAN_WEIGHT});
        if (rational)
            tp.fitInteriorRational();
        else
            tp.fitInterior();
        for (const Vector3& c : tp.interior)
            runner.expect(mh::norm3(c) == 0.0, "no interior on a flat triangle");
        const std::array<double, 3> l = mh::barycentric(-0.3, -0.1);
        const Vector3 linear = mh::add3(mh::add3(mh::scl3(t[0], l[0]), mh::scl3(t[1], l[1])), mh::scl3(t[2], l[2]));
        runner.expect(distance(tp.evaluate(-0.3, -0.1), linear) < 1e-14, "plane triangle");
    }
}

// The edges of the patches are the edge conics: the point of edge k at τ is the conic of its grids, control point and
// weight, from both the quadrilateral and the triangle.
CHECK_CASE("geometry: patch edges are the edge conics")
{
    const auto conic = [](const Vector3& xa, const Vector3& p, const Vector3& xb, double w, double tau)
    {
        const double b0 = (1.0 - tau) * (1.0 - tau);
        const double b1 = 2.0 * w * tau * (1.0 - tau);
        const double b2 = tau * tau;
        Vector3 out{};
        for (std::size_t c = 0; c < 3; ++c)
            out[c] = (b0 * xa[c] + b1 * p[c] + b2 * xb[c]) / (b0 + b1 + b2);
        return out;
    };
    mh::RationalPatch quad = probeQuadrilateral();
    quad.fitCentre();
    constexpr std::array<std::array<std::size_t, 2>, 4> SLOT{{{1, 0}, {2, 1}, {1, 2}, {0, 1}}};
    for (std::size_t k = 0; k < 4; ++k)
        for (const double tau : {0.2, 0.5, 0.85})
        {
            const mh::EdgePoint<double> e = mh::edgeEvaluation(quad, k, tau);
            const auto [i, j] = SLOT[k];
            const Vector3 expected =
                conic(quad.corners[k], quad.controlPoints[i][j], quad.corners[(k + 1) % 4], quad.weights[i][j], tau);
            runner.expect(distance(e.position, expected) < 1e-14, "quadrilateral edge " + std::to_string(k));
        }
    mh::TrianglePatch triangle = probeTriangle();
    triangle.fitInteriorRational();
    for (std::size_t k = 0; k < 3; ++k)
        for (const double tau : {0.2, 0.5, 0.85})
        {
            const mh::EdgePoint<double> e = mh::edgeEvaluation(triangle, k, tau);
            const Vector3 expected = conic(triangle.corners[k], triangle.controlPoints[k],
                                           triangle.corners[(k + 1) % 3], triangle.weights[k], tau);
            runner.expect(distance(e.position, expected) < 1e-14, "triangle edge " + std::to_string(k));
        }
}

// The derivatives of the frame (by nested duals) against central differences of the tangents and of the normal.
CHECK_CASE("geometry: point geometry derivatives")
{
    mh::RationalPatch patch = probeQuadrilateral();
    patch.fitCentre();
    const double u = 0.3;
    const double v = -0.4;
    const double h = 1e-5;
    const mh::SurfacePointGeometry g = mh::pointGeometry(patch, u, v);
    const std::array<std::array<double, 2>, 2> step{{{h, 0.0}, {0.0, h}}};
    for (std::size_t b = 0; b < 2; ++b)
    {
        const mh::SurfacePointGeometry gp = mh::pointGeometry(patch, u + step[b][0], v + step[b][1]);
        const mh::SurfacePointGeometry gm = mh::pointGeometry(patch, u - step[b][0], v - step[b][1]);
        for (std::size_t a = 0; a < 2; ++a)
        {
            const Vector3 fd = mh::scl3(mh::sub3(gp.tangents[a], gm.tangents[a]), 0.5 / h);
            runner.expect(distance(fd, g.tangentDerivatives[a][b]) < 1e-8, "da/dxi");
        }
        const Vector3 fd = mh::scl3(mh::sub3(gp.normal, gm.normal), 0.5 / h);
        runner.expect(distance(fd, g.normalDerivatives[b]) < 1e-8, "dn/dxi");
    }
    // the physical frame of the two element axes is orthonormal, and the inverse of the components is theirs
    const mh::PhysicalGeometry p = mh::physicalGeometryXy(g, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    runner.expect(std::abs(mh::dot3(p.frame[0], p.frame[1])) < 1e-15, "orthogonal frame");
    runner.expect(std::abs(mh::dot3(p.frame[0], g.normal)) < 1e-15, "tangent frame");
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
        {
            const double product = p.components[i][0] * p.inverse[0][j] + p.components[i][1] * p.inverse[1][j];
            runner.expect(std::abs(product - (i == j ? 1.0 : 0.0)) < 1e-14, "inverse of the components");
        }
}

// The denominator W of the rational map, as a polynomial, against the denominator of the patches; 1 at the corners.
CHECK_CASE("geometry: weight polynomial of the patches")
{
    mh::RationalPatch quad = probeQuadrilateral();
    quad.fitCentre();
    const mh::Polynomial2 wq = mh::weightPolynomial(quad);
    for (const auto& [u, v] : std::array<std::array<double, 2>, 3>{{{0.3, -0.4}, {-0.9, 0.1}, {0.7, 0.7}}})
    {
        const double expected = quad.homogeneous(0.5 * (1.0 + u), 0.5 * (1.0 + v)).denominator;
        runner.expect(std::abs(wq.evaluate(u, v) - expected) < 1e-15, "W of the quadrilateral");
    }
    for (const mh::ReferencePoint& c : mh::REFERENCE_SQUARE)
        runner.expect(std::abs(wq.evaluate(c[0], c[1]) - 1.0) < 1e-15, "W = 1 at a corner");
    const mh::TrianglePatch triangle = probeTriangle();
    const mh::Polynomial2 wt = mh::weightPolynomial(triangle);
    const std::array<double, 3> l = mh::barycentric(-0.2, -0.5);
    double expected = 0.0;
    for (std::size_t k = 0; k < 3; ++k)
        expected += l[k] * l[k] + 2.0 * triangle.weights[k] * l[k] * l[(k + 1) % 3];
    runner.expect(std::abs(wt.evaluate(-0.2, -0.5) - expected) < 1e-15, "W of the triangle");
    for (const mh::ReferencePoint& c : mh::REFERENCE_TRIANGLE)
        runner.expect(std::abs(wt.evaluate(c[0], c[1]) - 1.0) < 1e-15, "W = 1 at a corner of the triangle");
    mh::SpherePatch sphere;
    sphere.radius = 1.0;
    runner.expect(mh::weightPolynomial(sphere).evaluate(0.2, 0.1) == 1.0, "W = 1 on an analytic patch");
}

// Values of the reference crate at commit 9f55f0b (the probe P1 and P2): the centre control point fitted on a
// distorted curved quadrilateral and a point of its patch; the interiors of a curved triangle fitted inside and
// outside the rational form, and a point of each.
CHECK_CASE("geometry: patches against the reference")
{
    constexpr double TOLERANCE = 1e-12;
    mh::RationalPatch quad = probeQuadrilateral();
    runner.expectNear(quad.weights[1][1], 9.25092589779080487e-1, TOLERANCE, "centre weight");
    quad.fitCentre();
    runner.expect(relativeDifference(quad.controlPoints[1][1], {9.72370942000805982e-1, 1.10155037573820347e0,
                                                                8.06668614800438610e-1}) < TOLERANCE,
                  "fitted centre");
    runner.expect(relativeDifference(quad.evaluate(0.3, -0.4), {1.34072109282659047e0, 7.06017318829361140e-1,
                                                                4.80385930983617149e-1}) < TOLERANCE,
                  "patch point");
    mh::TrianglePatch inside = probeTriangle();
    mh::TrianglePatch outside = inside;
    inside.fitInteriorRational();
    outside.fitInterior();
    const std::array<Vector3, 3> expectedInside{
        {{-3.63817285070224747e-2, -5.58684929616916626e-2, 3.26535176147643658e-1},
         {-1.54431647936581103e-2, -2.37148255178199104e-2, 1.38606293409090808e-1},
         {-8.06616807099684932e-3, -1.23865652511624107e-2, 7.23958899146602275e-2}}};
    const std::array<Vector3, 3> expectedOutside{
        {{-3.68724133871218551e-2, -5.66219982484168147e-2, 3.30939196526315538e-1},
         {-1.56377087109123361e-2, -2.40135709572995015e-2, 1.40352374062647389e-1},
         {-8.19927937089019558e-3, -1.25909735634221649e-2, 7.35905973554989989e-2}}};
    for (std::size_t m = 0; m < 3; ++m)
    {
        runner.expect(relativeDifference(inside.interior[m], expectedInside[m]) < TOLERANCE, "interior, rational form");
        runner.expect(relativeDifference(outside.interior[m], expectedOutside[m]) < TOLERANCE, "interior, outside");
    }
    runner.expect(relativeDifference(inside.evaluate(-0.2, -0.5), {8.82855104553119330e-1, 5.19647234630978505e-1,
                                                                   3.70740052285683686e-1}) < TOLERANCE,
                  "triangle point, rational form");
    runner.expect(relativeDifference(outside.evaluate(-0.2, -0.5), {8.82860599738097274e-1, 5.19655673144371333e-1,
                                                                    3.70690731616535374e-1}) < TOLERANCE,
                  "triangle point, outside");
}

// An explicit cylinder and sphere patch along their lines are read as the rational patch of their corners and edge
// arcs, which is the same surface; a patch that is not one of them is not.
CHECK_CASE("geometry: explicit surfaces as rational patches")
{
    mh::CylinderPatch cylinder;
    cylinder.centre = {0.1, -0.2, 0.3};
    cylinder.axis = {1.0, 0.0, 0.0};
    cylinder.e1 = {0.0, 1.0, 0.0};
    cylinder.e2 = {0.0, 0.0, 1.0};
    cylinder.radius = 2.0;
    cylinder.angle = {0.2, 0.2, 0.9, 0.9};
    cylinder.z = {0.0, 1.5, 1.5, 0.0};
    const std::optional<mh::Patch> c = mh::asRationalPatch(cylinder);
    runner.expect(c.has_value() && c->rational() != nullptr, "cylinder read as a rational patch");
    mh::SpherePatch sphere;
    sphere.axis = {0.0, 0.0, 1.0};
    sphere.e1 = {1.0, 0.0, 0.0};
    sphere.e2 = {0.0, 1.0, 0.0};
    sphere.radius = 3.0;
    sphere.phi = {0.5, 0.5, 0.9, 0.9};
    sphere.theta = {0.1, 0.6, 0.6, 0.1};
    const std::optional<mh::Patch> s = mh::asRationalPatch(sphere);
    runner.expect(s.has_value(), "sphere read as a rational patch");
    if (c && s)
        for (const auto& [u, v] : std::array<std::array<double, 2>, 2>{{{0.25, -0.6}, {-0.5, 0.45}}})
        {
            const Vector3 xc = mh::sub3(c->evaluate(u, v), cylinder.centre);
            runner.expect(std::abs(std::hypot(xc[1], xc[2]) - 2.0) < 1e-12, "on the cylinder");
            runner.expect(std::abs(mh::norm3(s->evaluate(u, v)) - 3.0) < 1e-12, "on the sphere");
        }
    // a skew parametrisation of the sphere (the edges are not circles of it) is not a rational patch
    sphere.theta = {0.1, 0.6, 0.9, 0.1};
    runner.expect(!mh::asRationalPatch(sphere).has_value(), "skew sphere patch kept as it is");
}

// The element axes: the bisector of the diagonals of a quadrilateral and the side G1-G2 of a triangle; turning the
// numbering of a square by one grid turns its axes by a right angle.
CHECK_CASE("geometry: element axes")
{
    const std::array<Vector3, 4> square{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}}};
    const mh::ElementAxes a = mh::elementAxes(square);
    runner.expect(distance(a.axes[0], {1.0, 0.0, 0.0}) < 1e-15, "x_e of a square");
    runner.expect(distance(a.axes[1], {0.0, 1.0, 0.0}) < 1e-15, "y_e of a square");
    runner.expect(distance(a.axes[2], {0.0, 0.0, 1.0}) < 1e-15, "z_e of a square");
    const mh::ElementAxes turned = mh::elementAxes({square[1], square[2], square[3], square[0]});
    runner.expect(distance(turned.axes[0], {0.0, 1.0, 0.0}) < 1e-15, "turned by a right angle");
    const std::array<Vector3, 4> skew{{{0.0, 0.0, 0.0}, {2.0, 0.2, 0.1}, {2.5, 1.7, -0.1}, {0.2, 1.4, 0.05}}};
    const mh::ElementAxes b = mh::elementAxes(skew);
    // x_e makes equal angles with the diagonal G1-G3 and the reversed diagonal G2-G4, in the mean plane
    const auto inPlane = [&](const Vector3& d)
    { return mh::unit3(mh::sub3(d, mh::scl3(b.axes[2], mh::dot3(d, b.axes[2])))); };
    const Vector3 d1 = inPlane(mh::sub3(skew[2], skew[0]));
    const Vector3 d2 = inPlane(mh::sub3(skew[1], skew[3]));
    runner.expect(std::abs(mh::dot3(b.axes[0], d1) - mh::dot3(b.axes[0], d2)) < 1e-14, "bisector of the diagonals");
    runner.expect(std::abs(mh::dot3(b.axes[0], b.axes[2])) < 1e-15 && std::abs(mh::dot3(b.side, b.axes[2])) < 1e-15,
                  "axes in the mean plane");
    const mh::ElementAxes t = mh::elementAxesTriangle({skew[0], skew[1], skew[2]});
    runner.expect(distance(t.axes[0], mh::unit3(mh::sub3(skew[1], skew[0]))) < 1e-15, "x_e of a triangle");
    runner.expect(distance(t.side, t.axes[0]) == 0.0, "side of a triangle");
    runner.expect(mh::cornerThicknesses(3, {1.0, 2.0, 4.0, 0.0}) == std::array<double, 4>{1.0, 2.0, 5.0, 4.0},
                  "corner thicknesses of a triangle");
}
