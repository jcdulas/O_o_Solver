#include "oos/elements/mixedHybrid/geometry/ElementSurface.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"

#include <cmath>
#include <functional>
#include <limits>

namespace oos::elements::mixedHybrid
{

namespace
{

// The relative distance to the explicit surface below which its rational patch is the same surface.
constexpr double SAME_SURFACE_TOLERANCE = 1e-12;
// The points per direction of that comparison.
constexpr std::size_t SAME_SURFACE_POINTS = 7;
// Below this relative determinant the two end tangents of an edge are parallel: a straight edge.
constexpr double PARALLEL_END_TANGENTS = 1e-14;

// The local edge of shell `element` that is edge `edge` of the topology.
std::size_t localEdgeOf(const MeshReadingInput& input, const Topology& topology, std::size_t element, std::size_t edge)
{
    const ShellCorners& shell = input.shells[element];
    for (std::size_t k = 0; k < shell.count; ++k)
        if (EdgeKey::of(shell.at(k), shell.at(k + 1)) == topology.edges[edge])
            return k;
    throw base::InternalError("mesh reading: the element does not have the edge");
}

// The relative distance of a point from the explicit cylinder or sphere of a patch, or nothing for another kind.
std::function<double(const Vector3&)> implicitDistance(const Patch& patch)
{
    if (const CylinderPatch* c = std::get_if<CylinderPatch>(&patch.shape))
    {
        const Vector3 centre = c->centre;
        const Vector3 axis = unit3(c->axis);
        const double r = c->radius;
        return [=](const Vector3& x)
        {
            const Vector3 d = sub3(x, centre);
            return (norm3(sub3(d, scl3(axis, dot3(d, axis)))) - r) / r;
        };
    }
    if (const SpherePatch* s = std::get_if<SpherePatch>(&patch.shape))
    {
        const Vector3 centre = s->centre;
        const double r = s->radius;
        return [=](const Vector3& x) { return (norm3(sub3(x, centre)) - r) / r; };
    }
    return {};
}

} // namespace

ElementSurface elementSurface(const MeshReadingInput& input, const Topology& topology, std::size_t element,
                              const Patch* explicitSurface)
{
    const ShellCorners& shell = input.shells[element];
    const std::size_t nv = shell.count;
    std::array<Vector3, 4> bulge{};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 4> weight{nan, nan, nan, nan};
    bool facet = true;
    for (std::size_t k = 0; k < nv; ++k)
    {
        const std::size_t idx = topology.indexOf(shell.at(k), shell.at(k + 1));
        bulge[k] = topology.bulges[idx];
        weight[k] = topology.weights[idx];
        facet = facet && topology.isRaw(shell.at(k), shell.at(k + 1));
    }
    std::optional<Patch> given;
    if (explicitSurface != nullptr)
    {
        given = asRationalPatch(*explicitSurface);
        if (!given)
            given = *explicitSurface;
    }
    ElementSurface out;
    if (nv == 3)
    {
        const std::array<Vector3, 3> x{input.grids[shell.grids[0]], input.grids[shell.grids[1]],
                                       input.grids[shell.grids[2]]};
        if (given)
        {
            out.patch = *given;
        }
        else
        {
            TrianglePatch patch =
                TrianglePatch::fromEdges(x, {bulge[0], bulge[1], bulge[2]}, {weight[0], weight[1], weight[2]});
            // inside the rational form: the map is then exactly in the rational space of degree 4
            if (!facet)
                patch.fitInteriorRational();
            out.patch = patch;
        }
        const ElementAxes axes = elementAxesTriangle(x);
        out.axes = {axes.axes[0], axes.axes[1]};
        out.side = axes.side;
        return out;
    }
    const std::array<Vector3, 4> x{input.grids[shell.grids[0]], input.grids[shell.grids[1]],
                                   input.grids[shell.grids[2]], input.grids[shell.grids[3]]};
    if (given)
    {
        out.patch = *given;
    }
    else
    {
        RationalPatch patch = RationalPatch::fromWeightedEdges(x, bulge, weight);
        if (!facet)
            patch.fitCentre();
        out.patch = patch;
    }
    const ElementAxes axes = elementAxes(x);
    out.axes = {axes.axes[0], axes.axes[1]};
    out.side = axes.side;
    return out;
}

std::array<double, 4> cornerThicknesses(std::size_t cornerCount, const std::array<double, 4>& t)
{
    if (cornerCount == 3)
        return {t[0], t[1], t[1] + t[2] - t[0], t[2]};
    return t;
}

Polynomial2 weightPolynomial(const Patch& patch)
{
    if (const RationalPatch* rp = patch.rational())
    {
        // the Bernstein polynomials of s = (1 + ξ)/2 and t = (1 + η)/2
        const Polynomial2 s = Polynomial2::linearInXi(0.5, 0.5);
        const Polynomial2 t = Polynomial2::linearInEta(0.5, 0.5);
        const Polynomial2 one = Polynomial2::constant(1.0);
        const std::array<Polynomial2, 3> bs{(one - s) * (one - s), (s * (one - s)).scaled(2.0), s * s};
        const std::array<Polynomial2, 3> bt{(one - t) * (one - t), (t * (one - t)).scaled(2.0), t * t};
        Polynomial2 w;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                w = w + (bs[i] * bt[j]).scaled(rp->weights[i][j]);
        return w;
    }
    if (const TrianglePatch* tp = patch.triangle())
    {
        // the barycentric coordinates
        const std::array<Polynomial2, 3> l{Polynomial2::linearInXi(0.0, -0.5) + Polynomial2::linearInEta(0.0, -0.5),
                                           Polynomial2::linearInXi(0.5, 0.5), Polynomial2::linearInEta(0.5, 0.5)};
        Polynomial2 w;
        for (std::size_t k = 0; k < 3; ++k)
            w = w + l[k] * l[k] + (l[k] * l[(k + 1) % 3]).scaled(2.0 * tp->weights[k]);
        return w;
    }
    return Polynomial2::constant(1.0);
}

std::optional<Patch> asRationalPatch(const Patch& patch)
{
    const std::function<double(const Vector3&)> distance = implicitDistance(patch);
    if (!distance)
        return std::nullopt;
    std::array<Vector3, 4> x{};
    for (std::size_t k = 0; k < 4; ++k)
        x[k] = patch.evaluate(REFERENCE_SQUARE[k][0], REFERENCE_SQUARE[k][1]);
    std::array<Vector3, 4> bulge{};
    for (std::size_t k = 0; k < 4; ++k)
    {
        const ReferencePoint a = REFERENCE_SQUARE[k];
        const ReferencePoint b = REFERENCE_SQUARE[(k + 1) % 4];
        const std::array<double, 2> d{b[0] - a[0], b[1] - a[1]};
        const auto tangentAt = [&](const ReferencePoint& p)
        {
            const std::array<Vector3, 2> t = tangents(patch, p[0], p[1]).tangents;
            return add3(scl3(t[0], d[0]), scl3(t[1], d[1]));
        };
        const Vector3 ta = tangentAt(a);
        const Vector3 tb = tangentAt(b);
        const Vector3& xa = x[k];
        const Vector3& xb = x[(k + 1) % 4];
        // the control point: X_a + la t_a = X_b + lb t_b (least squares on the two lines)
        const Vector3 c = sub3(xb, xa);
        const std::array<std::array<double, 2>, 2> m{{{dot3(ta, ta), -dot3(ta, tb)}, {-dot3(ta, tb), dot3(tb, tb)}}};
        const std::array<double, 2> rhs{dot3(c, ta), -dot3(c, tb)};
        const double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if (std::abs(det) < PARALLEL_END_TANGENTS * m[0][0] * m[1][1])
            continue; // a straight edge
        const double la = (rhs[0] * m[1][1] - m[0][1] * rhs[1]) / det;
        const Vector3 control = add3(xa, scl3(ta, la));
        bulge[k] = scl3(sub3(control, scl3(add3(xa, xb), 0.5)), 0.5);
    }
    RationalPatch rational = RationalPatch::fromEdges(x, bulge);
    rational.fitCentre();
    double worst = 0.0;
    for (std::size_t i = 0; i < SAME_SURFACE_POINTS; ++i)
        for (std::size_t j = 0; j < SAME_SURFACE_POINTS; ++j)
        {
            const double u = -1.0 + static_cast<double>(i) / 3.0;
            const double v = -1.0 + static_cast<double>(j) / 3.0;
            worst = std::fmax(worst, std::abs(distance(rational.evaluate(u, v))));
        }
    if (worst < SAME_SURFACE_TOLERANCE)
        return Patch(rational);
    return std::nullopt;
}

std::vector<EdgeMoments> edgeMoments(const MeshReadingInput& input, const Topology& topology,
                                     std::span<const ElementSurface> surfaces, std::span<const Polynomial2> weights,
                                     std::size_t degree, std::size_t rulePoints)
{
    const std::vector<LineRulePoint> rule = unitIntervalRule(rulePoints);
    const std::size_t momentCount = degree - 1;
    std::vector<EdgeMoments> out(topology.edges.size());
    for (std::size_t e = 0; e < topology.edges.size(); ++e)
    {
        if (topology.edgeElements[e].empty())
            continue;
        // read on the first element of the edge, from the lower to the higher grid
        const std::size_t ie = topology.edgeElements[e].front();
        const ShellCorners& shell = input.shells[ie];
        const std::size_t kk = localEdgeOf(input, topology, ie, e);
        const bool forward = shell.at(kk) < shell.at(kk + 1);
        EdgeMoments moments{std::vector<Vector3>(momentCount, Vector3{}), std::vector<double>(momentCount, 0.0)};
        for (const LineRulePoint& r : rule)
        {
            const double tl = forward ? r.point : 1.0 - r.point;
            const ReferencePoint x = referenceEdgePointAndDirection(shell.count, kk, tl).first;
            const Vector3 xv = surfaces[ie].patch.evaluate(x[0], x[1]);
            const double wv = weights[ie].evaluate(x[0], x[1]);
            for (std::size_t l = 0; l < momentCount; ++l)
            {
                const double f = r.weight * legendreValue(l, 2.0 * r.point - 1.0);
                moments.weight[l] += f * wv;
                for (std::size_t c = 0; c < 3; ++c)
                    moments.position[l][c] += f * xv[c] * wv;
            }
        }
        out[e] = std::move(moments);
    }
    return out;
}

std::vector<EdgeGeometry> geometryReport(const MeshReadingInput& input, const Topology& topology,
                                         std::span<const double> tau, const EdgeSurfaceEvaluator& point)
{
    std::vector<EdgeGeometry> out;
    out.reserve(topology.edges.size());
    for (std::size_t e = 0; e < topology.edges.size(); ++e)
    {
        const EdgeKey edge = topology.edges[e];
        const std::vector<std::size_t>& owners = topology.edgeElements[e];
        double kink = 0.0;
        if (owners.size() >= 2)
        {
            // the points and normals of each element on the edge, in its own direction
            std::vector<std::vector<EdgeSurfacePoint>> normals;
            for (const std::size_t ie : owners)
            {
                const std::size_t kk = localEdgeOf(input, topology, ie, e);
                std::vector<EdgeSurfacePoint> list;
                for (const double t : tau)
                    list.push_back(point(ie, kk, t));
                normals.push_back(std::move(list));
            }
            for (std::size_t i = 0; i < owners.size(); ++i)
                for (std::size_t j = i + 1; j < owners.size(); ++j)
                    for (const auto& [xi, ni] : normals[i])
                    {
                        // the closest point of the other element on the edge (the first of equal ones)
                        const EdgeSurfacePoint* closest = &normals[j].front();
                        for (const EdgeSurfacePoint& p : normals[j])
                            if (norm3(sub3(p.first, xi)) < norm3(sub3(closest->first, xi)))
                                closest = &p;
                        // the angle between the two surfaces, whatever their orientation
                        const double c = std::fmin(std::abs(dot3(ni, closest->second)), 1.0);
                        kink = std::fmax(kink, std::acos(c));
                    }
        }
        out.push_back({edge, owners.size(), topology.isRaw(edge.lower, edge.higher),
                       input.isSharp(edge.lower, edge.higher), kink});
    }
    return out;
}

std::vector<EdgeGeometry> geometryReport(const MeshReadingInput& input, const Topology& topology,
                                         std::span<const ElementSurface> surfaces, std::size_t pointsPerEdge)
{
    std::vector<double> tau;
    for (const LineRulePoint& r : unitIntervalRule(pointsPerEdge))
        tau.push_back(r.point);
    const EdgeSurfaceEvaluator point = [&](std::size_t element, std::size_t localEdge, double t)
    {
        const Patch& patch = surfaces[element].patch;
        const ReferencePoint uv = referenceEdgePointAndDirection(patch.cornerCount(), localEdge, t).first;
        const SurfacePointFrame<double> f = frame(patch, uv[0], uv[1]);
        return EdgeSurfacePoint{f.position, f.normal};
    };
    return geometryReport(input, topology, tau, point);
}

} // namespace oos::elements::mixedHybrid
