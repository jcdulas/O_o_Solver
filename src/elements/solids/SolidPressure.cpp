#include "oos/elements/solids/SolidPressure.hpp"

#include "oos/elements/core/SolidIntegrationRules.hpp"

#include <algorithm>

namespace oos::elements
{

namespace
{

// The face rules: 3 × 3 Gauss points on [−1, 1]², and a triangle rule of degree 4 (the triangle of a pentahedron
// rule with one point along ζ, whose weight 2 is divided out).
constexpr std::size_t QUAD_POINTS = 3;
constexpr std::size_t TRIANGLE_DEGREE = 4;

// The shape functions of a face (3, 4, 6 or 8 nodes) and their derivatives along its parameters (a, b): a
// quadrilateral on [−1, 1]² with corners (−1, −1), (1, −1), (1, 1), (−1, 1) and mid-side nodes on the edges 0-1, 1-2,
// 2-3, 3-0 (serendipity); a triangle in area coordinates (a, b) with corners (0, 0), (1, 0), (0, 1) and mid-side nodes
// on 0-1, 1-2, 2-0.
struct FaceShape
{
    std::array<double, 8> n{};
    std::array<double, 8> da{};
    std::array<double, 8> db{};
    std::array<double, 4> corner{}; // the linear (bilinear) functions of the corners, for the pressure
};

[[nodiscard]] FaceShape quadShape(double a, double b, bool isQuadratic)
{
    constexpr std::array<double, 4> CA{-1.0, 1.0, 1.0, -1.0};
    constexpr std::array<double, 4> CB{-1.0, -1.0, 1.0, 1.0};
    FaceShape f;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const double pa = 1.0 + CA[i] * a;
        const double pb = 1.0 + CB[i] * b;
        f.corner[i] = 0.25 * pa * pb;
        if (!isQuadratic)
        {
            f.n[i] = f.corner[i];
            f.da[i] = 0.25 * CA[i] * pb;
            f.db[i] = 0.25 * CB[i] * pa;
            continue;
        }
        const double s = CA[i] * a + CB[i] * b - 1.0;
        f.n[i] = 0.25 * pa * pb * s;
        f.da[i] = 0.25 * CA[i] * pb * (s + pa);
        f.db[i] = 0.25 * CB[i] * pa * (s + pb);
    }
    if (!isQuadratic)
        return f;
    // Mid-side nodes: edges 0-1 (b = −1), 1-2 (a = 1), 2-3 (b = 1), 3-0 (a = −1).
    f.n[4] = 0.5 * (1.0 - a * a) * (1.0 - b);
    f.da[4] = -a * (1.0 - b);
    f.db[4] = -0.5 * (1.0 - a * a);
    f.n[5] = 0.5 * (1.0 + a) * (1.0 - b * b);
    f.da[5] = 0.5 * (1.0 - b * b);
    f.db[5] = -b * (1.0 + a);
    f.n[6] = 0.5 * (1.0 - a * a) * (1.0 + b);
    f.da[6] = -a * (1.0 + b);
    f.db[6] = 0.5 * (1.0 - a * a);
    f.n[7] = 0.5 * (1.0 - a) * (1.0 - b * b);
    f.da[7] = -0.5 * (1.0 - b * b);
    f.db[7] = -b * (1.0 - a);
    return f;
}

[[nodiscard]] FaceShape triangleShape(double a, double b, bool isQuadratic)
{
    const std::array<double, 3> l{1.0 - a - b, a, b};
    constexpr std::array<double, 3> LA{-1.0, 1.0, 0.0};
    constexpr std::array<double, 3> LB{-1.0, 0.0, 1.0};
    FaceShape f;
    for (std::size_t i = 0; i < 3; ++i)
    {
        f.corner[i] = l[i];
        if (!isQuadratic)
        {
            f.n[i] = l[i];
            f.da[i] = LA[i];
            f.db[i] = LB[i];
            continue;
        }
        f.n[i] = l[i] * (2.0 * l[i] - 1.0);
        f.da[i] = (4.0 * l[i] - 1.0) * LA[i];
        f.db[i] = (4.0 * l[i] - 1.0) * LB[i];
    }
    if (!isQuadratic)
        return f;
    constexpr std::array<std::array<std::size_t, 2>, 3> EDGES{{{0, 1}, {1, 2}, {2, 0}}};
    for (std::size_t e = 0; e < 3; ++e)
    {
        const std::size_t i = EDGES[e][0];
        const std::size_t j = EDGES[e][1];
        f.n[3 + e] = 4.0 * l[i] * l[j];
        f.da[3 + e] = 4.0 * (LA[i] * l[j] + l[i] * LA[j]);
        f.db[3 + e] = 4.0 * (LB[i] * l[j] + l[i] * LB[j]);
    }
    return f;
}

} // namespace

std::optional<SolidFaceNodes> findSolidFace(SolidShape shape, std::span<const std::int32_t> elementIds,
                                            std::span<const std::int32_t> faceCorners)
{
    for (const SolidFace& face : solidFaces(shape))
    {
        if (face.cornerCount != faceCorners.size())
            continue;
        SolidFaceNodes found;
        found.cornerCount = face.cornerCount;
        bool isMatch = true;
        for (std::size_t k = 0; k < faceCorners.size() && isMatch; ++k)
        {
            const auto corner = std::find_if(face.corners.begin(), face.corners.begin() + face.cornerCount,
                                             [&](std::uint8_t c) { return elementIds[c] == faceCorners[k]; });
            isMatch = corner != face.corners.begin() + face.cornerCount;
            if (isMatch)
                found.nodes[k] = *corner;
        }
        if (!isMatch)
            continue;
        found.count = found.cornerCount;
        if (isQuadraticSolid(shape))
        {
            for (std::size_t k = 0; k < found.cornerCount; ++k)
            {
                const std::size_t a = found.nodes[k];
                const std::size_t b = found.nodes[(k + 1) % found.cornerCount];
                for (const SolidEdge& edge : solidEdges(shape))
                    if ((edge.first == a && edge.second == b) || (edge.first == b && edge.second == a))
                        found.nodes[found.count++] = edge.middle;
            }
        }
        return found;
    }
    return std::nullopt;
}

SolidFaceLoads solidPressureLoads(SolidShape shape, std::span<const base::Vec3> positions, const SolidFaceNodes& face,
                                  const std::array<double, 4>& cornerPressures)
{
    const bool isQuad = face.cornerCount == 4;
    const bool isQuadratic = face.count > face.cornerCount;
    SolidFaceLoads loads;
    loads.count = face.count;
    for (std::size_t i = 0; i < face.count; ++i)
        loads.nodes[i] = face.nodes[i];

    // The outward side: away from the centroid of the element's corners.
    base::Vec3 centroid{};
    const std::size_t corners = solidCornerCount(shape);
    for (std::size_t c = 0; c < corners; ++c)
        centroid = centroid + positions[c];
    centroid = centroid / static_cast<double>(corners);
    base::Vec3 faceCentre{};
    for (std::size_t k = 0; k < face.cornerCount; ++k)
        faceCentre = faceCentre + positions[face.nodes[k]];
    faceCentre = faceCentre / static_cast<double>(face.cornerCount);

    std::array<base::Vec3, 8> forces{};
    double orientation = 0.0;
    const auto addPoint = [&](const FaceShape& f, double weight)
    {
        base::Vec3 ta{};
        base::Vec3 tb{};
        double pressure = 0.0;
        for (std::size_t i = 0; i < face.count; ++i)
        {
            ta = ta + f.da[i] * positions[face.nodes[i]];
            tb = tb + f.db[i] * positions[face.nodes[i]];
        }
        for (std::size_t k = 0; k < face.cornerCount; ++k)
            pressure += f.corner[k] * cornerPressures[k];
        const base::Vec3 area = base::cross(ta, tb);
        for (std::size_t i = 0; i < face.count; ++i)
            forces[i] = forces[i] - (weight * pressure * f.n[i]) * area;
        orientation += weight * base::dot(area, faceCentre - centroid);
    };
    if (isQuad)
    {
        const GaussLegendreRule g = gaussLegendreRule(QUAD_POINTS);
        for (std::size_t i = 0; i < g.count; ++i)
            for (std::size_t j = 0; j < g.count; ++j)
                addPoint(quadShape(g.abscissae[i], g.abscissae[j], isQuadratic), g.weights[i] * g.weights[j]);
    }
    else
    {
        for (const SolidQuadraturePoint& point : pentahedronRule(TRIANGLE_DEGREE, 1).all())
            addPoint(triangleShape(point.point.xi, point.point.eta, isQuadratic), 0.5 * point.weight);
    }
    // −p n with n outward: when the face's parametrisation points inward, its area vector is reversed.
    const double sign = orientation < 0.0 ? -1.0 : 1.0;
    for (std::size_t i = 0; i < face.count; ++i)
        loads.forces[i] = sign * forces[i];
    return loads;
}

} // namespace oos::elements
