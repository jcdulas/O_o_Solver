#include "ShellQuad4Kinematics.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

using quad4_detail::NODE_ETA;
using quad4_detail::NODE_XI;

// The diagonals G1-G3 and G2-G4 are parallel (no mean plane) when ‖d13 × d24‖ ≤ this factor · ‖d13‖·‖d24‖.
constexpr double PARALLEL_DIAGONALS_FACTOR = 1.0e-12;
// A nodal normal is undefined when ‖s × t‖ ≤ this factor · ‖s‖·‖t‖ (two edges at a node along one line).
constexpr double DEGENERATE_NORMAL_FACTOR = 1.0e-12;
// Spec §2.6: a corner cosine below this magnitude is set to 0 (a right angle up to round-off).
constexpr double RIGHT_ANGLE_COSINE = 1.0e-7;
// A corner cosine of magnitude at or above 1 − this puts two edges on one line: 1 − c² vanishes in the edge shear.
constexpr double STRAIGHT_CORNER_MARGIN = 1.0e-10;
// Spec §2.7: the warping measure S_w = 28 Σ + 1.
constexpr double WARP_MEASURE_SCALE = 28.0;

[[nodiscard]] Vector3<double> vectorOf(const base::Vec3& v)
{
    return {v[0], v[1], v[2]};
}

[[nodiscard]] Vector3<double> difference(const Vector3<double>& a, const Vector3<double>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] double lengthOf(const Vector3<double>& v)
{
    return std::sqrt(dotOf(v, v));
}

[[nodiscard]] ShellQuad4GeometryResult refused(std::string field, std::string problem)
{
    ShellQuad4GeometryResult result;
    result.problemField = std::move(field);
    result.problem = std::move(problem);
    return result;
}

// Spec §2.2(4): the unit normal of the bilinear surface of the basic positions at node a, (s × t)/‖s × t‖ with
// s = Σ_b ∂N_b/∂ξ x_b and t = Σ_b ∂N_b/∂η x_b at the node's natural coordinates; nothing when degenerate.
[[nodiscard]] std::optional<Vector3<double>> nodalNormal(const std::array<Vector3<double>, 4>& x, std::size_t a)
{
    const quad4_detail::BilinearShape shape = quad4_detail::bilinearShape(NODE_XI[a], NODE_ETA[a]);
    Vector3<double> s{};
    Vector3<double> t{};
    for (std::size_t b = 0; b < 4; ++b)
        for (std::size_t k = 0; k < 3; ++k)
        {
            s[k] += shape.dXi[b] * x[b][k];
            t[k] += shape.dEta[b] * x[b][k];
        }
    const Vector3<double> n = crossOf(s, t);
    const double length = lengthOf(n);
    if (!(length > DEGENERATE_NORMAL_FACTOR * lengthOf(s) * lengthOf(t)))
        return std::nullopt;
    return scaledBy(n, 1.0 / length);
}

// Spec §2.7(1): the warping measure of the anisotropic shear factor from the basic nodal normals,
// Σ_x = ½ ‖n1 × n2‖ + ½ ‖n3 × n4‖, Σ_y = ½ ‖n2 × n3‖ + ½ ‖n1 × n4‖, S_w = 28 max(Σ_x, Σ_y) + 1.
[[nodiscard]] double warpFactorOf(const std::array<Vector3<double>, 4>& n)
{
    const double alongX = 0.5 * lengthOf(crossOf(n[0], n[1])) + 0.5 * lengthOf(crossOf(n[2], n[3]));
    const double alongY = 0.5 * lengthOf(crossOf(n[1], n[2])) + 0.5 * lengthOf(crossOf(n[0], n[3]));
    return WARP_MEASURE_SCALE * std::max(alongX, alongY) + 1.0;
}

} // namespace

ShellQuad4GeometryResult buildShellQuad4Geometry(const std::array<base::Vec3, 4>& positions)
{
    std::array<Vector3<double>, 4> x{};
    for (std::size_t a = 0; a < 4; ++a)
        x[a] = vectorOf(positions[a]);

    // Spec §2.2(1): the normal k of the mean plane, along d13 × d24.
    const Vector3<double> d13 = difference(x[2], x[0]);
    const Vector3<double> d24 = difference(x[3], x[1]);
    const Vector3<double> diagonalCross = crossOf(d13, d24);
    const double crossLength = lengthOf(diagonalCross);
    if (!(crossLength > PARALLEL_DIAGONALS_FACTOR * lengthOf(d13) * lengthOf(d24)))
        return refused("node1",
                       "expected a quadrilateral, found diagonals ShellQuad4.node1-ShellQuad4.node3 and "
                       "ShellQuad4.node2-ShellQuad4.node4 parallel or of zero length; the four "
                       "grid points do not span a plane: correct their positions or the order of "
                       "ShellQuad4.node1-ShellQuad4.node4");
    const Vector3<double> k = scaledBy(diagonalCross, 1.0 / crossLength);

    // Spec §2.2(2): the user frame R, x along G1→G2 projected on the mean plane.
    const Vector3<double> e12 = difference(x[1], x[0]);
    const Vector3<double> projected = difference(e12, scaledBy(k, dotOf(e12, k)));
    if (!(lengthOf(projected) > 0.0))
        return refused("node2",
                       "expected ShellQuad4.node1 and ShellQuad4.node2 apart in the plane of the "
                       "element, found them on its normal; "
                       "correct their positions");
    const Vector3<double> userX = unitOf(projected);
    const Vector3<double> userY = crossOf(k, userX);

    // Spec §2.2(3): the element x axis bisects the two diagonals in the user plane, g = unit(d13) − unit(d24).
    std::array<std::array<double, 2>, 4> inUser{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        const Vector3<double> fromFirst = difference(x[a], x[0]);
        inUser[a] = {dotOf(fromFirst, userX), dotOf(fromFirst, userY)};
    }
    const double p0 = inUser[2][0] - inUser[0][0];
    const double p1 = inUser[2][1] - inUser[0][1];
    const double q0 = inUser[3][0] - inUser[1][0];
    const double q1 = inUser[3][1] - inUser[1][1];
    const double pLength = std::hypot(p0, p1);
    const double qLength = std::hypot(q0, q1);
    if (!(pLength > 0.0) || !(qLength > 0.0))
        return refused("node1",
                       "expected diagonals of non-zero length in the plane of the element, found one of zero "
                       "length; correct the grid point positions");
    const double bisectorAngle = std::atan2(p1 / pLength - q1 / qLength, p0 / pLength - q0 / qLength);

    ShellQuad4Geometry geometry;
    const Vector3<double> elementX =
        sumOf(scaledBy(userX, std::cos(bisectorAngle)), scaledBy(userY, std::sin(bisectorAngle)));
    const Vector3<double> elementY = crossOf(k, elementX);
    geometry.userAngle = -bisectorAngle;
    for (std::size_t r = 0; r < 3; ++r)
    {
        geometry.axes(r, 0) = elementX[r];
        geometry.axes(r, 1) = elementY[r];
        geometry.axes(r, 2) = k[r];
    }
    Vector3<double> centre{};
    for (std::size_t a = 0; a < 4; ++a)
        centre = sumOf(centre, scaledBy(x[a], 0.25));
    geometry.centre = {centre[0], centre[1], centre[2]};
    const std::array<Vector3<double>, 3> axes{elementX, elementY, k};
    const auto toElement = [&](const Vector3<double>& v)
    { return Vector3<double>{dotOf(v, axes[0]), dotOf(v, axes[1]), dotOf(v, axes[2])}; };
    for (std::size_t a = 0; a < 4; ++a)
        geometry.position[a] = toElement(difference(x[a], centre));

    // Spec §2.2(4): the nodal normals, in basic then in the element frame.
    std::array<Vector3<double>, 4> basicNormal{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        const std::optional<Vector3<double>> normal = nodalNormal(x, a);
        if (!normal)
            return refused(std::string("node").append(std::to_string(a + 1)),
                           "expected a quadrilateral corner, found the two edges at ShellQuad4.node" +
                               std::to_string(a + 1) +
                               " along one line or of zero length; reorder "
                               "ShellQuad4.node1-ShellQuad4.node4 around the element, or use a "
                               "ShellTria3 for a triangle");
        basicNormal[a] = *normal;
        geometry.normal[a] = toElement(*normal);
    }
    geometry.warpFactor = warpFactorOf(basicNormal);

    // Spec §2.6: the edges and the corner cosines.
    for (std::size_t e = 0; e < 4; ++e)
    {
        const std::size_t next = (e + 1) % 4;
        const Vector3<double> edge = difference(geometry.position[next], geometry.position[e]);
        geometry.edgeLength[e] = lengthOf(edge);
        if (!(geometry.edgeLength[e] > 0.0))
            return refused(std::string("node").append(std::to_string(next + 1)),
                           "expected four distinct grid points, found ShellQuad4.node" + std::to_string(e + 1) +
                               " and ShellQuad4.node" + std::to_string(next + 1) +
                               " at one position; correct their positions");
        geometry.edgeDirection[e] = scaledBy(edge, 1.0 / geometry.edgeLength[e]);
        geometry.edgeNormal[e] = unitOf(sumOf(geometry.normal[e], geometry.normal[next]));
    }
    for (std::size_t a = 0; a < 4; ++a)
    {
        const double cosine = -dotOf(geometry.edgeDirection[a], geometry.edgeDirection[(a + 3) % 4]);
        if (std::abs(cosine) >= 1.0 - STRAIGHT_CORNER_MARGIN)
            return refused(std::string("node").append(std::to_string(a + 1)),
                           "expected a quadrilateral corner at ShellQuad4.node" + std::to_string(a + 1) +
                               ", found its two edges along one line; the edge shear cannot be formed there: "
                               "remesh the element");
        geometry.cornerCosine[a] = std::abs(cosine) < RIGHT_ANGLE_COSINE ? 0.0 : cosine;
    }

    // Spec §2.7: the mean widths A and B and the aspect α.
    const std::array<Vector3<double>, 4>& p = geometry.position;
    geometry.meanWidthX = 0.5 * std::abs(p[1][0] + p[2][0] - p[0][0] - p[3][0]);
    geometry.meanWidthY = 0.5 * std::abs(p[3][1] + p[2][1] - p[0][1] - p[1][1]);
    if (!(geometry.meanWidthX > 0.0) || !(geometry.meanWidthY > 0.0))
        return refused("node1",
                       "expected a quadrilateral of non-zero width along both element axes, found a folded "
                       "one; reorder ShellQuad4.node1-ShellQuad4.node4 around the element");
    geometry.aspect =
        std::min(geometry.meanWidthX, geometry.meanWidthY) / std::max(geometry.meanWidthX, geometry.meanWidthY);
    ShellQuad4GeometryResult result;
    result.geometry = geometry;
    return result;
}

} // namespace oos::elements
