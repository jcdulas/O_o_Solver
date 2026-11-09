#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements
{

ShellQuad4DrillingGeometryResult buildShellQuad4DrillingGeometry(const std::array<base::Vec3, 4>& positions)
{
    ShellQuad4DrillingGeometryResult result;
    ShellQuad4DrillingGeometry& g = result.geometry;

    // The mean plane from the diagonals; x_u the edge G1→G2 projected on it.
    const base::Vec3 d1 = positions[2] - positions[0];
    const base::Vec3 d2 = positions[3] - positions[1];
    const base::Vec3 normal = base::cross(d1, d2);
    if (base::norm(normal) <= QUAD_DEGENERATE_NORMAL_FACTOR * base::norm(d1) * base::norm(d2) ||
        base::norm(d1) == 0.0 || base::norm(d2) == 0.0)
    {
        result.problem = QuadGeometryProblem::DegenerateNormal;
        return result;
    }
    const base::Vec3 n = base::normalized(normal);
    const base::Vec3 edge12 = positions[1] - positions[0];
    const base::Vec3 xu = base::normalized(edge12 - base::dot(edge12, n) * n);
    const base::Vec3 yu = base::cross(n, xu);

    // The x axis on the bisector of the projected diagonals, b = (e31 − e42)/‖e31 − e42‖.
    const auto projectedUnit = [&](base::Vec3 d)
    {
        const double px = base::dot(d, xu);
        const double py = base::dot(d, yu);
        const double length = std::hypot(px, py);
        return std::array<double, 2>{px / length, py / length};
    };
    const std::array<double, 2> e31 = projectedUnit(d1);
    const std::array<double, 2> e42 = projectedUnit(d2);
    const double bx = e31[0] - e42[0];
    const double by = e31[1] - e42[1];
    const double bisectorLength = std::hypot(bx, by);
    base::Vec3 xAxis = xu;
    if (bisectorLength > QUAD_BISECTOR_TOLERANCE)
        xAxis = base::normalized((bx / bisectorLength) * xu + (by / bisectorLength) * yu);
    const base::Vec3 yAxis = base::cross(n, xAxis);
    g.axes = base::matrixFromRows(xAxis, yAxis, n).transposed();

    // The plane coordinates and heights. They are taken from the mean of the four points (the formulas use only
    // differences of coordinates, and a local origin keeps the round-off of far-away models out).
    g.centre = 0.25 * (positions[0] + positions[1] + positions[2] + positions[3]);
    std::array<double, 4> z{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        const base::Vec3 relative = positions[i] - g.centre;
        g.x[i] = base::dot(relative, xAxis);
        g.y[i] = base::dot(relative, yAxis);
        z[i] = base::dot(relative, n);
    }
    const double zMean = 0.25 * (z[0] + z[1] + z[2] + z[3]);
    double largestDeviation = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        g.zDeviation[i] = z[i] - zMean;
        largestDeviation = std::max(largestDeviation, std::abs(g.zDeviation[i]));
    }
    const auto [xMin, xMax] = std::minmax_element(g.x.begin(), g.x.end());
    const auto [yMin, yMax] = std::minmax_element(g.y.begin(), g.y.end());
    const double extent = std::max(*xMax - *xMin, *yMax - *yMin);
    g.isWarped = largestDeviation > QUAD_WARP_TOLERANCE * extent;
    g.warpRatio = largestDeviation / (0.5 * (base::norm(d1) + base::norm(d2)));
    if (!g.isWarped)
        g.zDeviation = {0.0, 0.0, 0.0, 0.0}; // a flat element uses K8 on (u, v) only

    // The half-widths A and B, and the angle of the edge G1→G2.
    g.widthA = 0.5 * std::abs(g.x[1] + g.x[2] - g.x[0] - g.x[3]);
    g.widthB = 0.5 * std::abs(g.y[2] + g.y[3] - g.y[0] - g.y[1]);
    g.edgeAngle = std::atan2(g.y[1] - g.y[0], g.x[1] - g.x[0]);

    g.centreJacobian = g.jacobianAt(0.0, 0.0);
    const double parallelogramDeterminant = 0.25 * g.area();
    for (std::size_t i = 0; i < 4; ++i)
    {
        g.cornerJacobians[i] = g.jacobianAt(QUAD_CORNER_XI[i], QUAD_CORNER_ETA[i]);
        if (!(g.cornerJacobians[i].determinant > QUAD_MIN_CORNER_JACOBIAN_RATIO * parallelogramDeterminant))
        {
            result.problem = QuadGeometryProblem::FoldedCorner;
            result.corner = static_cast<std::int32_t>(i);
            return result;
        }
    }
    return result;
}

} // namespace oos::elements
