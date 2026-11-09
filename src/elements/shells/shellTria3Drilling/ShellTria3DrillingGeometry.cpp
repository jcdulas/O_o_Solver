#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"

#include <algorithm>

namespace oos::elements
{

ShellTria3DrillingGeometryResult buildShellTria3DrillingGeometry(const std::array<base::Vec3, 3>& positions)
{
    ShellTria3DrillingGeometryResult result;
    ShellTria3DrillingGeometry& g = result.geometry;
    const base::Vec3 edge12 = positions[1] - positions[0];
    const base::Vec3 edge13 = positions[2] - positions[0];
    const base::Vec3 normal = base::cross(edge12, edge13);
    if (base::norm(normal) <= TRIA_DEGENERATE_FACTOR * base::norm(edge12) * base::norm(edge13) ||
        base::norm(edge12) == 0.0 || base::norm(edge13) == 0.0)
    {
        result.isDegenerate = true;
        return result;
    }
    const base::Vec3 xAxis = base::normalized(edge12);
    const base::Vec3 n = base::normalized(normal);
    const base::Vec3 yAxis = base::cross(n, xAxis);
    g.axes = base::matrixFromRows(xAxis, yAxis, n).transposed();
    g.centre = (positions[0] + positions[1] + positions[2]) / 3.0;
    g.positions = positions;
    for (std::size_t i = 0; i < 3; ++i)
    {
        const base::Vec3 relative = positions[i] - g.centre;
        g.x[i] = base::dot(relative, xAxis);
        g.y[i] = base::dot(relative, yAxis);
    }
    g.area = 0.5 * base::norm(normal);
    const auto [yMin, yMax] = std::minmax_element(g.y.begin(), g.y.end());
    g.widthB = *yMax - *yMin;
    g.widthA = g.area / g.widthB;
    // Linear shape functions N_i = (a_i + b_i x + c_i y) / (2 A_e), b_i = y_j − y_k, c_i = x_k − x_j (i, j, k cyclic).
    for (std::size_t i = 0; i < 3; ++i)
    {
        const std::size_t j = (i + 1) % 3;
        const std::size_t k = (i + 2) % 3;
        g.dX[i] = (g.y[j] - g.y[k]) / (2.0 * g.area);
        g.dY[i] = (g.x[k] - g.x[j]) / (2.0 * g.area);
    }
    return result;
}

} // namespace oos::elements
