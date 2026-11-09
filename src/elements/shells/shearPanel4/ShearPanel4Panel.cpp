#include "oos/elements/shells/shearPanel4/ShearPanel4Panel.hpp"

#include "oos/elements/core/IntegrationRules.hpp"
#include "oos/elements/core/ShapeFunctions.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements
{

namespace
{

// The sign of the edge forces of a positive shear on edges 1-2, 2-3, 3-4, 4-1 (see ShearPanel4Panel.hpp).
constexpr std::array<double, 4> EDGE_SIGNS{-1.0, 1.0, -1.0, 1.0};

[[nodiscard]] base::Vec3 sub(const base::Vec3& a, const base::Vec3& b)
{
    return a - b;
}

[[nodiscard]] double determinant3(const std::array<std::array<double, 3>, 3>& m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// The rod areas of ShearPanelProp F1, F2 for an effectiveness F on two opposite edges of lengths la, lb.
[[nodiscard]] double rodArea(double effectiveness, double thickness, double area, double la, double lb)
{
    if (!(effectiveness > 0.0))
        return 0.0;
    if (effectiveness <= SHEAR_PANEL_EFFECTIVENESS_LIMIT)
        return effectiveness * thickness * area / (la + lb);
    return 0.5 * effectiveness * thickness * thickness;
}

} // namespace

ShearPanel4Panel::BuildResult ShearPanel4Panel::build(const std::array<base::Vec3, 4>& positions,
                                                            const ShearPanelSection& section)
{
    BuildResult result;
    ShearPanel4Panel& panel = result.panel;
    panel.m_section = section;
    const base::Vec3 d13 = sub(positions[2], positions[0]);
    const base::Vec3 d24 = sub(positions[3], positions[1]);
    const base::Vec3 normal = base::cross(d13, d24);
    const double normalLength = base::norm(normal);
    if (!(normalLength > 0.0) || normalLength <= SHEAR_PANEL_MIN_CORNER_RATIO * base::norm(d13) * base::norm(d24))
    {
        result.problem = ShearPanelProblem::Degenerate;
        return result;
    }
    const base::Vec3 n{normal[0] / normalLength, normal[1] / normalLength, normal[2] / normalLength};
    panel.m_area = 0.5 * normalLength;

    // The heights off the mean plane through the centroid: a flat panel only (the kick forces of a warped one are
    // not formed).
    base::Vec3 centre{};
    for (const base::Vec3& p : positions)
        for (std::size_t k = 0; k < 3; ++k)
            centre[k] += 0.25 * p[k];
    double largestHeight = 0.0;
    for (const base::Vec3& p : positions)
        largestHeight = std::max(largestHeight, std::abs(base::dot(sub(p, centre), n)));
    if (largestHeight > SHEAR_PANEL_WARP_TOLERANCE * 0.5 * (base::norm(d13) + base::norm(d24)))
    {
        result.problem = ShearPanelProblem::Warped;
        return result;
    }

    // The element frame: x along G1→G2 projected on the plane, y = n × x.
    base::Vec3 edge = sub(positions[1], positions[0]);
    const double along = base::dot(edge, n);
    for (std::size_t k = 0; k < 3; ++k)
        edge[k] -= along * n[k];
    const double edgeLength = base::norm(edge);
    const base::Vec3 x{edge[0] / edgeLength, edge[1] / edgeLength, edge[2] / edgeLength};
    const base::Vec3 y = base::cross(n, x);
    for (std::size_t k = 0; k < 3; ++k)
    {
        panel.m_axes(k, 0) = x[k];
        panel.m_axes(k, 1) = y[k];
        panel.m_axes(k, 2) = n[k];
    }
    for (std::size_t i = 0; i < 4; ++i)
    {
        const base::Vec3 r = sub(positions[i], centre);
        panel.m_x[i] = base::dot(r, x);
        panel.m_y[i] = base::dot(r, y);
    }

    // Edge vectors, and the convexity of each corner (the cross product of its two edges, positive for a convex
    // counter-clockwise panel).
    std::array<double, 4> ux{};
    std::array<double, 4> uy{};
    for (std::size_t e = 0; e < 4; ++e)
    {
        const std::size_t b = (e + 1) % 4;
        const double dx = panel.m_x[b] - panel.m_x[e];
        const double dy = panel.m_y[b] - panel.m_y[e];
        panel.m_lengths[e] = std::hypot(dx, dy);
        ux[e] = dx / panel.m_lengths[e];
        uy[e] = dy / panel.m_lengths[e];
    }
    for (std::size_t i = 0; i < 4; ++i)
    {
        const std::size_t previous = (i + 3) % 4;
        const double corner =
            panel.m_lengths[previous] * panel.m_lengths[i] * (ux[previous] * uy[i] - uy[previous] * ux[i]);
        if (!(corner > SHEAR_PANEL_MIN_CORNER_RATIO * panel.m_area))
        {
            result.problem = ShearPanelProblem::Folded;
            return result;
        }
    }

    // The equilibrium of the edge forces s_e c_e L_e u_e (2 forces, 1 moment about the centroid; the moment of a
    // force along edge e is that of any point of its line, here its start).
    std::array<std::array<double, 4>, 3> system{};
    for (std::size_t e = 0; e < 4; ++e)
    {
        const double scale = EDGE_SIGNS[e] * panel.m_lengths[e];
        system[0][e] = scale * ux[e];
        system[1][e] = scale * uy[e];
        system[2][e] = scale * (panel.m_x[e] * uy[e] - panel.m_y[e] * ux[e]);
    }
    // The null vector of the 3 × 4 system: its signed 3 × 3 minors (c_j = (−1)^j det(system without column j)).
    double mean = 0.0;
    for (std::size_t j = 0; j < 4; ++j)
    {
        std::array<std::array<double, 3>, 3> minor{};
        for (std::size_t r = 0; r < 3; ++r)
        {
            std::size_t column = 0;
            for (std::size_t c = 0; c < 4; ++c)
                if (c != j)
                    minor[r][column++] = system[r][c];
        }
        panel.m_pattern[j] = (j % 2 == 0 ? 1.0 : -1.0) * determinant3(minor);
        mean += 0.25 * panel.m_pattern[j];
    }
    for (double& c : panel.m_pattern)
        c /= mean;

    // b: each edge force halved at the two ends of its edge.
    for (std::size_t e = 0; e < 4; ++e)
    {
        const double half = 0.5 * EDGE_SIGNS[e] * panel.m_pattern[e] * panel.m_lengths[e];
        for (const std::size_t node : {e, (e + 1) % 4})
        {
            panel.m_cornerForces[3 * node] += half * ux[e];
            panel.m_cornerForces[3 * node + 1] += half * uy[e];
        }
    }
    panel.m_flexibility = panel.m_area / (section.shearModulus * section.thickness);
    return result;
}

base::MatN<12> ShearPanel4Panel::stiffness() const
{
    base::MatN<12> k;
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = 0; j < 12; ++j)
            k(i, j) = m_cornerForces[i] * m_cornerForces[j] / m_flexibility;
    // The extensional rods on the edges (PSHEAR F1, F2).
    const std::array<double, 2> areas{rodArea(m_section.f1, m_section.thickness, m_area, m_lengths[0], m_lengths[2]),
                                      rodArea(m_section.f2, m_section.thickness, m_area, m_lengths[1], m_lengths[3])};
    for (std::size_t e = 0; e < 4; ++e)
    {
        const double rod = areas[e % 2];
        if (!(rod > 0.0))
            continue;
        const std::size_t a = e;
        const std::size_t b = (e + 1) % 4;
        const double ux = (m_x[b] - m_x[a]) / m_lengths[e];
        const double uy = (m_y[b] - m_y[a]) / m_lengths[e];
        const double axial = m_section.youngModulus * rod / m_lengths[e];
        const std::array<double, 2> u{ux, uy};
        for (std::size_t r = 0; r < 2; ++r)
            for (std::size_t c = 0; c < 2; ++c)
            {
                const double term = axial * u[r] * u[c];
                k(3 * a + r, 3 * a + c) += term;
                k(3 * b + r, 3 * b + c) += term;
                k(3 * a + r, 3 * b + c) -= term;
                k(3 * b + r, 3 * a + c) -= term;
            }
    }
    return k;
}

double ShearPanel4Panel::meanFlow(const std::array<double, 12>& q) const
{
    double work = 0.0;
    for (std::size_t i = 0; i < 12; ++i)
        work += m_cornerForces[i] * q[i];
    return work / m_flexibility;
}

base::MatN<12> ShearPanel4Panel::mass(double areaMass) const
{
    const double total = (m_section.density * m_section.thickness + m_section.nonStructuralMass + areaMass) * m_area;
    base::MatN<12> m;
    for (std::size_t i = 0; i < 12; ++i)
        m(i, i) = 0.25 * total;
    return m;
}

base::MatN<12> ShearPanel4Panel::differentialStiffness(double meanFlow) const
{
    base::MatN<12> k;
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const PlaneJacobian jacobian = planeJacobianAt(bilinearShapeAt(point.xi, point.eta), m_x, m_y);
        addMembraneDifferentialStiffness<4, 3>(k, jacobian.dX, jacobian.dY, point.weight * jacobian.determinant,
                                               {0.0, 0.0, meanFlow});
    }
    return k;
}

} // namespace oos::elements
