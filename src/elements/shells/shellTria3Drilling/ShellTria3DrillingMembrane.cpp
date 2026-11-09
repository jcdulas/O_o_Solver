#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingMembrane.hpp"

#include "oos/elements/shells/ShellSection.hpp"

#include <utility>

namespace oos::elements
{

namespace
{

constexpr std::size_t U = 0;
constexpr std::size_t V = 1;
constexpr std::size_t THETA = 2;

// The three interior points of the quadratic rule in area coordinates, weight A/3 each.
constexpr std::array<std::array<double, 3>, 3> POINTS{
    {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}, {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}}};

struct GradientRows
{
    std::array<double, 9> uX{};
    std::array<double, 9> uY{};
    std::array<double, 9> vX{};
    std::array<double, 9> vY{};
    std::array<double, 9> theta{};
};

[[nodiscard]] GradientRows gradientRows(const ShellTria3DrillingGeometry& g, const std::array<double, 3>& l)
{
    GradientRows rows;
    for (std::size_t i = 0; i < 3; ++i)
    {
        rows.uX[3 * i + U] = g.dX[i];
        rows.uY[3 * i + U] = g.dY[i];
        rows.vX[3 * i + V] = g.dX[i];
        rows.vY[3 * i + V] = g.dY[i];
        rows.theta[3 * i + THETA] = l[i];
    }
    for (std::size_t i = 0; i < 3; ++i)
    {
        const std::size_t j = (i + 1) % 3;
        // ∂(4 L_i L_j)/∂x = 4 (L_j ∂L_i/∂x + L_i ∂L_j/∂x).
        const double mX = 4.0 * (l[j] * g.dX[i] + l[i] * g.dX[j]);
        const double mY = 4.0 * (l[j] * g.dY[i] + l[i] * g.dY[j]);
        const double cu = (g.y[j] - g.y[i]) / 8.0;
        const double cv = (g.x[i] - g.x[j]) / 8.0;
        for (const auto& [node, sign] : {std::pair<std::size_t, double>{j, 1.0}, {i, -1.0}})
        {
            rows.uX[3 * node + THETA] += sign * mX * cu;
            rows.uY[3 * node + THETA] += sign * mY * cu;
            rows.vX[3 * node + THETA] += sign * mX * cv;
            rows.vY[3 * node + THETA] += sign * mY * cv;
        }
    }
    return rows;
}

} // namespace

ShellTria3DrillingMembrane::ShellTria3DrillingMembrane(const ShellTria3DrillingGeometry& geometry, const base::Mat3& planeStress,
                                                   double thickness, double shearModulus)
    : m_geometry(geometry)
{
    const double penalty = DRILLING_MEMBRANE_PENALTY_RATIO * shearModulus;
    const double scale = geometry.area / 3.0 * thickness;
    for (const std::array<double, 3>& l : POINTS)
    {
        m_stiffness += scale * base::congruence(planeStress, strainMatrix(l));
        const GradientRows rows = gradientRows(geometry, l);
        std::array<double, 9> h{};
        for (std::size_t c = 0; c < 9; ++c)
            h[c] = rows.theta[c] - 0.5 * (rows.vX[c] - rows.uY[c]);
        for (std::size_t r = 0; r < 9; ++r)
            for (std::size_t c = 0; c < 9; ++c)
                m_stiffness(r, c) += scale * penalty * h[r] * h[c];
    }
}

base::MatN<3, 9> ShellTria3DrillingMembrane::strainMatrix(const std::array<double, 3>& l) const
{
    const GradientRows rows = gradientRows(m_geometry, l);
    base::MatN<3, 9> b;
    for (std::size_t c = 0; c < 9; ++c)
    {
        b(0, c) = rows.uX[c];
        b(1, c) = rows.vY[c];
        b(2, c) = rows.uY[c] + rows.vX[c];
    }
    return b;
}

} // namespace oos::elements
