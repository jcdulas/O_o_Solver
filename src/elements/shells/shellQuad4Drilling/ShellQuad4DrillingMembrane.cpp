#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingMembrane.hpp"

#include "oos/elements/core/IntegrationRules.hpp"

#include <utility>

namespace oos::elements
{

namespace
{

// The local DOFs of corner i on (u, v, θn).
constexpr std::size_t U = 0;
constexpr std::size_t V = 1;
constexpr std::size_t THETA = 2;

// The edge functions of the serendipity element on edges 1-2, 2-3, 3-4, 4-1 and their parent derivatives.
struct EdgeFunctions
{
    std::array<double, 4> dXi{};
    std::array<double, 4> dEta{};
};

[[nodiscard]] EdgeFunctions edgeFunctionsAt(double xi, double eta)
{
    EdgeFunctions e;
    // ½(1 − ξ²)(1 − η), ½(1 + ξ)(1 − η²), ½(1 − ξ²)(1 + η), ½(1 − ξ)(1 − η²).
    e.dXi = {-xi * (1.0 - eta), 0.5 * (1.0 - eta * eta), -xi * (1.0 + eta), -0.5 * (1.0 - eta * eta)};
    e.dEta = {-0.5 * (1.0 - xi * xi), -eta * (1.0 + xi), 0.5 * (1.0 - xi * xi), -eta * (1.0 - xi)};
    return e;
}

// The rows of ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y over (u, v, θn) at a point.
struct GradientRows
{
    std::array<double, 12> uX{};
    std::array<double, 12> uY{};
    std::array<double, 12> vX{};
    std::array<double, 12> vY{};
    std::array<double, 12> theta{}; // θn = Σ N_i θn_i
};

[[nodiscard]] GradientRows gradientRows(const ShellQuad4DrillingGeometry& geometry, double xi, double eta)
{
    GradientRows rows;
    const BilinearShape shape = bilinearShapeAt(xi, eta);
    const PlaneJacobian jacobian = geometry.jacobianAt(xi, eta);
    for (std::size_t i = 0; i < 4; ++i)
    {
        rows.uX[3 * i + U] = jacobian.dX[i];
        rows.uY[3 * i + U] = jacobian.dY[i];
        rows.vX[3 * i + V] = jacobian.dX[i];
        rows.vY[3 * i + V] = jacobian.dY[i];
        rows.theta[3 * i + THETA] = shape.n[i];
    }
    const EdgeFunctions edges = edgeFunctionsAt(xi, eta);
    for (std::size_t k = 0; k < 4; ++k)
    {
        const std::size_t i = k;
        const std::size_t j = (k + 1) % 4;
        // ∂M_k/∂x, ∂M_k/∂y = J⁻¹ [∂M_k/∂ξ; ∂M_k/∂η].
        const double mX = jacobian.inverse(0, 0) * edges.dXi[k] + jacobian.inverse(0, 1) * edges.dEta[k];
        const double mY = jacobian.inverse(1, 0) * edges.dXi[k] + jacobian.inverse(1, 1) * edges.dEta[k];
        // Δu = M_k (y_j − y_i)/8 (θn_j − θn_i), Δv = M_k (x_i − x_j)/8 (θn_j − θn_i).
        const double cu = (geometry.y[j] - geometry.y[i]) / 8.0;
        const double cv = (geometry.x[i] - geometry.x[j]) / 8.0;
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

ShellQuad4DrillingMembrane::ShellQuad4DrillingMembrane(const ShellQuad4DrillingGeometry& geometry, const base::Mat3& planeStress,
                                                   const std::array<double, 4>& thickness, double shearModulus)
    : m_geometry(geometry)
{
    const double penalty = DRILLING_MEMBRANE_PENALTY_RATIO * shearModulus;
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const BilinearShape shape = bilinearShapeAt(point.xi, point.eta);
        double t = 0.0;
        for (std::size_t i = 0; i < 4; ++i)
            t += shape.n[i] * thickness[i];
        const double scale = point.weight * geometry.jacobianAt(point.xi, point.eta).determinant * t;
        const base::MatN<3, 12> b = strainMatrix(point.xi, point.eta);
        m_stiffness += scale * base::congruence(planeStress, b);
        const GradientRows rows = gradientRows(geometry, point.xi, point.eta);
        std::array<double, 12> h{};
        for (std::size_t c = 0; c < 12; ++c)
            h[c] = rows.theta[c] - 0.5 * (rows.vX[c] - rows.uY[c]);
        for (std::size_t r = 0; r < 12; ++r)
            for (std::size_t c = 0; c < 12; ++c)
                m_stiffness(r, c) += scale * penalty * h[r] * h[c];
    }
}

base::MatN<3, 12> ShellQuad4DrillingMembrane::strainMatrix(double xi, double eta) const
{
    const GradientRows rows = gradientRows(m_geometry, xi, eta);
    base::MatN<3, 12> b;
    for (std::size_t c = 0; c < 12; ++c)
    {
        b(0, c) = rows.uX[c];
        b(1, c) = rows.vY[c];
        b(2, c) = rows.uY[c] + rows.vX[c];
    }
    return b;
}

} // namespace oos::elements
