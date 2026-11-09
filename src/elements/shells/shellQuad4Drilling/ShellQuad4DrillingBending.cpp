#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingBending.hpp"

#include "oos/elements/core/IntegrationRules.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

// The bending DOFs of corner i, node by node.
constexpr std::size_t W = 0;
constexpr std::size_t THETA_X = 1;
constexpr std::size_t THETA_Y = 2;

// The constant of the gradient terms of the edge-shear coefficients (cgx = D11 (1/A² + 39/B²)).
constexpr double EDGE_SHEAR_GRADIENT_FACTOR = 39.0;

// The factor of the transverse shear saturation (c / (1 + 12 c / (t κ G3))). It puts each edge-shear term in series
// with the true transverse shear energy of the same strain field: the mean term 12 𝒜 c̄ γ̄² against κ G t 𝒜 γ̄², the
// gradient term 4 𝒜 cg g² against κ G t 𝒜 g² / 3 (a strain linear across the element, mean of η² 1/3).
constexpr double TRANSVERSE_SHEAR_SATURATION_FACTOR = 12.0;

[[nodiscard]] double thicknessAt(double xi, double eta, const std::array<double, 4>& thickness)
{
    const BilinearShape shape = bilinearShapeAt(xi, eta);
    double t = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        t += shape.n[i] * thickness[i];
    return t;
}

// The unit edge vectors and lengths of the four edges G1→G2, G2→G3, G3→G4, G4→G1 in the element plane.
struct Edges
{
    std::array<double, 4> ux{};
    std::array<double, 4> uy{};
    std::array<double, 4> length{};
};

[[nodiscard]] Edges edgesOf(const ShellQuad4DrillingGeometry& geometry)
{
    Edges edges;
    for (std::size_t e = 0; e < 4; ++e)
    {
        const std::size_t a = e;
        const std::size_t b = (e + 1) % 4;
        const double dx = geometry.x[b] - geometry.x[a];
        const double dy = geometry.y[b] - geometry.y[a];
        edges.length[e] = std::hypot(dx, dy);
        edges.ux[e] = dx / edges.length[e];
        edges.uy[e] = dy / edges.length[e];
    }
    return edges;
}

// The rows over (γ0, γ1, γ2, γ3) of the mean shears γ̄x, γ̄y and the gradients gx, gy of the edge shear. The top and
// left edges enter oriented as −u2, −u3 with −γ2, −γ3; every ratio below is unchanged by that double sign change.
struct EdgeShearRows
{
    std::array<double, 4> meanX{};
    std::array<double, 4> meanY{};
    std::array<double, 4> gradientX{};
    std::array<double, 4> gradientY{};
};

[[nodiscard]] EdgeShearRows edgeShearRows(const Edges& edges)
{
    const std::array<double, 4>& ux = edges.ux;
    const std::array<double, 4>& uy = edges.uy;
    // γ̄x = p − a γ̄y, γ̄y = q − c γ̄x, with p = ½ (γ_b / u_b,x + γ_t / u_t,x), q = ½ (γ_r / u_r,y + γ_l / u_l,y),
    // a = ½ (u_b,y / u_b,x + u_t,y / u_t,x), c = ½ (u_r,x / u_r,y + u_l,x / u_l,y).
    const double a = 0.5 * (uy[0] / ux[0] + uy[2] / ux[2]);
    const double c = 0.5 * (ux[1] / uy[1] + ux[3] / uy[3]);
    const std::array<double, 4> p{0.5 / ux[0], 0.0, 0.5 / ux[2], 0.0};
    const std::array<double, 4> q{0.0, 0.5 / uy[1], 0.0, 0.5 / uy[3]};
    const double determinant = 1.0 - a * c;
    EdgeShearRows rows;
    for (std::size_t k = 0; k < 4; ++k)
    {
        rows.meanX[k] = (p[k] - a * q[k]) / determinant;
        rows.meanY[k] = (q[k] - c * p[k]) / determinant;
    }
    // γx^bottom = (γ0 − γ̄y u0,y) / u0,x, γx^top = (γ2 − γ̄y u2,y) / u2,x; γy^right = (γ1 − γ̄x u1,x) / u1,y,
    // γy^left = (γ3 − γ̄x u3,x) / u3,y; gx = ½ (γx^top − γx^bottom), gy = ½ (γy^right − γy^left).
    for (std::size_t k = 0; k < 4; ++k)
    {
        const double bottom = ((k == 0 ? 1.0 : 0.0) - rows.meanY[k] * uy[0]) / ux[0];
        const double top = ((k == 2 ? 1.0 : 0.0) - rows.meanY[k] * uy[2]) / ux[2];
        const double right = ((k == 1 ? 1.0 : 0.0) - rows.meanX[k] * ux[1]) / uy[1];
        const double left = ((k == 3 ? 1.0 : 0.0) - rows.meanX[k] * ux[3]) / uy[3];
        rows.gradientX[k] = 0.5 * (top - bottom);
        rows.gradientY[k] = 0.5 * (right - left);
    }
    return rows;
}

// C (4 × 12): γ_e = (w_b − w_a)/L_e − ½ u_e·(β_a + β_b), β = (−θy, θx), so −½ u_e·β = ½ u_e,x θy − ½ u_e,y θx.
[[nodiscard]] base::MatN<4, 12> edgeShearMatrix(const Edges& edges)
{
    base::MatN<4, 12> c;
    for (std::size_t e = 0; e < 4; ++e)
    {
        for (const std::size_t node : {e, (e + 1) % 4})
        {
            c(e, 3 * node + THETA_X) = -0.5 * edges.uy[e];
            c(e, 3 * node + THETA_Y) = 0.5 * edges.ux[e];
        }
        c(e, 3 * e + W) = -1.0 / edges.length[e];
        c(e, 3 * ((e + 1) % 4) + W) = 1.0 / edges.length[e];
    }
    return c;
}

// B_b (3 × 12) at a point: κx = −∂θy/∂x, κy = ∂θx/∂y, 2κxy = ∂θx/∂x − ∂θy/∂y.
[[nodiscard]] base::MatN<3, 12> curvatureMatrix(const PlaneJacobian& jacobian)
{
    base::MatN<3, 12> b;
    for (std::size_t i = 0; i < 4; ++i)
    {
        b(0, 3 * i + THETA_Y) = -jacobian.dX[i];
        b(1, 3 * i + THETA_X) = jacobian.dY[i];
        b(2, 3 * i + THETA_X) = jacobian.dX[i];
        b(2, 3 * i + THETA_Y) = -jacobian.dY[i];
    }
    return b;
}

// c / (1 + 12 c / (t κ G)) in a flexible direction (κ G > 0), c in a rigid one.
[[nodiscard]] double saturated(double c, double shearStiffness, double centreThickness)
{
    if (!(shearStiffness > 0.0))
        return c;
    return c / (1.0 + TRANSVERSE_SHEAR_SATURATION_FACTOR * c / (centreThickness * shearStiffness));
}

} // namespace

EdgeShearCoefficients edgeShearCoefficients(double d11, double d22, double widthA, double widthB,
                                            const TransverseShear& shear, double centreThickness)
{
    const double a2 = widthA * widthA;
    const double b2 = widthB * widthB;
    EdgeShearCoefficients c;
    c.meanX = saturated(d11 / a2, shear.stiffness[0], centreThickness);
    c.meanY = saturated(d22 / b2, shear.stiffness[1], centreThickness);
    c.gradientX = saturated(d11 * (1.0 / a2 + EDGE_SHEAR_GRADIENT_FACTOR / b2), shear.stiffness[0], centreThickness);
    c.gradientY = saturated(d22 * (1.0 / b2 + EDGE_SHEAR_GRADIENT_FACTOR / a2), shear.stiffness[1], centreThickness);
    return c;
}

bool ShellQuad4DrillingBending::hasEdgeShearFrame(const ShellQuad4DrillingGeometry& geometry)
{
    const Edges edges = edgesOf(geometry);
    if (std::abs(edges.ux[0]) <= EDGE_SHEAR_MIN_COMPONENT || std::abs(edges.ux[2]) <= EDGE_SHEAR_MIN_COMPONENT ||
        std::abs(edges.uy[1]) <= EDGE_SHEAR_MIN_COMPONENT || std::abs(edges.uy[3]) <= EDGE_SHEAR_MIN_COMPONENT)
        return false;
    const double a = 0.5 * (edges.uy[0] / edges.ux[0] + edges.uy[2] / edges.ux[2]);
    const double c = 0.5 * (edges.ux[1] / edges.uy[1] + edges.ux[3] / edges.uy[3]);
    return std::abs(1.0 - a * c) > EDGE_SHEAR_MIN_COMPONENT;
}

ShellQuad4DrillingBending::ShellQuad4DrillingBending(const ShellQuad4DrillingGeometry& geometry, const base::Mat3& bendingMatrix,
                                   double bendingRatio, const std::array<double, 4>& thickness,
                                   const TransverseShear& shear)
    : m_geometry(geometry),
      m_bendingMatrix(bendingMatrix),
      m_bendingRatio(bendingRatio),
      m_thickness(thickness),
      m_shear(shear),
      m_centreThickness(thicknessAt(0.0, 0.0, thickness))
{
    // The curvatures: K_curvature = Gmᵀ H⁻¹ Gm.
    base::MatN<5> h;
    base::MatN<5, 12> gm;
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const PlaneJacobian jacobian = geometry.jacobianAt(point.xi, point.eta);
        const base::MatN<3, 5> p = momentInterpolation(point.xi, point.eta);
        const base::Mat3 flexibility = inverted(bendingStiffnessAt(point.xi, point.eta));
        const double scale = point.weight * jacobian.determinant;
        h += scale * (p.transposed() * (flexibility * p));
        gm += scale * (p.transposed() * curvatureMatrix(jacobian));
    }
    m_momentFlexibility = inverted(h);
    m_gm = gm;
    m_momentParameters = m_momentFlexibility * gm;
    m_stiffness = gm.transposed() * m_momentParameters;
    m_curvatureStiffness = m_stiffness;

    // The edge shear: Cᵀ S C.
    const Edges edges = edgesOf(geometry);
    const EdgeShearRows rows = edgeShearRows(edges);
    m_shearRows = {rows.meanX, rows.meanY, rows.gradientX, rows.gradientY};
    const double centreThickness = thicknessAt(0.0, 0.0, thickness);
    const base::Mat3 centreStiffness = bendingStiffnessAt(0.0, 0.0);
    const double d11 = centreStiffness(0, 0);
    const double d22 = centreStiffness(1, 1);
    m_coefficients = edgeShearCoefficients(d11, d22, geometry.widthA, geometry.widthB, shear, centreThickness);
    m_unsaturatedCoefficients =
        edgeShearCoefficients(d11, d22, geometry.widthA, geometry.widthB, TransverseShear{}, centreThickness);
    const base::MatN<4> s = edgeShearForm(m_coefficients);
    const base::MatN<4, 12> c = edgeShearMatrix(edges);
    m_edgeShears = c;
    m_stiffness += base::congruence(s, c);
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t col = 0; col < 12; ++col)
        {
            m_meanShears(0, col) += rows.meanX[k] * c(k, col);
            m_meanShears(1, col) += rows.meanY[k] * c(k, col);
        }
}

base::MatN<4> ShellQuad4DrillingBending::edgeShearForm(const EdgeShearCoefficients& c) const
{
    const double area = m_geometry.area();
    const std::array<double, 4>& meanX = m_shearRows[0];
    const std::array<double, 4>& meanY = m_shearRows[1];
    const std::array<double, 4>& gradientX = m_shearRows[2];
    const std::array<double, 4>& gradientY = m_shearRows[3];
    base::MatN<4> s;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            s(i, j) =
                12.0 * area * (c.meanX * meanX[i] * meanX[j] + c.meanY * meanY[i] * meanY[j]) +
                4.0 * area * (c.gradientX * gradientX[i] * gradientX[j] + c.gradientY * gradientY[i] * gradientY[j]);
    return s;
}

namespace
{

// dc/dT = (c/T)·(1 + 2·c/c_r) of a saturated edge-shear coefficient (ShellQuad4DrillingBending::thicknessRate), zero where the
// coefficient is.
[[nodiscard]] double coefficientRate(double c, double unsaturated, double thickness)
{
    return unsaturated != 0.0 ? c / thickness * (1.0 + 2.0 * c / unsaturated) : 0.0;
}

// d²c/dT² = 2·c·q·(4q − 1)/T², q = c/c_r, of a saturated edge-shear coefficient
// (ShellQuad4DrillingBending::thicknessSecondRate), zero where the coefficient is.
[[nodiscard]] double coefficientSecondRate(double c, double unsaturated, double thickness)
{
    if (unsaturated == 0.0)
        return 0.0;
    const double q = c / unsaturated;
    return 2.0 * c * q * (4.0 * q - 1.0) / (thickness * thickness);
}

} // namespace

std::array<double, 2> ShellQuad4DrillingBending::shearForceSecondRates(const std::array<double, 12>& u12, double thickness) const
{
    const std::array<double, 2> mean = m_meanShears * u12;
    return {12.0 * coefficientSecondRate(m_coefficients.meanX, m_unsaturatedCoefficients.meanX, thickness) * mean[0],
            12.0 * coefficientSecondRate(m_coefficients.meanY, m_unsaturatedCoefficients.meanY, thickness) * mean[1]};
}

base::MatN<12> ShellQuad4DrillingBending::thicknessSecondRate(double thickness) const
{
    const auto rateOf = [&](double c, double unsaturated) { return coefficientSecondRate(c, unsaturated, thickness); };
    const EdgeShearCoefficients& c = m_coefficients;
    const EdgeShearCoefficients& r = m_unsaturatedCoefficients;
    const EdgeShearCoefficients rates{rateOf(c.meanX, r.meanX), rateOf(c.meanY, r.meanY),
                                      rateOf(c.gradientX, r.gradientX), rateOf(c.gradientY, r.gradientY)};
    base::MatN<12> rate = (6.0 / (thickness * thickness)) * m_curvatureStiffness;
    rate += base::congruence(edgeShearForm(rates), m_edgeShears);
    return rate;
}

std::array<double, 2> ShellQuad4DrillingBending::shearForceRates(const std::array<double, 12>& u12, double thickness) const
{
    const std::array<double, 2> mean = m_meanShears * u12;
    return {12.0 * coefficientRate(m_coefficients.meanX, m_unsaturatedCoefficients.meanX, thickness) * mean[0],
            12.0 * coefficientRate(m_coefficients.meanY, m_unsaturatedCoefficients.meanY, thickness) * mean[1]};
}

base::MatN<12> ShellQuad4DrillingBending::thicknessRate(double thickness) const
{
    const auto rateOf = [&](double c, double unsaturated) { return coefficientRate(c, unsaturated, thickness); };
    const EdgeShearCoefficients& c = m_coefficients;
    const EdgeShearCoefficients& r = m_unsaturatedCoefficients;
    const EdgeShearCoefficients rates{rateOf(c.meanX, r.meanX), rateOf(c.meanY, r.meanY),
                                      rateOf(c.gradientX, r.gradientX), rateOf(c.gradientY, r.gradientY)};
    base::MatN<12> rate = (3.0 / thickness) * m_curvatureStiffness;
    rate += base::congruence(edgeShearForm(rates), m_edgeShears);
    return rate;
}

namespace
{

// The rate of a saturated edge-shear coefficient c = c_r/(1 + 12 c_r/g), g = t·κG (saturated()), for the rates dc_r
// and d(κG) at a fixed centre thickness t: dc = q² dc_r + 12 (c/g)² t d(κG), q = g/(g + 12 c_r); dc_r in a rigid
// direction (κG not positive: no saturation).
[[nodiscard]] double saturatedRate(double c, double unsaturated, double unsaturatedRate, double shearStiffness,
                                   double shearStiffnessRate, double centreThickness)
{
    if (!(shearStiffness > 0.0))
        return unsaturatedRate;
    const double g = centreThickness * shearStiffness;
    const double q = g / (g + TRANSVERSE_SHEAR_SATURATION_FACTOR * unsaturated);
    const double ratio = c / g;
    return q * q * unsaturatedRate +
           TRANSVERSE_SHEAR_SATURATION_FACTOR * ratio * ratio * centreThickness * shearStiffnessRate;
}

} // namespace

ShellQuad4DrillingBending::SectionRate ShellQuad4DrillingBending::sectionRate(const base::Mat3& bendingMatrixRate,
                                                            const TransverseShear& shearRate, bool withStiffness) const
{
    SectionRate rate;
    // The curvature part (see the header).
    base::MatN<5> h;
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const base::MatN<3, 5> p = momentInterpolation(point.xi, point.eta);
        const base::Mat3 flexibility = inverted(bendingStiffnessAt(point.xi, point.eta));
        const double t = thicknessAt(point.xi, point.eta, m_thickness);
        const base::Mat3 stiffnessRate = m_bendingRatio * t * t * t / 12.0 * bendingMatrixRate;
        const base::Mat3 flexibilityRate = -1.0 * (flexibility * (stiffnessRate * flexibility));
        const double scale = point.weight * m_geometry.jacobianAt(point.xi, point.eta).determinant;
        h += scale * (p.transposed() * (flexibilityRate * p));
    }
    const base::MatN<5, 12> hTimesParameters = h * m_momentParameters;
    rate.momentParameters = -1.0 * (m_momentFlexibility * hTimesParameters);
    if (withStiffness)
        rate.stiffness = -1.0 * (m_momentParameters.transposed() * hTimesParameters);

    // The rates of the edge shear's coefficients, through their saturation.
    const double t = m_centreThickness;
    const double centreRatio = m_bendingRatio * t * t * t / 12.0;
    const double d11Rate = centreRatio * bendingMatrixRate(0, 0);
    const double d22Rate = centreRatio * bendingMatrixRate(1, 1);
    const double a2 = m_geometry.widthA * m_geometry.widthA;
    const double b2 = m_geometry.widthB * m_geometry.widthB;
    const EdgeShearCoefficients& c = m_coefficients;
    const EdgeShearCoefficients& r = m_unsaturatedCoefficients;
    const std::array<double, 2>& s = m_shear.stiffness;
    const std::array<double, 2>& ds = shearRate.stiffness;
    rate.coefficients.meanX = saturatedRate(c.meanX, r.meanX, d11Rate / a2, s[0], ds[0], t);
    rate.coefficients.meanY = saturatedRate(c.meanY, r.meanY, d22Rate / b2, s[1], ds[1], t);
    rate.coefficients.gradientX =
        saturatedRate(c.gradientX, r.gradientX, d11Rate * (1.0 / a2 + EDGE_SHEAR_GRADIENT_FACTOR / b2), s[0], ds[0], t);
    rate.coefficients.gradientY =
        saturatedRate(c.gradientY, r.gradientY, d22Rate * (1.0 / b2 + EDGE_SHEAR_GRADIENT_FACTOR / a2), s[1], ds[1], t);
    if (withStiffness)
        rate.stiffness += base::congruence(edgeShearForm(rate.coefficients), m_edgeShears);
    return rate;
}

std::array<double, 2> ShellQuad4DrillingBending::shearForceRatesOf(const EdgeShearCoefficients& coefficientRates,
                                                          const std::array<double, 12>& u12) const
{
    const std::array<double, 2> mean = m_meanShears * u12;
    return {12.0 * coefficientRates.meanX * mean[0], 12.0 * coefficientRates.meanY * mean[1]};
}

base::Mat3 ShellQuad4DrillingBending::bendingStiffnessAt(double xi, double eta) const
{
    const double t = thicknessAt(xi, eta, m_thickness);
    return m_bendingRatio * t * t * t / 12.0 * m_bendingMatrix;
}

base::MatN<3, 5> ShellQuad4DrillingBending::momentInterpolation(double xi, double eta) const
{
    const base::MatN<2>& j0 = m_geometry.centreJacobian.j;
    const double r = m_geometry.centreJacobian.determinant / m_geometry.jacobianAt(xi, eta).determinant;
    const double xiLength = std::hypot(j0(0, 0), j0(0, 1));
    const double etaLength = std::hypot(j0(1, 0), j0(1, 1));
    const double ex = j0(0, 0) / xiLength;
    const double ey = j0(0, 1) / xiLength;
    const double fx = j0(1, 0) / etaLength;
    const double fy = j0(1, 1) / etaLength;
    base::MatN<3, 5> p;
    p(0, 0) = 1.0;
    p(1, 1) = 1.0;
    p(2, 2) = 1.0;
    p(0, 3) = eta * r * ex * ex;
    p(1, 3) = eta * r * ey * ey;
    p(2, 3) = eta * r * ex * ey;
    p(0, 4) = xi * r * fx * fx;
    p(1, 4) = xi * r * fy * fy;
    p(2, 4) = xi * r * fx * fy;
    return p;
}

std::array<double, 3> ShellQuad4DrillingBending::momentAt(double xi, double eta, const std::array<double, 12>& u12,
                                                 const std::array<double, 5>& initial) const
{
    return momentInterpolation(xi, eta) * momentParameters(u12, initial);
}

std::array<double, 5> ShellQuad4DrillingBending::momentParameters(const std::array<double, 12>& u12,
                                                         const std::array<double, 5>& initial) const
{
    std::array<double, 5> parameters = m_momentParameters * u12;
    for (std::size_t k = 0; k < 5; ++k)
        parameters[k] -= initial[k];
    return parameters;
}

ShellQuad4DrillingBending::InitialCurvatureLoads ShellQuad4DrillingBending::initialCurvatureLoads(
    const std::array<std::array<double, 3>, 4>& atGaussPoints) const
{
    std::array<double, 5> g{};
    const std::array<QuadraturePoint, 4> points = gaussRule2x2();
    for (std::size_t k = 0; k < 4; ++k)
    {
        const double scale = points[k].weight * m_geometry.jacobianAt(points[k].xi, points[k].eta).determinant;
        const std::array<double, 5> term =
            momentInterpolation(points[k].xi, points[k].eta).transposed() * atGaussPoints[k];
        for (std::size_t c = 0; c < 5; ++c)
            g[c] += scale * term[c];
    }
    InitialCurvatureLoads loads;
    loads.parameters = m_momentFlexibility * g;
    loads.nodal = m_gm.transposed() * loads.parameters;
    return loads;
}

std::array<double, 3> ShellQuad4DrillingBending::curvatureAt(double xi, double eta, const std::array<double, 12>& u12) const
{
    return inverted(bendingStiffnessAt(xi, eta)) * momentAt(xi, eta, u12);
}

std::array<double, 2> ShellQuad4DrillingBending::shearForces(const std::array<double, 12>& u12) const
{
    const std::array<double, 2> mean = m_meanShears * u12;
    return {12.0 * m_coefficients.meanX * mean[0], 12.0 * m_coefficients.meanY * mean[1]};
}

} // namespace oos::elements
