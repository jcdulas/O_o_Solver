#include "oos/elements/solids/solidHexa/SolidHexaIncompatibleModes.hpp"

#include "oos/base/math/MatN.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/core/SolidIntegrationRules.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"
#include "oos/elements/solids/SolidIsoparametric.hpp"
#include "oos/elements/solids/SolidKinematics.hpp"

#include <array>
#include <cmath>

namespace oos::elements
{

namespace
{

constexpr SolidShape SHAPE = SolidShape::Hexa8;
constexpr std::size_t CORNERS = 8;
constexpr std::size_t DOFS = 3 * CORNERS;
// Three normal and four shear modes for each cyclic pair of parent axes (EAS-21).
constexpr std::size_t MODES_PER_AXIS = 7;
constexpr std::size_t MODES = 3 * MODES_PER_AXIS;
// The 2 × 2 × 2 Gauss rule of the element.
constexpr std::size_t GAUSS_POINTS_PER_AXIS = 2;
constexpr std::size_t GAUSS_POINTS = 8;

using Mat6x24 = base::MatN<6, DOFS>;
using ModeStrains = base::MatN<6, MODES>;
using Mat6x8 = base::MatN<6, CORNERS>;
using ModeMatrix = base::MatN<MODES, MODES>;
using ModeDisplacements = base::MatN<MODES, DOFS>;
using ModeTemperatures = base::MatN<MODES, CORNERS>;
using Mat24 = base::MatN<DOFS, DOFS>;
using Mat24x8 = base::MatN<DOFS, CORNERS>;

// Engineering strain of the symmetric product of two parent-coordinate gradients.
[[nodiscard]] Voigt6 strainDirection(const base::Vec3& a, const base::Vec3& b)
{
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[0] * b[1] + a[1] * b[0],
            a[1] * b[2] + a[2] * b[1], a[0] * b[2] + a[2] * b[0]};
}

// What one Gauss point contributes: B (compatible strains), G (strains of the modes), N (to interpolate ΔT) and
// |det J|·w.
struct GaussPoint
{
    ParentPoint point;
    Mat6x24 compatible;
    ModeStrains modes;
    std::array<double, CORNERS> shape{};
    double weight = 0.0;
};

// The 2 × 2 × 2 points of an element: B and G in the material system.
[[nodiscard]] std::array<GaussPoint, GAUSS_POINTS> gaussPointsOf(std::span<const base::Vec3> x)
{
    // Taylor's correction: the mode derivatives through J0⁻¹ (column k of J0⁻¹ is the gradient of the parent
    // coordinate ξ_k at the centre) and scaled by det J0 / det J.
    const ShapeGradients centre = shapeGradientsAt(SHAPE, x, solidCentre(SHAPE));
    const base::Mat3 centreInverse = base::inverse(centre.jacobian);
    std::array<base::Vec3, 3> parentGradients{};
    for (std::size_t k = 0; k < 3; ++k)
        parentGradients[k] = base::normalized({centreInverse(0, k), centreInverse(1, k), centreInverse(2, k)});
    // Column scaling cancels in the condensation. Unit directions avoid powers of the element aspect ratio.
    std::array<Voigt6, 3> normal{}, shear{};
    for (std::size_t k = 0; k < 3; ++k)
    {
        normal[k] = strainDirection(parentGradients[k], parentGradients[k]);
        shear[k] = strainDirection(parentGradients[k], parentGradients[(k + 1) % 3]);
    }

    std::array<GaussPoint, GAUSS_POINTS> points{};
    SolidOperator compatible;
    std::size_t g = 0;
    for (const SolidQuadraturePoint& quadrature : hexahedronRule(GAUSS_POINTS_PER_AXIS).all())
    {
        GaussPoint& point = points[g++];
        const ShapeGradients gradients = shapeGradientsAt(SHAPE, x, quadrature.point);
        point.point = quadrature.point;
        point.weight = std::abs(gradients.determinant) * quadrature.weight;
        for (std::size_t a = 0; a < CORNERS; ++a)
            point.shape[a] = gradients.shape.n[a];
        fillStrainOperator(gradients, compatible);
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t j = 0; j < DOFS; ++j)
                point.compatible(r, j) = compatible(r, j);
        const double ratio = centre.determinant / gradients.determinant;
        const std::array<double, 3> parent{quadrature.point.xi, quadrature.point.eta, quadrature.point.zeta};
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double a = parent[k];
            const double b = parent[(k + 1) % 3];
            const double c = parent[(k + 2) % 3];
            const std::array<double, MODES_PER_AXIS> polynomial{a, a * b, a * c, a, b, a * c, b * c};
            for (std::size_t m = 0; m < MODES_PER_AXIS; ++m)
            {
                const Voigt6& direction = m < 3 ? normal[k] : shear[k];
                for (std::size_t r = 0; r < 6; ++r)
                    point.modes(r, MODES_PER_AXIS * k + m) = ratio * polynomial[m] * direction[r];
            }
        }
    }
    return points;
}

// The condensation of the modes: K, α = −condensation·u + thermalModes·ΔT, and the nodal loads of the modes under a
// thermal strain per unit change of each node.
struct Condensation
{
    Mat24 stiffness;            // K_uu − K_uα·K_αα⁻¹·K_αu
    ModeDisplacements modesOfU; // K_αα⁻¹·K_αu
    ModeTemperatures modesOfT;  // K_αα⁻¹·F_α, f_α = F_α·ΔT
    Mat24x8 thermalLoad;        // −K_uα·K_αα⁻¹·F_α
};

[[nodiscard]] Condensation condense(const std::array<GaussPoint, GAUSS_POINTS>& points, const Mat6& d,
                                    const Voigt6& expansion)
{
    Mat24 kuu;
    ModeDisplacements kau;
    ModeMatrix kaa;
    ModeTemperatures fa;
    std::array<double, 6> thermalStress{}; // D·a
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t s = 0; s < 6; ++s)
            thermalStress[r] += d(r, s) * expansion[s];
    for (const GaussPoint& point : points)
    {
        const Mat6x24 db = d * point.compatible;
        const ModeStrains dg = d * point.modes;
        kuu += point.weight * (point.compatible.transposed() * db);
        kau += point.weight * (point.modes.transposed() * db);
        kaa += point.weight * (point.modes.transposed() * dg);
        for (std::size_t m = 0; m < MODES; ++m)
        {
            double work = 0.0; // G(:, m)·D·a
            for (std::size_t r = 0; r < 6; ++r)
                work += point.modes(r, m) * thermalStress[r];
            for (std::size_t a = 0; a < CORNERS; ++a)
                fa(m, a) += point.weight * work * point.shape[a];
        }
    }
    const ModeMatrix kaaInverse = inverted(kaa);
    Condensation c;
    c.modesOfU = kaaInverse * kau;
    c.modesOfT = kaaInverse * fa;
    c.stiffness = kuu - kau.transposed() * c.modesOfU;
    c.thermalLoad = -1.0 * (kau.transposed() * c.modesOfT);
    return c;
}

template <std::size_t Columns>
void copyOperator(const base::MatN<6, Columns>& from, SolidOperator& to)
{
    to.reset(Columns);
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t j = 0; j < Columns; ++j)
            to(r, j) = from(r, j);
}

void fillStiffness(const Condensation& c, SolidResponse& response)
{
    response.stiffness.reset(DOFS);
    for (std::size_t i = 0; i < DOFS; ++i)
        for (std::size_t j = 0; j < DOFS; ++j)
            response.stiffness(i, j) = c.stiffness(i, j);
    for (std::size_t i = 0; i < DOFS; ++i)
        for (std::size_t a = 0; a < CORNERS; ++a)
            response.thermalLoad[i * CORNERS + a] = c.thermalLoad(i, a);
}

void fillRecovery(std::span<const base::Vec3> x, const Mat6& d, const std::array<GaussPoint, GAUSS_POINTS>& points,
                  const Condensation& c, SolidResponse& response)
{
    // The total strain at each Gauss point: (B − G·K_αα⁻¹·K_αu)·u + G·K_αα⁻¹·F_α·ΔT.
    std::array<Mat6x24, GAUSS_POINTS> strainOfU{};
    std::array<Mat6x8, GAUSS_POINTS> strainOfT{};
    for (std::size_t g = 0; g < GAUSS_POINTS; ++g)
    {
        strainOfU[g] = points[g].compatible - points[g].modes * c.modesOfU;
        strainOfT[g] = points[g].modes * c.modesOfT;
    }
    const std::span<const ParentPoint> corners = solidCornerPoints(SHAPE);
    response.pointCount = 1 + corners.size();

    // The centre: the modes' strain vanishes there (ξ_k = 0).
    SolidOperator b;
    fillStrainOperator(shapeGradientsAt(SHAPE, x, solidCentre(SHAPE)), b);
    Mat6x24 centre;
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t j = 0; j < DOFS; ++j)
            centre(r, j) = b(r, j);
    copyOperator(centre, response.strain[0]);
    copyOperator(d * centre, response.stress[0]);
    copyOperator(Mat6x8{}, response.thermalStrain[0]);

    // The corners: Σ_g T_g(corner)·ε_g.
    for (std::size_t c = 0; c < corners.size(); ++c)
    {
        Mat6x24 strain;
        Mat6x8 thermal;
        for (std::size_t g = 0; g < GAUSS_POINTS; ++g)
        {
            const double weight = hexaGaussExtrapolationWeight(points[g].point, corners[c]);
            strain += weight * strainOfU[g];
            thermal += weight * strainOfT[g];
        }
        copyOperator(strain, response.strain[1 + c]);
        copyOperator(d * strain, response.stress[1 + c]);
        copyOperator(thermal, response.thermalStrain[1 + c]);
    }
}

} // namespace

void buildSolidHexaIncompatibleModesResponse(std::span<const base::Vec3> positions, const Mat6& elasticity,
                                            const Voigt6& expansion, SolidResponseParts parts, SolidResponse& response)
{
    const std::array<GaussPoint, GAUSS_POINTS> points = gaussPointsOf(positions);
    const Condensation condensation = condense(points, elasticity, expansion);
    response.hasThermalModes = true;
    if (parts == SolidResponseParts::Stiffness)
        fillStiffness(condensation, response);
    else
        fillRecovery(positions, elasticity, points, condensation, response);
}

} // namespace oos::elements
