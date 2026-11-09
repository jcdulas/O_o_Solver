#include "oos/elements/solids/SolidIsoparametric.hpp"

#include "oos/elements/solids/SolidKinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace oos::elements
{

namespace
{

// The trilinear extrapolation of the 2 × 2 × 2 Gauss points to the corners of the parent hexahedron: the corners are
// at √3 times the points in the parent coordinates of the cell the points span.
const double GAUSS_TO_CORNER_SCALE = std::sqrt(3.0);

// The Gauss points per axis of the values extrapolated to the corners (CornerRecovery).
constexpr std::size_t EXTRAPOLATION_POINTS_PER_AXIS = 2;

// The corners of the parent hexahedron in the order of the card (G1-G4 at ζ = −1, G5-G8 at ζ = +1), and those of a
// pyramid, the hexahedron whose four corners at ζ = +1 are collapsed into its apex G5.
constexpr std::size_t HEXA_CORNERS = 8;
constexpr std::size_t PYRAMID_CORNERS = 5;
constexpr std::size_t PYRAMID_BASE_CORNERS = 4;
constexpr std::array<ParentPoint, HEXA_CORNERS> HEXA_CORNER_POINTS{{{-1.0, -1.0, -1.0},
                                                                   {1.0, -1.0, -1.0},
                                                                   {1.0, 1.0, -1.0},
                                                                   {-1.0, 1.0, -1.0},
                                                                   {-1.0, -1.0, 1.0},
                                                                   {1.0, -1.0, 1.0},
                                                                   {1.0, 1.0, 1.0},
                                                                   {-1.0, 1.0, 1.0}}};

// D·B for one point, and the product Bᵀ·(D·B)·weight added to K.
void addCongruence(const SolidOperator& b, const Mat6& d, double weight, SolidMatrix& k)
{
    const std::size_t n = b.columns;
    SolidOperator db;
    db.reset(n);
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (std::size_t s = 0; s < 6; ++s)
                sum += d(r, s) * b(s, j);
            db(r, j) = sum;
        }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 6; ++r)
                sum += b(r, i) * db(r, j);
            k(i, j) += weight * sum;
        }
}

// The stress operator D·B and the strain operator B at a point, scaled by `scale` and added to the operators.
void addRecoveryAt(const ShapeGradients& gradients, const Mat6& d, double scale, SolidOperator& stress,
                   SolidOperator& strain)
{
    SolidOperator b;
    fillStrainOperator(gradients, b);
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t j = 0; j < b.columns; ++j)
        {
            double sum = 0.0;
            for (std::size_t s = 0; s < 6; ++s)
                sum += d(r, s) * b(s, j);
            stress(r, j) += scale * sum;
            strain(r, j) += scale * b(r, j);
        }
}

void buildRecovery(SolidShape shape, std::span<const base::Vec3> positions, const Mat6& d, CornerRecovery corners,
                   SolidResponse& response)
{
    const std::size_t dofs = 3 * solidNodeCount(shape);
    const std::span<const ParentPoint> cornerPoints = solidCornerPoints(shape);
    response.pointCount = 1 + cornerPoints.size();
    for (std::size_t p = 0; p < response.pointCount; ++p)
    {
        response.stress[p].reset(dofs);
        response.strain[p].reset(dofs);
    }
    if (corners != CornerRecovery::AllFromHexaGauss)
        addRecoveryAt(shapeGradientsAt(shape, positions, solidCentre(shape)), d, 1.0, response.stress[0],
                      response.strain[0]);
    if (corners == CornerRecovery::Direct)
    {
        for (std::size_t c = 0; c < cornerPoints.size(); ++c)
            addRecoveryAt(shapeGradientsAt(shape, positions, cornerPoints[c]), d, 1.0, response.stress[1 + c],
                          response.strain[1 + c]);
        return;
    }
    // From the Gauss points: corner c of the parent hexahedron gets Σ_g T_g(c)·value_g (hexaGaussExtrapolationWeight);
    // a pyramid's collapsed corners (ζ = 1) all go into its apex, with a weight ¼ each; with AllFromHexaGauss the
    // centre gets T_g(0) = ⅛ of each point.
    const bool isPyramid = cornerPoints.size() == PYRAMID_CORNERS;
    for (const SolidQuadraturePoint& point : hexahedronRule(EXTRAPOLATION_POINTS_PER_AXIS).all())
    {
        const ShapeGradients gradients = shapeGradientsAt(shape, positions, point.point);
        if (corners == CornerRecovery::AllFromHexaGauss)
            addRecoveryAt(gradients, d, hexaGaussExtrapolationWeight(point.point, ParentPoint{}), response.stress[0],
                          response.strain[0]);
        for (std::size_t c = 0; c < HEXA_CORNERS; ++c)
        {
            const double t = hexaGaussExtrapolationWeight(point.point, HEXA_CORNER_POINTS[c]);
            const bool isCollapsed = isPyramid && c >= PYRAMID_BASE_CORNERS;
            const std::size_t target = isCollapsed ? PYRAMID_BASE_CORNERS : c;
            const double share = isCollapsed ? 0.25 : 1.0;
            addRecoveryAt(gradients, d, t * share, response.stress[1 + target], response.strain[1 + target]);
        }
    }
}

} // namespace

double hexaGaussExtrapolationWeight(const ParentPoint& point, const ParentPoint& place)
{
    const double sx = point.xi > 0.0 ? 1.0 : -1.0;
    const double sy = point.eta > 0.0 ? 1.0 : -1.0;
    const double sz = point.zeta > 0.0 ? 1.0 : -1.0;
    return 0.125 * (1.0 + sx * GAUSS_TO_CORNER_SCALE * place.xi) * (1.0 + sy * GAUSS_TO_CORNER_SCALE * place.eta) *
           (1.0 + sz * GAUSS_TO_CORNER_SCALE * place.zeta);
}

void buildIsoparametricResponse(SolidShape shape, std::span<const base::Vec3> positions, const Mat6& elasticity,
                                const SolidRule& rule, CornerRecovery corners, SolidResponseParts parts,
                                SolidResponse& response)
{
    if (parts == SolidResponseParts::Recovery)
    {
        buildRecovery(shape, positions, elasticity, corners, response);
        return;
    }
    const std::size_t dofs = 3 * solidNodeCount(shape);
    response.stiffness.reset(dofs);
    SolidOperator b;
    for (const SolidQuadraturePoint& point : rule.all())
    {
        const ShapeGradients gradients = shapeGradientsAt(shape, positions, point.point);
        fillStrainOperator(gradients, b);
        addCongruence(b, elasticity, std::abs(gradients.determinant) * point.weight, response.stiffness);
    }
}

void buildScalarMass(SolidShape shape, std::span<const base::Vec3> positions, double density, const SolidRule& rule,
                     SolidMatrix& mass)
{
    const std::size_t n = solidNodeCount(shape);
    mass.reset(n);
    for (const SolidQuadraturePoint& point : rule.all())
    {
        const ShapeGradients gradients = shapeGradientsAt(shape, positions, point.point);
        const double weight = density * std::abs(gradients.determinant) * point.weight;
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t c = 0; c < n; ++c)
                mass(a, c) += weight * gradients.shape.n[a] * gradients.shape.n[c];
    }
}

double solidVolume(SolidShape shape, std::span<const base::Vec3> positions, const SolidRule& rule)
{
    double volume = 0.0;
    for (const SolidQuadraturePoint& point : rule.all())
        volume += std::abs(shapeGradientsAt(shape, positions, point.point).determinant) * point.weight;
    return volume;
}

int solidJacobianSign(SolidShape shape, std::span<const base::Vec3> positions, const SolidRule& rule, bool withCorners,
                      double relativeTolerance)
{
    double smallest = 0.0;
    double largest = 0.0;
    double sum = 0.0;
    double weights = 0.0;
    bool isFirst = true;
    const auto visit = [&](ParentPoint point, double weight)
    {
        const double det = shapeGradientsAt(shape, positions, point).determinant;
        smallest = isFirst ? det : std::min(smallest, det);
        largest = isFirst ? det : std::max(largest, det);
        isFirst = false;
        sum += std::abs(det) * weight;
        weights += weight;
    };
    for (const SolidQuadraturePoint& point : rule.all())
        visit(point.point, point.weight);
    if (withCorners)
        for (const ParentPoint& corner : solidCornerPoints(shape))
            visit(corner, 0.0);
    const double floor = relativeTolerance * sum / weights;
    if (smallest > floor)
        return 1;
    if (largest < -floor)
        return -1;
    return 0;
}

void buildInitialStressStiffness(SolidShape shape, std::span<const base::Vec3> positions, const Voigt6& stress,
                                 const SolidRule& rule, SolidMatrix& differential)
{
    const std::size_t n = solidNodeCount(shape);
    differential.reset(3 * n);
    const base::Mat3 sigma = base::matrixFromRows({stress[0], stress[3], stress[5]}, {stress[3], stress[1], stress[4]},
                                                  {stress[5], stress[4], stress[2]});
    for (const SolidQuadraturePoint& point : rule.all())
    {
        const ShapeGradients gradients = shapeGradientsAt(shape, positions, point.point);
        const double weight = std::abs(gradients.determinant) * point.weight;
        for (std::size_t a = 0; a < n; ++a)
        {
            const base::Vec3 flux = sigma * gradients.gradients[a];
            for (std::size_t b = 0; b < n; ++b)
            {
                const double term = weight * base::dot(flux, gradients.gradients[b]);
                for (std::size_t i = 0; i < 3; ++i)
                    differential(3 * a + i, 3 * b + i) += term;
            }
        }
    }
}

} // namespace oos::elements
