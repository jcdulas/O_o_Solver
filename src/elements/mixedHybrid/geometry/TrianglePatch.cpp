#include "oos/elements/mixedHybrid/geometry/TrianglePatch.hpp"

#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"

#include <cmath>
#include <optional>

namespace oos::elements::mixedHybrid
{

namespace
{

// Gauss points per edge of the fit of the interior.
constexpr std::size_t FIT_POINTS = 10;
// The unknowns of the fit: the amplitudes of the three interior vectors.
constexpr std::size_t UNKNOWNS = 3;
// The gradients (∂/∂ξ, ∂/∂η) of the barycentric coordinates L_0, L_1, L_2.
constexpr std::array<std::array<double, 2>, 3> BARYCENTRIC_GRADIENT{{{-0.5, -0.5}, {0.5, 0.0}, {0.0, 0.5}}};

} // namespace

TrianglePatch TrianglePatch::fromEdges(const std::array<Vector3, 3>& corners, const std::array<Vector3, 3>& bulges,
                                       const std::array<double, 3>& edgeWeights)
{
    TrianglePatch patch;
    patch.corners = corners;
    patch.bulges = bulges;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const Vector3 chord = sub3(corners[(k + 1) % 3], corners[k]);
        const Vector3 control = scl3(bulges[k], 2.0);
        patch.controlPoints[k] = add3(add3(corners[k], scl3(chord, 0.5)), control);
        patch.weights[k] = std::isfinite(edgeWeights[k]) ? edgeWeights[k] : controlPolygonWeight(chord, bulges[k]);
    }
    return patch;
}

void TrianglePatch::fitInterior()
{
    fitInteriorTerms(false);
}

void TrianglePatch::fitInteriorRational()
{
    interiorInsideRationalForm = true;
    fitInteriorTerms(true);
}

void TrianglePatch::fitInteriorTerms(bool insideRationalForm)
{
    std::array<Vector3, 3> cornerNormal{};
    for (std::size_t k = 0; k < 3; ++k)
        cornerNormal[k] = frame(*this, REFERENCE_TRIANGLE[k][0], REFERENCE_TRIANGLE[k][1]).normal;
    const ReferencePoint centre = referenceCentre(3);
    const Vector3 nc = frame(*this, centre[0], centre[1]).normal;
    const LineRule rule = gaussLegendre(FIT_POINTS);
    Matrix ata(UNKNOWNS, UNKNOWNS);
    Vector atb(UNKNOWNS, 0.0);
    double residualBefore = 0.0;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::size_t i = k;
        const std::size_t j = (k + 1) % 3;
        const std::size_t o = (k + 2) % 3;
        const Vector3& na = cornerNormal[i];
        const Vector3& nb = cornerNormal[j];
        // the cosine of half the angle of the two normals
        const double wn = std::sqrt(std::fmax(0.5 * (1.0 + dot3(na, nb)), 0.0));
        if (!(wn > 0.0))
            return;
        const std::array<double, 2> d{REFERENCE_TRIANGLE[j][0] - REFERENCE_TRIANGLE[i][0],
                                      REFERENCE_TRIANGLE[j][1] - REFERENCE_TRIANGLE[i][1]};
        // the inward direction in the parameters, and the derivative of L_o along it
        const std::array<double, 2> inward{-d[1], d[0]};
        const double dlo = BARYCENTRIC_GRADIENT[o][0] * inward[0] + BARYCENTRIC_GRADIENT[o][1] * inward[1];
        for (std::size_t q = 0; q < rule.points.size(); ++q)
        {
            const double tau = 0.5 * (rule.points[q] + 1.0);
            const std::array<double, 2> uv = referenceEdgePoint(3, k, tau);
            const std::array<Vector3, 2> a = tangents(*this, uv[0], uv[1]).tangents;
            const Vector3 velocity = add3(scl3(a[0], d[0]), scl3(a[1], d[1]));
            const Vector3 t = unit3(velocity);
            // the normal turned from n_i to n_j about a fixed axis, with the parameter of a circular arc
            const Vector3 turned =
                add3(add3(scl3(na, (1.0 - tau) * (1.0 - tau)), scl3(add3(na, nb), tau * (1.0 - tau) / wn)),
                     scl3(nb, tau * tau));
            const Vector3 target = unit3(sub3(turned, scl3(t, dot3(turned, t))));
            // the inward derivative of the patch, less its part along the edge: what is left does not depend on the
            // direction taken in the parameters, and the three edges are weighed alike
            Vector3 g = add3(scl3(a[0], inward[0]), scl3(a[1], inward[1]));
            g = sub3(g, scl3(t, dot3(g, t)));
            const double scaleOfG = 1.0 / norm3(g);
            const double residual = dot3(target, g) * scaleOfG;
            double base = dlo * (1.0 - tau) * tau * dot3(target, nc) * scaleOfG;
            if (insideRationalForm)
            {
                // the denominator of the edge conic at tau
                const double wd = (1.0 - tau) * (1.0 - tau) + 2.0 * weights[k] * tau * (1.0 - tau) + tau * tau;
                base /= wd;
            }
            Vector3 row{};
            row[i] = base * (1.0 - tau);
            row[j] = base * tau;
            const double w = rule.weights[q] * norm3(velocity);
            residualBefore += w * residual * residual;
            for (std::size_t r = 0; r < UNKNOWNS; ++r)
            {
                atb[r] -= w * row[r] * residual;
                for (std::size_t c = 0; c < UNKNOWNS; ++c)
                    ata(r, c) += w * row[r] * row[c];
            }
        }
    }
    if (residualBefore == 0.0)
        return;
    const std::optional<Cholesky> factor = Cholesky::factor(ata);
    if (!factor)
        return;
    const Vector alpha = factor->solve(atb);
    if (!(std::isfinite(alpha[0]) && std::isfinite(alpha[1]) && std::isfinite(alpha[2])))
        return;
    for (std::size_t m = 0; m < 3; ++m)
        interior[m] = scl3(nc, alpha[m]);
}

} // namespace oos::elements::mixedHybrid
