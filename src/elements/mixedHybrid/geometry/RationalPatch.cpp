#include "oos/elements/mixedHybrid/geometry/RationalPatch.hpp"

#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// Gauss points per edge of the fit of the centre control point.
constexpr std::size_t FIT_POINTS = 10;
// The unknowns of the fit: the three coordinates of the centre control point.
constexpr std::size_t UNKNOWNS = 3;
// The ridge added to the normal equations, relative to their mean diagonal term.
constexpr double RIDGE = 1e-12;
// The round-off below which the residual of the plain fit is not taken, relative to the residual before the fit.
constexpr double MISFIT_FLOOR = 1e-15;

// The slots (i, j) of the control points of the four edges.
constexpr std::array<std::pair<std::size_t, std::size_t>, 4> EDGE_SLOT{{{1, 0}, {2, 1}, {1, 2}, {0, 1}}};

// The angle of the normal n about the binormal bn of an edge, measured from bn × t in the plane normal to t.
double angleAbout(const Vector3& bn, const Vector3& n, const Vector3& t)
{
    const Vector3 p = cross3(bn, t);
    return std::atan2(dot3(n, bn), dot3(n, p));
}

// vᵀ A v for a 3 × 3 matrix: (vᵀ A) first, then the product with v.
double quadraticForm(const Matrix& a, const Vector3& v)
{
    double out = 0.0;
    for (std::size_t j = 0; j < UNKNOWNS; ++j)
    {
        double row = 0.0;
        for (std::size_t i = 0; i < UNKNOWNS; ++i)
            row += v[i] * a(i, j);
        out += row * v[j];
    }
    return out;
}

} // namespace

RationalPatch RationalPatch::fromEdges(const std::array<Vector3, 4>& corners, const std::array<Vector3, 4>& bulges)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return fromWeightedEdges(corners, bulges, {nan, nan, nan, nan});
}

RationalPatch RationalPatch::fromWeightedEdges(const std::array<Vector3, 4>& corners,
                                               const std::array<Vector3, 4>& bulges,
                                               const std::array<double, 4>& edgeWeights)
{
    RationalPatch patch;
    for (auto& row : patch.weights)
        row = {1.0, 1.0, 1.0};
    patch.controlPoints[0][0] = corners[0];
    patch.controlPoints[2][0] = corners[1];
    patch.controlPoints[2][2] = corners[2];
    patch.controlPoints[0][2] = corners[3];
    for (std::size_t k = 0; k < 4; ++k)
    {
        const Vector3 chord = sub3(corners[(k + 1) % 4], corners[k]);
        const Vector3 control = scl3(bulges[k], 2.0);
        const auto [i, j] = EDGE_SLOT[k];
        patch.controlPoints[i][j] = add3(add3(corners[k], scl3(chord, 0.5)), control);
        patch.weights[i][j] = std::isfinite(edgeWeights[k]) ? edgeWeights[k] : controlPolygonWeight(chord, bulges[k]);
    }
    const auto& p = patch.controlPoints;
    for (std::size_t c = 0; c < 3; ++c)
        patch.controlPoints[1][1][c] = 0.5 * (p[1][0][c] + p[1][2][c]) + 0.5 * (p[0][1][c] + p[2][1][c]) -
                                       0.25 * (corners[0][c] + corners[1][c] + corners[2][c] + corners[3][c]);
    const auto& w = patch.weights;
    patch.weights[1][1] = 0.25 * (w[1][0] + w[1][2]) * (w[0][1] + w[2][1]);
    patch.corners = corners;
    patch.bulges = bulges;
    return patch;
}

RationalPatch::Homogeneous RationalPatch::homogeneous(double s, double t) const
{
    // the quadratic Bernstein polynomials of s and t and their derivatives
    const std::array<double, 3> bs{(1.0 - s) * (1.0 - s), 2.0 * s * (1.0 - s), s * s};
    const std::array<double, 3> bt{(1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t};
    const std::array<double, 3> ds{-2.0 * (1.0 - s), 2.0 - 4.0 * s, 2.0 * s};
    const std::array<double, 3> dt{-2.0 * (1.0 - t), 2.0 - 4.0 * t, 2.0 * t};
    Homogeneous out;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            const double w = weights[i][j];
            const double b = w * bs[i] * bt[j];
            const double bS = w * ds[i] * bt[j];
            const double bT = w * bs[i] * dt[j];
            out.denominator += b;
            out.denominatorDerivatives[0] += bS;
            out.denominatorDerivatives[1] += bT;
            for (std::size_t c = 0; c < 3; ++c)
            {
                out.numerator[c] += b * controlPoints[i][j][c];
                out.numeratorDerivatives[0][c] += bS * controlPoints[i][j][c];
                out.numeratorDerivatives[1][c] += bT * controlPoints[i][j][c];
            }
        }
    return out;
}

void RationalPatch::fitCentre()
{
    std::array<Vector3, 4> cornerNormal{};
    for (std::size_t k = 0; k < 4; ++k)
        cornerNormal[k] = frame(*this, REFERENCE_SQUARE[k][0], REFERENCE_SQUARE[k][1]).normal;
    const LineRule rule = gaussLegendre(FIT_POINTS);
    Matrix ata(UNKNOWNS, UNKNOWNS);
    Vector atb(UNKNOWNS, 0.0);
    // the sum of the weights, and of the squares of the residual before the fit
    double weightSum = 0.0;
    double residualBefore = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        const Vector3& na = cornerNormal[k];
        const Vector3& nb = cornerNormal[(k + 1) % 4];
        const Vector3 chord = unit3(sub3(corners[(k + 1) % 4], corners[k]));
        // the binormal of the plane of the edge: that of its bulge, or of the normal at its first grid
        const Vector3 bn = norm3(bulges[k]) > 0.0 ? unit3(cross3(chord, bulges[k])) : unit3(cross3(chord, na));
        const Vector3 v0 = edgeEvaluation(*this, k, 0.0).velocity;
        const Vector3 v1 = edgeEvaluation(*this, k, 1.0).velocity;
        const double ba = angleAbout(bn, na, unit3(v0));
        double bb = angleAbout(bn, nb, unit3(v1));
        // the twist from one grid to the other, within half a turn
        constexpr double TWO_PI = 2.0 * std::numbers::pi;
        while (bb - ba > std::numbers::pi)
            bb -= TWO_PI;
        while (bb - ba < -std::numbers::pi)
            bb += TWO_PI;
        const ReferencePoint from = REFERENCE_SQUARE[k];
        const ReferencePoint to = REFERENCE_SQUARE[(k + 1) % 4];
        const std::array<double, 2> d{to[0] - from[0], to[1] - from[1]};
        // the inward direction in the parameters
        const std::array<double, 2> inward{-d[1], d[0]};
        for (std::size_t q = 0; q < rule.points.size(); ++q)
        {
            const double tau = 0.5 * (rule.points[q] + 1.0);
            const std::array<double, 2> uv = referenceEdgePoint(4, k, tau);
            const Vector3 velocity = edgeEvaluation(*this, k, tau).velocity;
            const Vector3 t = unit3(velocity);
            const double beta = ba + (bb - ba) * tau;
            const Vector3 target = add3(scl3(cross3(bn, t), std::cos(beta)), scl3(bn, std::sin(beta)));
            const double s = 0.5 * (1.0 + uv[0]);
            const double tt = 0.5 * (1.0 + uv[1]);
            const Homogeneous h = homogeneous(s, tt);
            // the unnormalised inward derivative of the patch: N_in D − N D_in
            Vector3 g{};
            for (std::size_t c = 0; c < 3; ++c)
                g[c] = (h.numeratorDerivatives[0][c] * inward[0] + h.numeratorDerivatives[1][c] * inward[1]) *
                           h.denominator -
                       h.numerator[c] *
                           (h.denominatorDerivatives[0] * inward[0] + h.denominatorDerivatives[1] * inward[1]);
            // its derivative with respect to the centre control point (times the identity): w11 B11_in D
            const double b11Inward = (2.0 - 4.0 * s) * 2.0 * tt * (1.0 - tt) * inward[0] +
                                     2.0 * s * (1.0 - s) * (2.0 - 4.0 * tt) * inward[1];
            const double scaleOfG = 1.0 / norm3(g);
            const double residual = dot3(target, g) * scaleOfG;
            const double k0 = weights[1][1] * b11Inward * h.denominator * scaleOfG;
            const Vector3 row{k0 * target[0], k0 * target[1], k0 * target[2]};
            const double w = rule.weights[q] * norm3(velocity);
            weightSum += w;
            residualBefore += w * residual * residual;
            for (std::size_t i = 0; i < UNKNOWNS; ++i)
            {
                atb[i] -= w * row[i] * residual;
                for (std::size_t j = 0; j < UNKNOWNS; ++j)
                    ata(i, j) += w * row[i] * row[j];
            }
        }
    }
    // Along the normal of the patch at its centre the point changes the shape, and the normals fix it: one unknown,
    // always well posed (zero on a flat element).
    const Vector3 nc = frame(*this, 0.0, 0.0).normal;
    const double den = quadraticForm(ata, nc);
    if (!(den > 0.0) || weightSum == 0.0)
        return;
    const double alpha = (nc[0] * atb[0] + nc[1] * atb[1] + nc[2] * atb[2]) / den;
    // In the surface the point changes the parameters of the patch, and the shape only through the curvature: the
    // normals see it weakly, or not at all (along the generators of a cylinder), and a plain fit of the three
    // coordinates sends it away by the size of the element to cancel a misfit of a thousandth of a degree. The
    // in-surface move is therefore the estimate of a quantity expected of the order of the normal move, from normals
    // known to within the misfit that no position of the point removes (the residual of the plain fit): where the
    // normals can be met exactly (a sphere) it is the plain fit, where they cannot it stays small.
    Vector solution{nc[0] * alpha, nc[1] * alpha, nc[2] * alpha};
    const double ridge = RIDGE * trace(ata) / static_cast<double>(UNKNOWNS);
    Matrix plain = ata;
    for (std::size_t i = 0; i < UNKNOWNS; ++i)
        plain(i, i) += ridge;
    const std::optional<Cholesky> plainFactor = Cholesky::factor(plain);
    if (plainFactor && alpha != 0.0)
    {
        const Vector free = plainFactor->solve(atb);
        // the mean square of the residual that remains, not below the round-off of the residual before the fit
        const double freeDotAtb = free[0] * atb[0] + free[1] * atb[1] + free[2] * atb[2];
        const double misfit = (std::fmax(residualBefore - freeDotAtb, 0.0) + MISFIT_FLOOR * residualBefore) / weightSum;
        const double penalty = weightSum * misfit / (alpha * alpha);
        // the penalty acts on the in-surface move only: I − nc ncᵀ
        Matrix regularised = ata;
        for (std::size_t i = 0; i < UNKNOWNS; ++i)
        {
            for (std::size_t j = 0; j < UNKNOWNS; ++j)
                regularised(i, j) += penalty * ((i == j ? 1.0 : 0.0) - nc[i] * nc[j]);
            regularised(i, i) += ridge;
        }
        if (const std::optional<Cholesky> factor = Cholesky::factor(regularised))
            solution = factor->solve(atb);
    }
    for (std::size_t i = 0; i < 3; ++i)
        controlPoints[1][1][i] += solution[i];
}

} // namespace oos::elements::mixedHybrid
