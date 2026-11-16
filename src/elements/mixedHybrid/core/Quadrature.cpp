#include "oos/elements/mixedHybrid/core/Quadrature.hpp"

#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace oos::elements::mixedHybrid
{

namespace
{

// Newton's method on P_n converges quadratically from its estimate: a few iterations reach the round-off.
constexpr int MAX_NEWTON_ITERATIONS = 100;
constexpr double NEWTON_STEP_TOLERANCE = 1e-16;

// The values (P_n(z), P_{n−1}(z)) by the Legendre recurrence (n ≥ 1).
struct LegendrePair
{
    double pn = 0.0;
    double previous = 0.0;
};

LegendrePair legendrePair(std::size_t n, double z)
{
    double p0 = 1.0;
    double p1 = z;
    for (std::size_t k = 2; k <= n; ++k)
    {
        const double kf = static_cast<double>(k);
        const double p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    return {p1, p0};
}

} // namespace

LineRule gaussLegendre(std::size_t n)
{
    LineRule rule{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
    const double nf = static_cast<double>(n);
    for (std::size_t i = 0; i < (n + 1) / 2; ++i)
    {
        double z = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (nf + 0.5));
        double derivative = 1.0;
        for (int iteration = 0; iteration < MAX_NEWTON_ITERATIONS; ++iteration)
        {
            const LegendrePair p = legendrePair(n, z);
            derivative = nf * (z * p.pn - p.previous) / (z * z - 1.0);
            const double step = p.pn / derivative;
            z -= step;
            if (std::fabs(step) < NEWTON_STEP_TOLERANCE)
                break;
        }
        rule.points[i] = -z;
        rule.points[n - 1 - i] = z;
        const double weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
        rule.weights[i] = weight;
        rule.weights[n - 1 - i] = weight;
    }
    if (n % 2 == 1)
        rule.points[n / 2] = 0.0;
    return rule;
}

LineRule gaussJacobi(std::size_t n, double alpha)
{
    Matrix jacobiMatrix(n, n);
    for (std::size_t k = 0; k < n; ++k)
    {
        const double kf = static_cast<double>(k);
        const double den = (2.0 * kf + alpha) * (2.0 * kf + alpha + 2.0);
        jacobiMatrix(k, k) = den != 0.0 ? -alpha * alpha / den : 0.0;
        if (k + 1 < n)
        {
            const double m = kf + 1.0;
            const double c = 2.0 * m + alpha;
            const double b = 2.0 * m * (m + alpha) / (c * std::sqrt(c * c - 1.0));
            jacobiMatrix(k, k + 1) = b;
            jacobiMatrix(k + 1, k) = b;
        }
    }
    const SymmetricEigen eigen = jacobiEigen(jacobiMatrix);
    const double mu0 = std::pow(2.0, alpha + 1.0) / (alpha + 1.0);
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return eigen.values[a] < eigen.values[b]; });
    LineRule rule{std::vector<double>(n), std::vector<double>(n)};
    for (std::size_t i = 0; i < n; ++i)
    {
        const double first = eigen.vectors(0, order[i]);
        rule.points[i] = eigen.values[order[i]];
        rule.weights[i] = mu0 * first * first;
    }
    return rule;
}

std::vector<TriangleRulePoint> triangleRule(std::size_t n)
{
    const LineRule gauss = gaussLegendre(n);
    std::vector<TriangleRulePoint> out;
    out.reserve(n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double a = gauss.points[i];
            const double b = gauss.points[j];
            const double weight = gauss.weights[i] * gauss.weights[j] * 0.5 * (1.0 - b);
            out.push_back({0.5 * (1.0 + a) * (1.0 - b) - 1.0, b, weight});
        }
    }
    return out;
}

std::vector<VolumeRulePoint> tetrahedronRule(std::size_t n)
{
    const LineRule ra = gaussLegendre(n);
    const LineRule rb = gaussJacobi(n, 1.0);
    const LineRule rc = gaussJacobi(n, 2.0);
    std::vector<VolumeRulePoint> out;
    out.reserve(n * n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                const double a = ra.points[i];
                const double b = rb.points[j];
                const double c = rc.points[k];
                const std::array<double, 3> point{0.25 * (1.0 + a) * (1.0 - b) * (1.0 - c) - 1.0,
                                                  0.5 * (1.0 + b) * (1.0 - c) - 1.0, c};
                out.push_back({point, ra.weights[i] * rb.weights[j] * rc.weights[k] / 8.0});
            }
        }
    }
    return out;
}

std::vector<VolumeRulePoint> hexahedronRule(std::size_t n)
{
    const LineRule gauss = gaussLegendre(n);
    const std::vector<double>& x = gauss.points;
    const std::vector<double>& w = gauss.weights;
    std::vector<VolumeRulePoint> out;
    out.reserve(n * n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                out.push_back({{x[i], x[j], x[k]}, w[i] * w[j] * w[k]});
    return out;
}

std::vector<VolumeRulePoint> pentahedronRule(std::size_t n)
{
    const LineRule gauss = gaussLegendre(n);
    std::vector<VolumeRulePoint> out;
    out.reserve(n * n * n);
    for (const TriangleRulePoint& p : triangleRule(n))
        for (std::size_t k = 0; k < n; ++k)
            out.push_back({{p.u, p.v, gauss.points[k]}, p.weight * gauss.weights[k]});
    return out;
}

std::vector<VolumeRulePoint> pyramidRule(std::size_t n)
{
    const LineRule gauss = gaussLegendre(n);
    const LineRule jacobi = gaussJacobi(n, 2.0);
    const std::vector<double>& x = gauss.points;
    const std::vector<double>& w = gauss.weights;
    std::vector<VolumeRulePoint> out;
    out.reserve(n * n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                const double q = 0.5 * (1.0 - jacobi.points[k]);
                out.push_back({{x[i] * q, x[j] * q, jacobi.points[k]}, w[i] * w[j] * jacobi.weights[k] / 4.0});
            }
        }
    }
    return out;
}

std::vector<SurfaceRulePoint> faceRule(std::size_t cornerCount, std::size_t n)
{
    std::vector<SurfaceRulePoint> out;
    out.reserve(n * n);
    if (cornerCount == 3)
    {
        // (u, v) of the reference triangle (area 2) to (L1, L2) (area 1/2): the weights divided by 4.
        for (const TriangleRulePoint& p : triangleRule(n))
        {
            const std::array<double, 3> l = triangleBarycentric(p.u, p.v);
            out.push_back({{l[1], l[2]}, p.weight / 4.0});
        }
        return out;
    }
    const LineRule gauss = gaussLegendre(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            out.push_back({{0.5 * (1.0 + gauss.points[i]), 0.5 * (1.0 + gauss.points[j])},
                           0.25 * gauss.weights[i] * gauss.weights[j]});
    return out;
}

std::vector<SurfaceRulePoint> referenceSurfaceRule(std::size_t cornerCount, std::size_t n)
{
    const LineRule gauss = gaussLegendre(n);
    const std::vector<double>& x = gauss.points;
    const std::vector<double>& w = gauss.weights;
    std::vector<SurfaceRulePoint> out;
    out.reserve(n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            if (cornerCount == 4)
            {
                out.push_back({{x[i], x[j]}, w[i] * w[j]});
            }
            else
            {
                const double a = x[i];
                const double b = x[j];
                out.push_back({{0.5 * (1.0 + a) * (1.0 - b) - 1.0, b}, w[i] * w[j] * 0.5 * (1.0 - b)});
            }
        }
    }
    return out;
}

std::vector<LineRulePoint> unitIntervalRule(std::size_t n)
{
    const LineRule gauss = gaussLegendre(n);
    std::vector<LineRulePoint> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        out.push_back({0.5 * (gauss.points[i] + 1.0), 0.5 * gauss.weights[i]});
    return out;
}

} // namespace oos::elements::mixedHybrid
