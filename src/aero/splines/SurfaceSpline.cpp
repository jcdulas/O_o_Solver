#include "oos/aero/splines/SurfaceSpline.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/dense/DenseLu.hpp"

#include <cmath>
#include <numbers>
#include <string>

namespace oos::aero
{

namespace
{

// 16π D with D = 1 (see the class comment).
constexpr double PLATE_FACTOR = 16.0 * std::numbers::pi;
// The polynomial terms 1, x, y.
constexpr std::size_t POLYNOMIAL_TERMS = 3;
// The points are taken on a line when the smaller principal spread of their positions (the eigenvalue of their 2 × 2
// covariance) is below this share of the larger: the plane of the spline is then undetermined (a beam spline fits).
constexpr double COLLINEAR_SPREAD_RATIO = 1.0e-20;

// Whether the points lie on a line (see COLLINEAR_SPREAD_RATIO).
bool areOnALine(std::span<const std::array<double, 2>> points)
{
    double meanX = 0.0;
    double meanY = 0.0;
    for (const std::array<double, 2>& p : points)
    {
        meanX += p[0];
        meanY += p[1];
    }
    meanX /= static_cast<double>(points.size());
    meanY /= static_cast<double>(points.size());
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;
    for (const std::array<double, 2>& p : points)
    {
        xx += (p[0] - meanX) * (p[0] - meanX);
        yy += (p[1] - meanY) * (p[1] - meanY);
        xy += (p[0] - meanX) * (p[1] - meanY);
    }
    const double half = 0.5 * (xx + yy);
    const double radius = std::sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy);
    const double larger = half + radius;
    const double smaller = half - radius;
    return !(larger > 0.0) || smaller <= COLLINEAR_SPREAD_RATIO * larger;
}

} // namespace

double SurfaceSpline::kernel(double rSquared)
{
    return rSquared > 0.0 ? rSquared * std::log(rSquared) / PLATE_FACTOR : 0.0;
}

double SurfaceSpline::kernelSlope(double dx, double dy)
{
    // ∂/∂x (r² ln r²) = 2 dx (ln r² + 1), with the limit 0 at r = 0.
    const double rSquared = dx * dx + dy * dy;
    return rSquared > 0.0 ? 2.0 * dx * (std::log(rSquared) + 1.0) / PLATE_FACTOR : 0.0;
}

SurfaceSpline::Interpolation SurfaceSpline::interpolate(std::span<const std::array<double, 2>> points, double dz,
                                                        std::span<const std::array<double, 2>> targets,
                                                        base::MemoryBudget& budget)
{
    const std::size_t n = points.size();
    if (n < POLYNOMIAL_TERMS || areOnALine(points))
        throw base::SolverError(
            base::concatenate({"a surface spline needs at least three structural points not on a line, found ",
                               std::to_string(n), n < POLYNOMIAL_TERMS ? "" : " on a line",
                               "; add points off the line to its AeroSurfaceSpline1.nodeSetId, or use a beam "
                               "spline (AeroBeamSpline2, AeroBeamSpline5)"}));
    // [0 Rᵀ; R K + DZ I] [a; P] = [0; u], R the rows (1, x_j, y_j).
    const std::size_t order = n + POLYNOMIAL_TERMS;
    linalg::DenseMatrix system(budget, order, order, "surface spline system");
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::array<double, 3> row{1.0, points[j][0], points[j][1]};
        for (std::size_t k = 0; k < POLYNOMIAL_TERMS; ++k)
        {
            system(POLYNOMIAL_TERMS + j, k) = row[k];
            system(k, POLYNOMIAL_TERMS + j) = row[k];
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const double dx = points[j][0] - points[i][0];
            const double dy = points[j][1] - points[i][1];
            system(POLYNOMIAL_TERMS + j, POLYNOMIAL_TERMS + i) = kernel(dx * dx + dy * dy);
        }
        system(POLYNOMIAL_TERMS + j, POLYNOMIAL_TERMS + j) += dz;
    }
    const linalg::DenseLu factor(std::move(system), budget);
    if (factor.isSingular())
        throw base::SolverError(
            "the structural points of a surface spline are repeated (two at the same place): its "
            "system is singular; remove the repeated points from its AeroSurfaceSpline1.nodeSetId");
    // The coefficients for a unit displacement of each point: the columns of the inverse on [0; I].
    linalg::DenseMatrix coefficients(budget, order, n, "surface spline coefficients");
    for (std::size_t i = 0; i < n; ++i)
        coefficients(POLYNOMIAL_TERMS + i, i) = 1.0;
    factor.solve(coefficients.view());
    Interpolation result{linalg::DenseMatrix(budget, targets.size(), n, "surface spline plunge"),
                         linalg::DenseMatrix(budget, targets.size(), n, "surface spline slope")};
    for (std::size_t t = 0; t < targets.size(); ++t)
    {
        const double x = targets[t][0];
        const double y = targets[t][1];
        for (std::size_t i = 0; i < n; ++i)
        {
            double plunge = coefficients(0, i) + coefficients(1, i) * x + coefficients(2, i) * y;
            double slope = coefficients(1, i);
            for (std::size_t j = 0; j < n; ++j)
            {
                const double dx = x - points[j][0];
                const double dy = y - points[j][1];
                const double force = coefficients(POLYNOMIAL_TERMS + j, i);
                plunge += force * kernel(dx * dx + dy * dy);
                slope += force * kernelSlope(dx, dy);
            }
            result.plunge(t, i) = plunge;
            result.slope(t, i) = slope;
        }
    }
    return result;
}

} // namespace oos::aero
