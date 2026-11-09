#include "oos/aero/splines/BeamSpline.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/dense/DenseLu.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace oos::aero
{

namespace
{

// The unknowns of a station: deflection, rotation about x, rotation about y (the columns of the interpolation).
constexpr std::size_t STATION_UNKNOWNS = 3;

double forceKernel(double d)
{
    return std::abs(d) * d * d / 12.0;
}

// The deflection at y of a unit couple at y_i (d = y − y_i), and the slope at y of a unit force at y_i.
double coupleKernel(double d)
{
    return -d * std::abs(d) / 4.0;
}

double forceSlopeKernel(double d)
{
    return d * std::abs(d) / 4.0;
}

double coupleSlopeKernel(double d)
{
    return -std::abs(d) / 2.0;
}

// The twist at y of a unit torque at y_i (d = y − y_i), GJ = 1 / DTOR: the Green function of θ'' = −T δ / GJ of a
// torsion bar, negative, so that an attachment spring adds its flexibility to the diagonal in series.
double torsionKernel(double d, double torsionRatio)
{
    return -torsionRatio * std::abs(d) / 2.0;
}

} // namespace

BeamSpline::Interpolation BeamSpline::interpolate(std::span<const double> stations, const Flexibility& flexibility,
                                                  std::span<const std::array<double, 2>> targets,
                                                  base::MemoryBudget& budget)
{
    const std::size_t n = stations.size();
    std::vector<double> distinct(stations.begin(), stations.end());
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    if (distinct.size() < 2)
        throw base::SolverError(
            base::concatenate({"a beam spline needs structural points at two stations along its axis at least, found ",
                               std::to_string(distinct.size()), "; add points to its AeroBeamSpline2.nodeSetId"}));
    const bool hasSlopes = flexibility.dthx >= 0.0;
    const bool hasTwists = flexibility.dthy >= 0.0;
    Interpolation result{linalg::DenseMatrix(budget, targets.size(), STATION_UNKNOWNS * n, "beam spline plunge"),
                         linalg::DenseMatrix(budget, targets.size(), STATION_UNKNOWNS * n, "beam spline slope"),
                         linalg::DenseMatrix(budget, targets.size(), STATION_UNKNOWNS * n, "beam spline span slope")};

    // Bending: [a0, a1, F, M] with the equilibrium rows first; the right-hand sides are the deflections, then the
    // slopes.
    const std::size_t couples = hasSlopes ? n : 0;
    const std::size_t order = 2 + n + couples;
    linalg::DenseMatrix bending(budget, order, order, "beam spline bending system");
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::size_t row = 2 + j;
        bending(row, 0) = bending(0, row) = 1.0;
        bending(row, 1) = bending(1, row) = stations[j];
        for (std::size_t i = 0; i < n; ++i)
            bending(row, 2 + i) = forceKernel(stations[j] - stations[i]);
        bending(row, row) += flexibility.dz;
        if (!hasSlopes)
            continue;
        const std::size_t slopeRow = 2 + n + j;
        bending(slopeRow, 1) = bending(1, slopeRow) = 1.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            bending(row, 2 + n + i) = coupleKernel(stations[j] - stations[i]);
            bending(slopeRow, 2 + i) = forceSlopeKernel(stations[j] - stations[i]);
            bending(slopeRow, 2 + n + i) = coupleSlopeKernel(stations[j] - stations[i]);
        }
        bending(slopeRow, slopeRow) += flexibility.dthx;
    }
    const linalg::DenseLu bendingFactor(std::move(bending), budget);
    if (bendingFactor.isSingular())
        throw base::SolverError(
            "the bending system of a beam spline is singular (structural points at the same "
            "station with no attachment flexibility); give AeroBeamSpline2.linearFlexibility > 0 (and "
            "AeroBeamSpline2.rotationFlexibilityX > 0 with attached "
            "slopes) or remove the repeated points");
    const std::size_t attached = n + couples;
    linalg::DenseMatrix bendingCoefficients(budget, order, attached, "beam spline bending coefficients");
    for (std::size_t k = 0; k < attached; ++k)
        bendingCoefficients(2 + k, k) = 1.0;
    bendingFactor.solve(bendingCoefficients.view());
    for (std::size_t t = 0; t < targets.size(); ++t)
    {
        const double y = targets[t][1];
        for (std::size_t k = 0; k < attached; ++k)
        {
            double w = bendingCoefficients(0, k) + bendingCoefficients(1, k) * y;
            double slope = bendingCoefficients(1, k);
            for (std::size_t i = 0; i < n; ++i)
            {
                w += bendingCoefficients(2 + i, k) * forceKernel(y - stations[i]);
                slope += bendingCoefficients(2 + i, k) * forceSlopeKernel(y - stations[i]);
                if (hasSlopes)
                {
                    w += bendingCoefficients(2 + n + i, k) * coupleKernel(y - stations[i]);
                    slope += bendingCoefficients(2 + n + i, k) * coupleSlopeKernel(y - stations[i]);
                }
            }
            // The column of the deflection (k < n) or of the rotation about x (k ≥ n) of its station.
            const std::size_t column = k < n ? STATION_UNKNOWNS * k : STATION_UNKNOWNS * (k - n) + 1;
            result.plunge(t, column) = w;
            result.spanSlope(t, column) = slope;
        }
    }

    // Torsion: [b0, T], the twists attached.
    if (!hasTwists)
        return result;
    linalg::DenseMatrix torsion(budget, 1 + n, 1 + n, "beam spline torsion system");
    for (std::size_t j = 0; j < n; ++j)
    {
        torsion(1 + j, 0) = torsion(0, 1 + j) = 1.0;
        for (std::size_t i = 0; i < n; ++i)
            torsion(1 + j, 1 + i) = torsionKernel(stations[j] - stations[i], flexibility.dtor);
        torsion(1 + j, 1 + j) += flexibility.dthy;
    }
    const linalg::DenseLu torsionFactor(std::move(torsion), budget);
    if (torsionFactor.isSingular())
        throw base::SolverError(
            "the torsion system of a beam spline is singular (repeated stations with attached "
            "twists and no flexibility); give AeroBeamSpline2.rotationFlexibilityY > 0 or remove the repeated "
            "points");
    linalg::DenseMatrix torsionCoefficients(budget, 1 + n, n, "beam spline torsion coefficients");
    for (std::size_t k = 0; k < n; ++k)
        torsionCoefficients(1 + k, k) = 1.0;
    torsionFactor.solve(torsionCoefficients.view());
    for (std::size_t t = 0; t < targets.size(); ++t)
    {
        const double x = targets[t][0];
        const double y = targets[t][1];
        for (std::size_t k = 0; k < n; ++k)
        {
            double theta = torsionCoefficients(0, k);
            double twistRate = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                const double d = y - stations[i];
                theta += torsionCoefficients(1 + i, k) * torsionKernel(d, flexibility.dtor);
                twistRate -= torsionCoefficients(1 + i, k) * flexibility.dtor * (d > 0.0 ? 0.5 : d < 0.0 ? -0.5 : 0.0);
            }
            result.plunge(t, STATION_UNKNOWNS * k + 2) = -x * theta;
            result.slope(t, STATION_UNKNOWNS * k + 2) = -theta;
            result.spanSlope(t, STATION_UNKNOWNS * k + 2) = -x * twistRate;
        }
    }
    return result;
}

} // namespace oos::aero
