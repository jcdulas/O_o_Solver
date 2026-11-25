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
                                                  base::MemoryBudget& budget, Geometry* keep)
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
    linalg::DenseLu bendingFactor(std::move(bending), budget);
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
    if (keep)
    {
        keep->stations.assign(stations.begin(), stations.end());
        keep->targets.assign(targets.begin(), targets.end());
        keep->flexibility = flexibility;
        keep->bending = std::make_unique<linalg::DenseLu>(std::move(bendingFactor));
        keep->bendCoefficients = std::move(bendingCoefficients);
    }
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
    linalg::DenseLu torsionFactor(std::move(torsion), budget);
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
    if (keep)
    {
        keep->torsion = std::make_unique<linalg::DenseLu>(std::move(torsionFactor));
        keep->twistCoefficients = std::move(torsionCoefficients);
    }
    return result;
}

BeamSpline::Interpolation BeamSpline::Geometry::derivative(std::span<const double> velocity,
                                                           base::MemoryBudget& budget) const
{
    const std::size_t n = stations.size(), columns = bendCoefficients.columns();
    const bool hasSlopes = columns > n;
    // Directional derivative of |d|. A moving kink does not have a two-sided tangent.
    const auto absRate = [](double d, double v)
    {
        if (d == 0 && v != 0)
            throw base::SolverError("undefined beam spline derivative at coincident moving stations");
        return d > 0 ? v : d < 0 ? -v : 0.0;
    };
    linalg::DenseMatrix db(budget, bending->order(), columns, "beam spline bending coefficient rates");
    linalg::DenseMatrix dt(budget, torsion ? n + 1 : 0, torsion ? n : 0, "beam spline torsion coefficient rates");
    // Solve A dC = -dA C for bending and torsion with their original factors.
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t c = 0; c < columns; ++c)
        {
            db(1, c) -= velocity[j] * bendCoefficients(2 + j, c);
            db(2 + j, c) -= velocity[j] * bendCoefficients(1, c);
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const double d = stations[j] - stations[i], v = velocity[j] - velocity[i];
            const double da = (hasSlopes || torsion) ? absRate(d, v) : 0.;
            for (std::size_t c = 0; c < columns; ++c)
            {
                db(2 + j, c) -= v * forceSlopeKernel(d) * bendCoefficients(2 + i, c);
                if (hasSlopes)
                {
                    db(2 + j, c) -= v * coupleSlopeKernel(d) * bendCoefficients(2 + n + i, c);
                    db(2 + n + j, c) += v * coupleSlopeKernel(d) * bendCoefficients(2 + i, c) +
                                        0.5 * da * bendCoefficients(2 + n + i, c);
                }
            }
            if (torsion)
                for (std::size_t c = 0; c < n; ++c)
                    dt(1 + j, c) += 0.5 * flexibility.dtor * da * twistCoefficients(1 + i, c);
        }
    }
    bending->solve(db.view());
    if (torsion)
        torsion->solve(dt.view());
    Interpolation out{linalg::DenseMatrix(budget, targets.size(), 3 * n, "beam plunge rates"),
                      linalg::DenseMatrix(budget, targets.size(), 3 * n, "beam chord slope rates"),
                      linalg::DenseMatrix(budget, targets.size(), 3 * n, "beam span slope rates")};
    for (std::size_t t = 0; t < targets.size(); ++t)
    {
        const auto [x, y] = targets[t];
        for (std::size_t c = 0; c < columns; ++c)
        {
            const std::size_t col = c < n ? 3 * c : 3 * (c - n) + 1;
            double plunge = db(0, c) + y * db(1, c), slope = db(1, c);
            for (std::size_t i = 0; i < n; ++i)
            {
                const double d = y - stations[i], v = -velocity[i];
                plunge += db(2 + i, c) * forceKernel(d) + bendCoefficients(2 + i, c) * forceSlopeKernel(d) * v;
                slope += db(2 + i, c) * forceSlopeKernel(d) - bendCoefficients(2 + i, c) * coupleSlopeKernel(d) * v;
                if (hasSlopes)
                {
                    plunge +=
                        db(2 + n + i, c) * coupleKernel(d) + bendCoefficients(2 + n + i, c) * coupleSlopeKernel(d) * v;
                    slope +=
                        db(2 + n + i, c) * coupleSlopeKernel(d) - 0.5 * bendCoefficients(2 + n + i, c) * absRate(d, v);
                }
            }
            out.plunge(t, col) = plunge;
            out.spanSlope(t, col) = slope;
        }
        if (torsion)
            for (std::size_t c = 0; c < n; ++c)
            {
                double theta = dt(0, c), span = 0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    const double d = y - stations[i];
                    theta += dt(1 + i, c) * torsionKernel(d, flexibility.dtor) -
                             0.5 * flexibility.dtor * twistCoefficients(1 + i, c) * absRate(d, -velocity[i]);
                    span -= 0.5 * flexibility.dtor * dt(1 + i, c) * (d > 0 ? 1 : d < 0 ? -1 : 0);
                }
                out.plunge(t, 3 * c + 2) = -x * theta;
                out.slope(t, 3 * c + 2) = -theta;
                out.spanSlope(t, 3 * c + 2) = -x * span;
            }
    }
    return out;
}

} // namespace oos::aero
