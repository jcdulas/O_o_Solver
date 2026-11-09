#include "oos/design/optimiser/FeasibleDirections.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/design/optimiser/LinearProgram.hpp"
#include "oos/design/optimiser/OptimiserDefaults.hpp"
#include "oos/design/optimiser/QuadraticProgram.hpp"
#include "oos/design/optimiser/ScaledProblem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace oos::design
{

namespace
{

namespace defaults = optimiser_defaults;

// Zoutendijk's direction problem with push-off, as CONMIN solves it (OptimiserDefaults.hpp): the participation
// coefficient Φ of the violated constraints, 5 multiplied by 10 each time no direction reduces the violation, at most
// 1000; the push-off factors θ_j = THETA·(1 − g_j/CT)², THETA = 1, each at most 50.
constexpr double INITIAL_PUSH_WEIGHT = defaults::INITIAL_PARTICIPATION;
constexpr double PUSH_WEIGHT_GROWTH = defaults::PARTICIPATION_GROWTH;
constexpr double MAXIMUM_PUSH_WEIGHT = defaults::MAXIMUM_PARTICIPATION;
constexpr double PUSH_OFF_THETA = defaults::PUSH_OFF_THETA;
constexpr double MAXIMUM_PUSH_OFF = defaults::MAXIMUM_PUSH_OFF_FACTOR;
// β below this means the direction problem found no move that reduces the violation.
constexpr double MINIMUM_PUSH = 1.0e-10;
// The first step of a line search changes the objective by this fraction of |f| (ADS DLOBJ1, CONMIN ABOBJ1: 0.1).
constexpr double FIRST_STEP_OBJECTIVE_FRACTION = defaults::FIRST_STEP_OBJECTIVE_FRACTION;
// The largest first step of a line search, in scaled units (S has its largest component 1): ADS DX2, 0.2, the largest
// change of a variable attempted on the first step; then the search doubles.
constexpr double DEFAULT_FIRST_STEP = defaults::FIRST_STEP_VARIABLE_CHANGE;
// The line search: doublings of the step and bisections of the first step that fails.
constexpr int EXPANSION_LIMIT = 30;
constexpr int BISECTION_LIMIT = 40;
// Newton corrections back to the boundary g = 0 of the binding constraints (and of those a trial step pushed above
// zero).
constexpr int CORRECTION_LIMIT = 5;
// A constraint within this of zero is on its boundary: the restoration stops (round-off of the normalised values).
constexpr double RESTORED = 1.0e-12;
// A direction whose largest scaled component is below this has no length.
constexpr double ZERO_DIRECTION = 1.0e-12;
// A scaled step below this changes nothing.
constexpr double ZERO_STEP = 1.0e-12;
// A variable within this scaled distance of a bound is on it (a side constraint is active).
constexpr double BOUND_TOLERANCE = 1.0e-10;

double dot(std::span<const double> a, std::span<const double> b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

double largestMagnitude(std::span<const double> v)
{
    double largest = 0.0;
    for (const double value : v)
        largest = std::max(largest, std::abs(value));
    return largest;
}

class Run
{
public:
    Run(const OptimisationProblem& problem, std::span<const double> start, const OptimiserSettings& settings,
        base::MemoryBudget& budget)
        : m_scaled(problem, settings, start, budget), m_settings(settings), m_n(m_scaled.variableCount())
    {
    }

    OptimiserResult run()
    {
        ScaledPoint point = m_scaled.evaluate(m_scaled.start());
        std::vector<OptimiserIteration> history{m_scaled.record(0, point)};
        std::size_t smallChanges = 0;
        std::size_t unconstrainedRun = 0;
        std::vector<double> previousDirection;
        double previousGradientNorm = 0.0;
        double pushWeight = INITIAL_PUSH_WEIGHT;
        for (std::size_t iteration = 1; iteration <= m_settings.iterationLimit; ++iteration)
        {
            std::vector<std::size_t> rows;
            bool hasViolated = false;
            for (std::size_t j = 0; j < point.constraints.size(); ++j)
                if (point.constraints[j] >= m_settings.constraintActiveThreshold)
                {
                    rows.push_back(j);
                    hasViolated = hasViolated || point.constraints[j] > m_settings.constraintViolationThreshold;
                }
            // Φ goes back to its initial value at a feasible design (CONMIN, subroutine CONMIN).
            if (!hasViolated)
                pushWeight = INITIAL_PUSH_WEIGHT;
            std::vector<double> objectiveGradient;
            std::span<const double> rowGradients;
            m_scaled.gradients(point, rows, objectiveGradient, rowGradients);

            std::vector<double> direction;
            std::vector<bool> binding; // per row: binding in the projection (a positive multiplier)
            if (rows.empty())
            {
                direction = steepestOrConjugate(point, objectiveGradient, unconstrainedRun, previousDirection,
                                                previousGradientNorm);
                ++unconstrainedRun;
                if (largestMagnitude(direction) <= ZERO_DIRECTION * std::max(1.0, std::abs(point.objective)))
                    return finish(point, OptimiserTermination::KuhnTucker, iteration, std::move(history));
            }
            else if (!hasViolated)
            {
                unconstrainedRun = 0;
                direction = projectedDirection(point, objectiveGradient, rowGradients, rows.size(), binding);
                if (largestMagnitude(direction) <=
                    m_settings.kuhnTuckerTolerance * std::max(largestMagnitude(objectiveGradient), ZERO_DIRECTION))
                {
                    // A Kuhn-Tucker point on the boundary that a line search left slightly outside (0 < g ≤ CTMIN):
                    // restored onto g = 0 before it is returned.
                    if (std::optional<ScaledPoint> restored = restoreOntoBoundary(point, rows, rowGradients))
                    {
                        point = std::move(*restored);
                        history.push_back(m_scaled.record(iteration, point));
                        continue;
                    }
                    return finish(point, OptimiserTermination::KuhnTucker, iteration, std::move(history));
                }
            }
            else
            {
                unconstrainedRun = 0;
                std::optional<std::vector<double>> pushed;
                while (!(pushed = pushedDirection(point, objectiveGradient, rowGradients, rows, pushWeight)))
                {
                    if (pushWeight >= MAXIMUM_PUSH_WEIGHT)
                        return finish(point, OptimiserTermination::Infeasible, iteration, std::move(history));
                    pushWeight = std::min(pushWeight * PUSH_WEIGHT_GROWTH, MAXIMUM_PUSH_WEIGHT);
                }
                direction = std::move(*pushed);
            }
            const double scale = largestMagnitude(direction);
            if (scale <= ZERO_DIRECTION)
                return finish(point, hasViolated ? OptimiserTermination::Infeasible : OptimiserTermination::KuhnTucker,
                              iteration, std::move(history));
            for (double& value : direction)
                value /= scale;
            previousDirection = direction;
            previousGradientNorm = dot(objectiveGradient, objectiveGradient);

            const std::optional<ScaledPoint> next =
                search(point, direction, objectiveGradient, rows, rowGradients, binding);
            if (!next)
            {
                // No step improves a point slightly outside (0 < g ≤ CTMIN): it is restored onto g = 0 and the
                // optimisation goes on from there, rather than ending at a point that the next design cycle would
                // start from again.
                if (!hasViolated)
                    if (std::optional<ScaledPoint> restored = restoreOntoBoundary(point, rows, rowGradients))
                    {
                        point = std::move(*restored);
                        smallChanges = 0;
                        history.push_back(m_scaled.record(iteration, point));
                        continue;
                    }
                return finish(point, hasViolated ? OptimiserTermination::Infeasible : OptimiserTermination::NoProgress,
                              iteration, std::move(history));
            }
            const bool isFeasible =
                next->constraints.empty() || next->maxConstraint <= m_settings.constraintViolationThreshold;
            smallChanges =
                isFeasible && m_scaled.isSmallObjectiveChange(point.objective, next->objective) ? smallChanges + 1 : 0;
            point = *next;
            history.push_back(m_scaled.record(iteration, point));
            if (smallChanges >= m_settings.convergedIterations)
                return finish(point, OptimiserTermination::ObjectiveConverged, iteration, std::move(history));
        }
        return finish(point, OptimiserTermination::IterationLimit, m_settings.iterationLimit, std::move(history));
    }

private:
    [[nodiscard]] OptimiserResult finish(const ScaledPoint& point, OptimiserTermination termination,
                                         std::size_t iterations, std::vector<OptimiserIteration> history) const
    {
        return m_scaled.finish(point, OptimiserMethod::FeasibleDirections, termination, iterations, std::move(history));
    }

    [[nodiscard]] bool isAtLower(const ScaledPoint& point, std::size_t i) const noexcept
    {
        return point.y[i] <= m_scaled.lower()[i] + BOUND_TOLERANCE;
    }
    [[nodiscard]] bool isAtUpper(const ScaledPoint& point, std::size_t i) const noexcept
    {
        return point.y[i] >= m_scaled.upper()[i] - BOUND_TOLERANCE;
    }

    // Steepest descent, or Fletcher-Reeves after an unconstrained iteration (restarted every n), without the
    // components that leave the bounds.
    [[nodiscard]] std::vector<double> steepestOrConjugate(const ScaledPoint& point, const std::vector<double>& gradient,
                                                          std::size_t unconstrainedRun,
                                                          const std::vector<double>& previous,
                                                          double previousNorm) const
    {
        std::vector<double> direction(m_n);
        for (std::size_t i = 0; i < m_n; ++i)
            direction[i] = -gradient[i];
        if (unconstrainedRun > 0 && unconstrainedRun % m_n != 0 && previousNorm > 0.0 && previous.size() == m_n)
        {
            const double beta = dot(gradient, gradient) / previousNorm;
            for (std::size_t i = 0; i < m_n; ++i)
                direction[i] += beta * previous[i];
            if (dot(direction, gradient) >= 0.0)
                for (std::size_t i = 0; i < m_n; ++i)
                    direction[i] = -gradient[i];
        }
        for (std::size_t i = 0; i < m_n; ++i)
            if ((isAtLower(point, i) && direction[i] < 0.0) || (isAtUpper(point, i) && direction[i] > 0.0))
                direction[i] = 0.0;
        return direction;
    }

    // S = argmin ½|S|² + ∇f·S subject to ∇g_j·S ≤ 0 (active) and the side constraints.
    std::vector<double> projectedDirection(const ScaledPoint& point, const std::vector<double>& gradient,
                                           std::span<const double> rowGradients, std::size_t rowCount,
                                           std::vector<bool>& binding) const
    {
        base::MemoryBudget& budget = m_scaled.budget();
        base::BudgetedBuffer<double> identity(budget, m_n * m_n, "feasible directions metric");
        for (std::size_t i = 0; i < m_n; ++i)
            identity[i * m_n + i] = 1.0;
        std::vector<std::size_t> sides; // the variables on a bound
        for (std::size_t i = 0; i < m_n; ++i)
            if (isAtLower(point, i) || isAtUpper(point, i))
                sides.push_back(i);
        const std::size_t count = rowCount + sides.size();
        base::BudgetedBuffer<double> normals(budget, count * m_n, "feasible directions normals");
        const std::vector<double> rhs(count, 0.0);
        for (std::size_t k = 0; k < rowCount; ++k)
            for (std::size_t i = 0; i < m_n; ++i)
                normals[k * m_n + i] = -rowGradients[k * m_n + i];
        for (std::size_t s = 0; s < sides.size(); ++s)
            normals[(rowCount + s) * m_n + sides[s]] = isAtLower(point, sides[s]) ? 1.0 : -1.0;
        const QuadraticProgramSolution solution =
            QuadraticProgram::solve(identity.span(), gradient, normals.span(), rhs, budget);
        binding.assign(rowCount, false);
        if (solution.status != QuadraticProgramStatus::Optimal)
            return std::vector<double>(m_n, 0.0);
        for (const std::size_t row : solution.active)
            if (row < rowCount && solution.multipliers[row] > 0.0)
                binding[row] = true;
        return solution.x;
    }

    // Zoutendijk's problem with push-off: minimise ∇f·S − Φ·β over (S, β); nothing when β = 0 (no direction reduces
    // the violation at this weight).
    [[nodiscard]] std::optional<std::vector<double>> pushedDirection(const ScaledPoint& point,
                                                                     const std::vector<double>& gradient,
                                                                     std::span<const double> rowGradients,
                                                                     const std::vector<std::size_t>& rows,
                                                                     double weight) const
    {
        const std::size_t columns = m_n + 1;
        const double gradientNorm = std::sqrt(dot(gradient, gradient));
        std::vector<double> cost(columns, 0.0);
        for (std::size_t i = 0; i < m_n; ++i)
            cost[i] = gradientNorm > 0.0 ? gradient[i] / gradientNorm : 0.0;
        cost[m_n] = -weight;
        base::BudgetedBuffer<double> a(m_scaled.budget(), rows.size() * columns, "feasible directions push-off rows");
        std::vector<double> b(rows.size(), 0.0);
        // θ_j = THETA·(1 − g_j/CT)², from 0 at g_j = CT to 4·THETA at g_j = |CT|, at most 50, then scaled so that the
        // largest is 1 (CONMIN, subroutine CNMN05).
        std::vector<double> pushOff(rows.size(), 0.0);
        double largestPushOff = 0.0;
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            const double ratio = 1.0 - point.constraints[rows[k]] / m_settings.constraintActiveThreshold;
            pushOff[k] = std::min(PUSH_OFF_THETA * ratio * ratio, MAXIMUM_PUSH_OFF);
            largestPushOff = std::max(largestPushOff, pushOff[k]);
        }
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            const std::span<const double> row{rowGradients.data() + k * m_n, m_n};
            const double norm = std::sqrt(dot(row, row));
            for (std::size_t i = 0; i < m_n; ++i)
                a[k * columns + i] = norm > 0.0 ? row[i] / norm : 0.0;
            a[k * columns + m_n] = largestPushOff > 0.0 ? pushOff[k] / largestPushOff : 1.0;
        }
        std::vector<double> lower(columns, -1.0);
        std::vector<double> upper(columns, 1.0);
        for (std::size_t i = 0; i < m_n; ++i)
        {
            if (isAtLower(point, i))
                lower[i] = 0.0;
            if (isAtUpper(point, i))
                upper[i] = 0.0;
        }
        lower[m_n] = 0.0;
        upper[m_n] = std::sqrt(static_cast<double>(m_n));
        const LinearProgramSolution solution = LinearProgram::solve(cost, a.span(), b, lower, upper, m_scaled.budget());
        if (solution.status != LinearProgramStatus::Optimal || solution.x[m_n] <= MINIMUM_PUSH)
            return std::nullopt;
        return std::vector<double>(solution.x.begin(), solution.x.begin() + static_cast<std::ptrdiff_t>(m_n));
    }

    // The largest step along S within the bounds.
    [[nodiscard]] double maximumStep(const ScaledPoint& point, const std::vector<double>& direction) const
    {
        double step = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < m_n; ++i)
        {
            if (direction[i] > 0.0)
                step = std::min(step, (m_scaled.upper()[i] - point.y[i]) / direction[i]);
            else if (direction[i] < 0.0)
                step = std::min(step, (m_scaled.lower()[i] - point.y[i]) / direction[i]);
        }
        return std::max(step, 0.0);
    }

    // The point slightly outside (0 < g ≤ CTMIN) brought back onto g = 0 of the constraints above zero; nothing when
    // it is not outside or the restoration does not reduce its largest constraint.
    std::optional<ScaledPoint> restoreOntoBoundary(const ScaledPoint& point, const std::vector<std::size_t>& rows,
                                                   std::span<const double> rowGradients)
    {
        if (point.maxConstraint <= RESTORED)
            return std::nullopt;
        std::vector<bool> over(rows.size());
        for (std::size_t k = 0; k < rows.size(); ++k)
            over[k] = point.constraints[rows[k]] > 0.0;
        ScaledPoint restored =
            trial(point, std::vector<double>(m_n, 0.0), 0.0, rows, rowGradients, over, Restoration::OntoBoundary);
        if (!(restored.maxConstraint < point.maxConstraint))
            return std::nullopt;
        return restored;
    }

    // The trial point at step α, brought back to the boundary of the binding constraints (none: no correction). Along
    // a search, a constraint the start already had above zero (0 < g ≤ CTMIN, tolerated) is held at its start value:
    // the step is not charged with removing a violation the start had, which would make every step from such a point
    // look worse than the start (restoreOntoBoundary removes it); `OntoBoundary` brings every one onto g = 0.
    enum class Restoration : std::uint8_t
    {
        HoldingStartLevel,
        OntoBoundary
    };
    ScaledPoint trial(const ScaledPoint& point, const std::vector<double>& direction, double step,
                      const std::vector<std::size_t>& rows, std::span<const double> rowGradients,
                      const std::vector<bool>& binding, Restoration restoration = Restoration::HoldingStartLevel)
    {
        std::vector<double> target(rows.size(), 0.0); // the value each corrected constraint is brought to
        if (restoration == Restoration::HoldingStartLevel)
            for (std::size_t k = 0; k < rows.size(); ++k)
                target[k] = std::max(point.constraints[rows[k]], 0.0);
        std::vector<double> y = point.y;
        for (std::size_t i = 0; i < m_n; ++i)
            y[i] += step * direction[i];
        m_scaled.clip(y);
        ScaledPoint candidate = m_scaled.evaluate(std::move(y));
        if (binding.empty())
            return candidate;
        for (int correction = 0; correction < CORRECTION_LIMIT; ++correction)
        {
            std::vector<std::size_t> over; // positions in rows
            double largestExcess = 0.0;
            for (std::size_t k = 0; k < rows.size(); ++k)
                if (binding[k] || candidate.constraints[rows[k]] > target[k])
                {
                    over.push_back(k);
                    largestExcess = std::max(largestExcess, std::abs(candidate.constraints[rows[k]] - target[k]));
                }
            if (over.empty() || largestExcess <= RESTORED)
                break;
            over = independentRows(over, rowGradients);
            // Δy = −Aᵀ·(A·Aᵀ)⁻¹·(g − target), the least-norm step onto g = target (Rosen's restoration, the gradients
            // of the start).
            const std::size_t q = over.size();
            std::vector<double> gram(q * q, 0.0); // q: the active constraints above zero, a handful
            std::vector<double> excess(q, 0.0);
            for (std::size_t r = 0; r < q; ++r)
            {
                excess[r] = candidate.constraints[rows[over[r]]] - target[over[r]];
                for (std::size_t s = 0; s < q; ++s)
                    gram[r * q + s] =
                        dot({rowGradients.data() + over[r] * m_n, m_n}, {rowGradients.data() + over[s] * m_n, m_n});
            }
            const std::optional<std::vector<double>> weights = solveSymmetric(gram, excess, q);
            if (!weights)
                break;
            std::vector<double> corrected = candidate.y;
            for (std::size_t r = 0; r < q; ++r)
                for (std::size_t i = 0; i < m_n; ++i)
                    corrected[i] -= (*weights)[r] * rowGradients[over[r] * m_n + i];
            m_scaled.clip(corrected);
            candidate = m_scaled.evaluate(std::move(corrected));
        }
        return candidate;
    }

    // The rows of `positions` whose gradients are independent of the earlier ones (modified Gram-Schmidt): constraints
    // with the same gradient (the stress of a symmetric structure in two mirrored subcases) are restored by the first
    // one alone.
    [[nodiscard]] std::vector<std::size_t> independentRows(const std::vector<std::size_t>& positions,
                                                           std::span<const double> rowGradients) const
    {
        // A gradient keeping less than this fraction of its squared norm outside the span of the others depends on
        // them (round-off of equal rows is far below it).
        constexpr double DEPENDENCE = 1.0e-10;
        std::vector<std::size_t> kept;
        std::vector<std::vector<double>> basis; // orthonormal, one per kept row (a handful of n-vectors)
        for (const std::size_t position : positions)
        {
            std::vector<double> residual(rowGradients.begin() + static_cast<std::ptrdiff_t>(position * m_n),
                                         rowGradients.begin() + static_cast<std::ptrdiff_t>((position + 1) * m_n));
            const double norm = dot(residual, residual);
            for (const std::vector<double>& direction : basis)
            {
                const double projection = dot(residual, direction);
                for (std::size_t i = 0; i < m_n; ++i)
                    residual[i] -= projection * direction[i];
            }
            const double remaining = dot(residual, residual);
            if (!(remaining > DEPENDENCE * norm))
                continue;
            const double length = std::sqrt(remaining);
            for (double& value : residual)
                value /= length;
            basis.push_back(std::move(residual));
            kept.push_back(position);
        }
        return kept;
    }

    // Solves the small symmetric system of the correction by Gaussian elimination with partial pivoting; nothing when
    // it is singular (left only by round-off once the dependent rows are dropped: the correction is skipped).
    static std::optional<std::vector<double>> solveSymmetric(std::vector<double> matrix, std::vector<double> rhs,
                                                             std::size_t q)
    {
        constexpr double SINGULAR_PIVOT = 1.0e-14;
        double largest = 0.0;
        for (const double value : matrix)
            largest = std::max(largest, std::abs(value));
        for (std::size_t column = 0; column < q; ++column)
        {
            std::size_t pivot = column;
            for (std::size_t r = column + 1; r < q; ++r)
                if (std::abs(matrix[r * q + column]) > std::abs(matrix[pivot * q + column]))
                    pivot = r;
            if (std::abs(matrix[pivot * q + column]) <= SINGULAR_PIVOT * largest)
                return std::nullopt;
            if (pivot != column)
            {
                for (std::size_t j = 0; j < q; ++j)
                    std::swap(matrix[pivot * q + j], matrix[column * q + j]);
                std::swap(rhs[pivot], rhs[column]);
            }
            for (std::size_t r = column + 1; r < q; ++r)
            {
                const double factor = matrix[r * q + column] / matrix[column * q + column];
                for (std::size_t j = column; j < q; ++j)
                    matrix[r * q + j] -= factor * matrix[column * q + j];
                rhs[r] -= factor * rhs[column];
            }
        }
        std::vector<double> solution(q, 0.0);
        for (std::size_t r = q; r-- > 0;)
        {
            double value = rhs[r];
            for (std::size_t j = r + 1; j < q; ++j)
                value -= matrix[r * q + j] * solution[j];
            solution[r] = value / matrix[r * q + r];
        }
        return solution;
    }

    // Whether `candidate` improves on `reference`: from a feasible start, feasible (every g ≤ CTMIN) with a lower
    // objective; from an infeasible one, a smaller largest violation, or the same with a lower objective.
    [[nodiscard]] bool improves(const ScaledPoint& start, const ScaledPoint& candidate,
                                const ScaledPoint& reference) const noexcept
    {
        const double violation = m_settings.constraintViolationThreshold;
        const bool startsFeasible = start.constraints.empty() || start.maxConstraint <= violation;
        if (startsFeasible)
            return (candidate.constraints.empty() || candidate.maxConstraint <= violation) &&
                   candidate.objective < reference.objective;
        const double before = std::max(reference.maxConstraint, 0.0);
        const double after = std::max(candidate.maxConstraint, 0.0);
        if (after <= m_settings.constraintViolationThreshold && before > m_settings.constraintViolationThreshold)
            return true;
        return after < before || (after <= before && candidate.objective < reference.objective);
    }

    std::optional<ScaledPoint> search(const ScaledPoint& point, const std::vector<double>& direction,
                                      const std::vector<double>& gradient, const std::vector<std::size_t>& rows,
                                      std::span<const double> rowGradients, const std::vector<bool>& binding)
    {
        const double maximum = maximumStep(point, direction);
        if (maximum <= ZERO_STEP)
            return std::nullopt;
        const double slope = dot(gradient, direction);
        double step = DEFAULT_FIRST_STEP;
        if (slope < 0.0 && point.objective != 0.0)
            step = std::min(step, FIRST_STEP_OBJECTIVE_FRACTION * std::abs(point.objective) / -slope);
        step = std::min(step, maximum);

        std::optional<ScaledPoint> best;
        double bestStep = 0.0;
        double failedStep = 0.0;
        std::optional<double> previousObjective;
        double previousStep = 0.0;
        // Expansion: double while the point improves.
        for (int expansion = 0; expansion < EXPANSION_LIMIT; ++expansion)
        {
            ScaledPoint candidate = trial(point, direction, step, rows, rowGradients, binding);
            if (improves(point, candidate, best ? *best : point))
            {
                previousObjective = best ? std::optional<double>(best->objective) : std::nullopt;
                previousStep = bestStep;
                best = std::move(candidate);
                bestStep = step;
                if (step >= maximum)
                    return best;
                step = std::min(2.0 * step, maximum);
                continue;
            }
            failedStep = step;
            // The objective passed its minimum along S at a feasible point: a quadratic through the last three
            // objectives (a polynomial interpolation).
            const bool isFeasible =
                candidate.constraints.empty() || candidate.maxConstraint <= m_settings.constraintViolationThreshold;
            if (best && isFeasible && candidate.objective >= best->objective)
            {
                const double s0 = previousObjective ? previousStep : 0.0;
                const double f0 = previousObjective ? *previousObjective : point.objective;
                const std::optional<double> fitted =
                    parabolaMinimum(s0, f0, bestStep, best->objective, step, candidate.objective);
                if (fitted && *fitted > s0 && *fitted < step && std::abs(*fitted - bestStep) > ZERO_STEP)
                {
                    ScaledPoint refined = trial(point, direction, *fitted, rows, rowGradients, binding);
                    if (improves(point, refined, *best))
                        best = std::move(refined);
                }
                return best;
            }
            break;
        }
        if (failedStep <= 0.0)
            return best;
        // Bisection between the best step (or 0) and the first that failed.
        double low = bestStep;
        double high = failedStep;
        for (int bisection = 0; bisection < BISECTION_LIMIT && high - low > ZERO_STEP; ++bisection)
        {
            const double middle = 0.5 * (low + high);
            ScaledPoint candidate = trial(point, direction, middle, rows, rowGradients, binding);
            if (improves(point, candidate, best ? *best : point))
            {
                best = std::move(candidate);
                low = middle;
            }
            else
                high = middle;
        }
        return best;
    }

    // The minimum of the parabola through three points, nothing when it opens downwards or is degenerate.
    static std::optional<double> parabolaMinimum(double x0, double f0, double x1, double f1, double x2, double f2)
    {
        const double d01 = (f1 - f0) / (x1 - x0);
        const double d12 = (f2 - f1) / (x2 - x1);
        const double curvature = (d12 - d01) / (x2 - x0);
        if (!(curvature > 0.0) || !std::isfinite(curvature))
            return std::nullopt;
        // f(x) = f0 + d01·(x − x0) + curvature·(x − x0)(x − x1): f' = 0 at x = (x0 + x1)/2 − d01/(2·curvature).
        return 0.5 * (x0 + x1) - d01 / (2.0 * curvature);
    }

    ScaledProblem m_scaled;
    const OptimiserSettings& m_settings;
    std::size_t m_n;
};

} // namespace

OptimiserResult FeasibleDirections::run(const OptimisationProblem& problem, std::span<const double> start,
                                        const OptimiserSettings& settings, base::MemoryBudget& budget)
{
    Run run(problem, start, settings, budget);
    return run.run();
}

} // namespace oos::design
