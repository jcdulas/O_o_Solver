#include "oos/design/optimiser/ScaledProblem.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace oos::design
{

ScaledProblem::ScaledProblem(const OptimisationProblem& problem, const OptimiserSettings& settings,
                             std::span<const double> start, base::MemoryBudget& budget)
    : m_problem(problem), m_settings(settings), m_budget(budget), m_constraintCount(problem.constraintCount())
{
    const std::size_t n = problem.variableCount();
    const std::span<const double> lower = problem.lowerBounds();
    const std::span<const double> upper = problem.upperBounds();
    if (start.size() != n || lower.size() != n || upper.size() != n)
        throw base::OptimisationError(
            base::concatenate({"optimiser: expected a start point and bounds of ", std::to_string(n),
                               " values (the variables of the "
                               "problem), found ",
                               std::to_string(start.size()), " start values, ", std::to_string(lower.size()),
                               " lower and ", std::to_string(upper.size()), " upper bounds"}));
    if (n == 0)
        throw base::OptimisationError(
            "optimiser: expected at least one variable, found none; declare the design variables (DesignVar, "
            "DesignProblem.add_variable)");
    m_scale.resize(n);
    m_lower.resize(n);
    m_upper.resize(n);
    m_start.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i] > upper[i])
            throw base::OptimisationError(
                base::concatenate({"optimiser: expected finite bounds lower ≤ upper for variable ", std::to_string(i),
                                   ", found [", std::to_string(lower[i]), ", ", std::to_string(upper[i]),
                                   "]; give the approximate problem its move limits or finite bounds"}));
        const double x = std::clamp(start[i], lower[i], upper[i]);
        double scale = 1.0;
        if (m_settings.isScaled)
        {
            // A variable at zero takes the half width of its range, so that its scaled range is [-1, 1] or so.
            const double halfRange = 0.5 * (upper[i] - lower[i]);
            scale = std::abs(x) > 0.0 ? std::abs(x) : (halfRange > 0.0 ? halfRange : 1.0);
        }
        m_scale[i] = scale;
        m_lower[i] = lower[i] / scale;
        m_upper[i] = upper[i] / scale;
        m_start[i] = x / scale;
    }
}

std::vector<double> ScaledProblem::toX(std::span<const double> y) const
{
    std::vector<double> x(y.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        x[i] = y[i] * m_scale[i];
    return x;
}

ScaledPoint ScaledProblem::evaluate(std::vector<double> y)
{
    ScaledPoint point;
    point.constraints.assign(m_constraintCount, 0.0);
    const std::vector<double> x = toX(y);
    point.objective = m_problem.evaluate(x, point.constraints);
    point.y = std::move(y);
    point.maxConstraint = -std::numeric_limits<double>::infinity();
    for (const double g : point.constraints)
        point.maxConstraint = std::max(point.maxConstraint, g);
    if (!std::isfinite(point.objective))
        throw base::OptimisationError(base::concatenate(
            {"optimiser: the objective is not finite at a trial point (", std::to_string(point.objective),
             "); check the approximate problem (a reciprocal approximation of a variable that reaches zero)"}));
    if (m_functionEvaluations == 0)
        m_absoluteChange = m_settings.absoluteObjectiveChange(point.objective);
    ++m_functionEvaluations;
    return point;
}

void ScaledProblem::gradients(const ScaledPoint& point, std::span<const std::size_t> rows,
                              std::vector<double>& objectiveGradient, std::span<const double>& constraintGradients)
{
    const std::size_t n = variableCount();
    const std::size_t size = rows.size() * n;
    if (m_rowGradients.size() < size)
        m_rowGradients = base::BudgetedBuffer<double>(m_budget, size, "optimiser constraint gradients");
    objectiveGradient.assign(n, 0.0);
    const std::vector<double> x = toX(point.y);
    const std::span<double> storage = m_rowGradients.span().first(size);
    m_problem.gradients(x, objectiveGradient, rows, storage);
    // d/dy_i = s_i · d/dx_i.
    for (std::size_t i = 0; i < n; ++i)
        objectiveGradient[i] *= m_scale[i];
    for (std::size_t k = 0; k < rows.size(); ++k)
        for (std::size_t i = 0; i < n; ++i)
            storage[k * n + i] *= m_scale[i];
    constraintGradients = storage;
    ++m_gradientEvaluations;
}

void ScaledProblem::clip(std::vector<double>& y) const noexcept
{
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = std::clamp(y[i], m_lower[i], m_upper[i]);
}

OptimiserIteration ScaledProblem::record(std::size_t iteration, const ScaledPoint& point) const noexcept
{
    OptimiserIteration entry;
    entry.iteration = iteration;
    entry.objective = point.objective;
    entry.maxConstraint = point.maxConstraint;
    for (const double g : point.constraints)
    {
        if (g > m_settings.constraintViolationThreshold)
            ++entry.violatedConstraints;
        else if (g >= m_settings.constraintActiveThreshold)
            ++entry.activeConstraints;
    }
    return entry;
}

bool ScaledProblem::isSmallObjectiveChange(double before, double after) const noexcept
{
    const double change = std::abs(after - before);
    return change <= m_absoluteChange || change <= m_settings.relativeObjectiveChange * std::abs(before);
}

OptimiserResult ScaledProblem::finish(const ScaledPoint& point, OptimiserMethod method,
                                      OptimiserTermination termination, std::size_t iterations,
                                      std::vector<OptimiserIteration> history) const
{
    // A variable within this fraction of its scaled range from a bound is reported at the bound (an active side
    // constraint).
    constexpr double BOUND_TOLERANCE = 1.0e-9;
    OptimiserResult result;
    result.method = method;
    result.x = toX(point.y);
    result.objective = point.objective;
    result.constraints = point.constraints;
    result.maxConstraint = point.maxConstraint;
    result.isFeasible = point.constraints.empty() || point.maxConstraint <= m_settings.constraintViolationThreshold;
    result.termination = termination;
    result.iterations = iterations;
    result.functionEvaluations = m_functionEvaluations;
    result.gradientEvaluations = m_gradientEvaluations;
    result.history = std::move(history);
    for (std::size_t j = 0; j < point.constraints.size(); ++j)
        if (point.constraints[j] >= m_settings.constraintActiveThreshold)
            result.activeConstraints.push_back(j);
    for (std::size_t i = 0; i < point.y.size(); ++i)
    {
        const double tolerance = BOUND_TOLERANCE * std::max(1.0, m_upper[i] - m_lower[i]);
        if (point.y[i] <= m_lower[i] + tolerance)
            result.activeBounds.push_back(-static_cast<std::int64_t>(i + 1));
        else if (point.y[i] >= m_upper[i] - tolerance)
            result.activeBounds.push_back(static_cast<std::int64_t>(i + 1));
    }
    return result;
}

} // namespace oos::design
