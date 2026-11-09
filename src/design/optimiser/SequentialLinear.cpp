#include "oos/design/optimiser/SequentialLinear.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/design/optimiser/LinearProgram.hpp"
#include "oos/design/optimiser/OptimiserDefaults.hpp"
#include "oos/design/optimiser/ScaledProblem.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

namespace oos::design
{

namespace
{

// The first move limit, in scaled units: ADS's initial relative move limit of sequential linear programming,
// RMVLMZ = 0.2 (OptimiserDefaults.hpp), a fifth of each variable's starting magnitude.
constexpr double INITIAL_MOVE_LIMIT = optimiser_defaults::DIRECT_MOVE_LIMIT;
// The move limit as a trust region (R. Fletcher, Practical Methods of Optimization, 2nd ed., 1987, §5.1): with r the
// ratio of the actual to the predicted decrease of the merit, it shrinks to a quarter after a step with r < 0.25 and
// doubles (up to the first limit) after a step with r > 0.75 that reached it. This project also halves the limit of a
// variable whose step changed sign (the oscillation of SLP about a vertex that is not the optimum).
constexpr double MOVE_LIMIT_GROWTH = 2.0;
constexpr double MOVE_LIMIT_SHRINK = 0.25;
constexpr double OSCILLATION_SHRINK = 0.5;
constexpr double GOOD_AGREEMENT = 0.75;
constexpr double POOR_AGREEMENT = 0.25;
// Move limits below this (scaled) end the run: the design no longer moves.
constexpr double MINIMUM_MOVE_LIMIT = 1.0e-9;
// The weight of the violation in the elastic problem, relative to the objective's gradient: large enough that the
// least violation comes first.
constexpr double ELASTIC_WEIGHT = 1.0e3;
// The merit's penalty is at least this many times the largest LP multiplier (exactness of the L1 penalty needs more
// than the multiplier), and at least |f0| per unit of normalised violation.
constexpr double PENALTY_OVER_MULTIPLIER = 2.0;
// A step whose largest scaled component is below this has converged.
constexpr double STEP_TOLERANCE = 1.0e-9;
// A predicted decrease below this (relative to |f|) is no decrease.
constexpr double ZERO_DECREASE = 1.0e-14;

class Run
{
public:
    Run(const OptimisationProblem& problem, std::span<const double> start, const OptimiserSettings& settings,
        base::MemoryBudget& budget)
        : m_scaled(problem, settings, start, budget),
          m_settings(settings),
          m_n(m_scaled.variableCount()),
          m_m(m_scaled.constraintCount())
    {
    }

    OptimiserResult run()
    {
        ScaledPoint point = m_scaled.evaluate(m_scaled.start());
        m_penalty = std::max(std::abs(point.objective), 1.0);
        std::vector<OptimiserIteration> history{m_scaled.record(0, point)};
        std::vector<double> moveLimits(m_n, INITIAL_MOVE_LIMIT);
        std::vector<double> previousStep(m_n, 0.0);
        std::vector<std::size_t> rows(m_m);
        std::iota(rows.begin(), rows.end(), std::size_t{0});
        std::size_t smallChanges = 0;
        for (std::size_t iteration = 1; iteration <= m_settings.iterationLimit; ++iteration)
        {
            std::vector<double> gradient;
            std::span<const double> rowGradients;
            m_scaled.gradients(point, rows, gradient, rowGradients);
            std::optional<ScaledPoint> accepted;
            std::vector<double> step;
            while (!accepted)
            {
                if (*std::max_element(moveLimits.begin(), moveLimits.end()) < MINIMUM_MOVE_LIMIT)
                    return finish(point, OptimiserTermination::NoProgress, iteration, std::move(history));
                std::optional<std::vector<double>> solved = solveSubproblem(point, gradient, rowGradients, moveLimits);
                if (!solved)
                    return finish(point, OptimiserTermination::Infeasible, iteration, std::move(history));
                step = std::move(*solved);
                double largest = 0.0;
                for (const double value : step)
                    largest = std::max(largest, std::abs(value));
                const bool isFeasible = m_m == 0 || point.maxConstraint <= m_settings.constraintViolationThreshold;
                if (largest <= STEP_TOLERANCE)
                    return finish(point,
                                  isFeasible ? OptimiserTermination::KuhnTucker : OptimiserTermination::Infeasible,
                                  iteration, std::move(history));
                std::vector<double> y = point.y;
                for (std::size_t i = 0; i < m_n; ++i)
                    y[i] += step[i];
                m_scaled.clip(y);
                ScaledPoint candidate = m_scaled.evaluate(std::move(y));
                const double predicted = merit(point) - linearisedMerit(point, gradient, rowGradients, step);
                const double actual = merit(point) - merit(candidate);
                if (predicted > ZERO_DECREASE * std::max(1.0, std::abs(point.objective)) && actual > 0.0)
                {
                    const double agreement = actual / predicted;
                    for (std::size_t i = 0; i < m_n; ++i)
                    {
                        const bool atLimit = std::abs(step[i]) >= (1.0 - 1.0e-6) * moveLimits[i];
                        if (step[i] * previousStep[i] < 0.0)
                            moveLimits[i] *= OSCILLATION_SHRINK;
                        else if (agreement < POOR_AGREEMENT)
                            moveLimits[i] *= MOVE_LIMIT_SHRINK;
                        else if (agreement > GOOD_AGREEMENT && atLimit)
                            moveLimits[i] = std::min(moveLimits[i] * MOVE_LIMIT_GROWTH, INITIAL_MOVE_LIMIT);
                    }
                    accepted = std::move(candidate);
                }
                else
                    for (double& limit : moveLimits)
                        limit *= MOVE_LIMIT_SHRINK;
            }
            previousStep = step;
            const bool isFeasible = m_m == 0 || accepted->maxConstraint <= m_settings.constraintViolationThreshold;
            smallChanges = isFeasible && m_scaled.isSmallObjectiveChange(point.objective, accepted->objective)
                               ? smallChanges + 1
                               : 0;
            point = std::move(*accepted);
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
        return m_scaled.finish(point, OptimiserMethod::SequentialLinear, termination, iterations, std::move(history));
    }

    [[nodiscard]] double merit(const ScaledPoint& point) const noexcept
    {
        double violation = 0.0;
        for (const double g : point.constraints)
            violation += std::max(g, 0.0);
        return point.objective + m_penalty * violation;
    }

    [[nodiscard]] double linearisedMerit(const ScaledPoint& point, const std::vector<double>& gradient,
                                         std::span<const double> rowGradients, const std::vector<double>& step) const
    {
        double value = point.objective;
        for (std::size_t i = 0; i < m_n; ++i)
            value += gradient[i] * step[i];
        double violation = 0.0;
        for (std::size_t j = 0; j < m_m; ++j)
        {
            double g = point.constraints[j];
            for (std::size_t i = 0; i < m_n; ++i)
                g += rowGradients[j * m_n + i] * step[i];
            violation += std::max(g, 0.0);
        }
        return value + m_penalty * violation;
    }

    // The LP of the iteration (elastic when the linearised constraints have no solution within the move limits); the
    // penalty of the merit raised from its multipliers.
    std::optional<std::vector<double>> solveSubproblem(const ScaledPoint& point, const std::vector<double>& gradient,
                                                       std::span<const double> rowGradients,
                                                       const std::vector<double>& moveLimits)
    {
        std::vector<double> lower(m_n);
        std::vector<double> upper(m_n);
        for (std::size_t i = 0; i < m_n; ++i)
        {
            lower[i] = std::min(0.0, std::max(m_scaled.lower()[i] - point.y[i], -moveLimits[i]));
            upper[i] = std::max(0.0, std::min(m_scaled.upper()[i] - point.y[i], moveLimits[i]));
        }
        std::vector<double> b(m_m);
        for (std::size_t j = 0; j < m_m; ++j)
            b[j] = -point.constraints[j];
        LinearProgramSolution solution =
            LinearProgram::solve(gradient, rowGradients, b, lower, upper, m_scaled.budget());
        if (solution.status == LinearProgramStatus::Optimal)
        {
            raisePenalty(solution.multipliers);
            return solution.x;
        }
        // Elastic: (d, z), minimise ∇f·d + M·z, ∇g_j·d − z ≤ −g_j, 0 ≤ z ≤ max(max g, 0) (d = 0, z = max g is
        // feasible).
        const std::size_t columns = m_n + 1;
        double gradientScale = 1.0;
        for (const double value : gradient)
            gradientScale = std::max(gradientScale, std::abs(value));
        std::vector<double> cost(gradient.begin(), gradient.end());
        cost.push_back(ELASTIC_WEIGHT * gradientScale);
        base::BudgetedBuffer<double> a(m_scaled.budget(), m_m * columns, "SLP elastic rows");
        for (std::size_t j = 0; j < m_m; ++j)
        {
            for (std::size_t i = 0; i < m_n; ++i)
                a[j * columns + i] = rowGradients[j * m_n + i];
            a[j * columns + m_n] = -1.0;
        }
        lower.push_back(0.0);
        upper.push_back(std::max(point.maxConstraint, 0.0));
        solution = LinearProgram::solve(cost, a.span(), b, lower, upper, m_scaled.budget());
        if (solution.status != LinearProgramStatus::Optimal)
            return std::nullopt;
        raisePenalty(solution.multipliers);
        solution.x.pop_back();
        return solution.x;
    }

    void raisePenalty(const std::vector<double>& multipliers)
    {
        for (const double multiplier : multipliers)
            m_penalty = std::max(m_penalty, PENALTY_OVER_MULTIPLIER * multiplier);
    }

    ScaledProblem m_scaled;
    const OptimiserSettings& m_settings;
    std::size_t m_n;
    std::size_t m_m;
    double m_penalty = 1.0;
};

} // namespace

OptimiserResult SequentialLinear::run(const OptimisationProblem& problem, std::span<const double> start,
                                      const OptimiserSettings& settings, base::MemoryBudget& budget)
{
    Run run(problem, start, settings, budget);
    return run.run();
}

} // namespace oos::design
