#include "oos/design/optimiser/SequentialQuadratic.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/design/optimiser/QuadraticProgram.hpp"
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

// Armijo's sufficient decrease of the merit along the step, and the backtracking.
constexpr double ARMIJO_FRACTION = 0.1;
constexpr double BACKTRACKING_FACTOR = 0.5;
constexpr int BACKTRACKING_LIMIT = 40;
// The merit's penalty: at least this many times the largest multiplier (Powell), at least |f0| per unit of violation.
constexpr double PENALTY_OVER_MULTIPLIER = 1.5;
// Powell's damping of the BFGS update: sᵀy at least this fraction of sᵀBs.
constexpr double DAMPING_THRESHOLD = 0.2;
constexpr double DAMPED_FRACTION = 0.8;
// The elastic variable of an inconsistent subproblem: its weight relative to the objective's gradient, and the
// curvature that keeps the subproblem strictly convex.
constexpr double ELASTIC_WEIGHT = 1.0e3;
constexpr double ELASTIC_CURVATURE = 1.0;
// A step whose largest scaled component is below this (relative to 1 + |y|) has converged.
constexpr double STEP_TOLERANCE = 1.0e-9;

double dot(std::span<const double> a, std::span<const double> b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

class Run
{
public:
    Run(const OptimisationProblem& problem, std::span<const double> start, const OptimiserSettings& settings,
        base::MemoryBudget& budget)
        : m_scaled(problem, settings, start, budget),
          m_settings(settings),
          m_n(m_scaled.variableCount()),
          m_m(m_scaled.constraintCount()),
          m_hessian(budget, m_n * m_n, "SQP quasi-Newton matrix")
    {
        resetHessian();
    }

    OptimiserResult run()
    {
        ScaledPoint point = m_scaled.evaluate(m_scaled.start());
        m_penalty = std::max(std::abs(point.objective), 1.0);
        std::vector<OptimiserIteration> history{m_scaled.record(0, point)};
        std::vector<std::size_t> rows(m_m);
        std::iota(rows.begin(), rows.end(), std::size_t{0});
        std::size_t smallChanges = 0;
        bool hasFailedSearch = false;
        std::vector<double> multipliers(m_m, 0.0);
        std::optional<std::vector<double>> previousLagrangian; // ∇L at the previous point with the current multipliers
        std::vector<double> previousStep;
        for (std::size_t iteration = 1; iteration <= m_settings.iterationLimit; ++iteration)
        {
            std::vector<double> gradient;
            std::span<const double> rowGradients;
            m_scaled.gradients(point, rows, gradient, rowGradients);
            if (previousLagrangian)
                update(previousStep, lagrangianGradient(gradient, rowGradients, multipliers), *previousLagrangian);

            const std::optional<QuadraticProgramSolution> solved = solveSubproblem(point, gradient, rowGradients);
            if (!solved)
                return finish(point, OptimiserTermination::Infeasible, iteration, std::move(history));
            std::vector<double> step(solved->x.begin(), solved->x.begin() + static_cast<std::ptrdiff_t>(m_n));
            for (std::size_t j = 0; j < m_m; ++j)
                multipliers[j] = solved->multipliers[j];
            for (const double multiplier : multipliers)
                m_penalty = std::max(m_penalty, PENALTY_OVER_MULTIPLIER * multiplier);

            double largestStep = 0.0;
            double largestY = 0.0;
            for (std::size_t i = 0; i < m_n; ++i)
            {
                largestStep = std::max(largestStep, std::abs(step[i]));
                largestY = std::max(largestY, std::abs(point.y[i]));
            }
            const bool isFeasible = m_m == 0 || point.maxConstraint <= m_settings.constraintViolationThreshold;
            if (largestStep <= STEP_TOLERANCE * (1.0 + largestY))
                return finish(point, isFeasible ? OptimiserTermination::KuhnTucker : OptimiserTermination::Infeasible,
                              iteration, std::move(history));

            // Backtracking on the L1 merit; its directional derivative along d is ∇f·d − μ·Σ max(0, g_j) when the
            // linearised constraints hold (Nocedal and Wright, eq. 18.29).
            const double base = merit(point);
            double violation = 0.0;
            for (const double g : point.constraints)
                violation += std::max(g, 0.0);
            const double slope = std::min(dot(gradient, step) - m_penalty * violation, 0.0);
            std::optional<ScaledPoint> accepted;
            double alpha = 1.0;
            for (int trial = 0; trial < BACKTRACKING_LIMIT; ++trial, alpha *= BACKTRACKING_FACTOR)
            {
                std::vector<double> y = point.y;
                for (std::size_t i = 0; i < m_n; ++i)
                    y[i] += alpha * step[i];
                m_scaled.clip(y);
                ScaledPoint candidate = m_scaled.evaluate(std::move(y));
                if (merit(candidate) <= base + ARMIJO_FRACTION * alpha * slope && merit(candidate) < base)
                {
                    accepted = std::move(candidate);
                    break;
                }
            }
            if (!accepted)
            {
                if (hasFailedSearch)
                    return finish(point,
                                  isFeasible ? OptimiserTermination::NoProgress : OptimiserTermination::Infeasible,
                                  iteration, std::move(history));
                hasFailedSearch = true;
                resetHessian();
                previousLagrangian.reset();
                continue;
            }
            hasFailedSearch = false;
            previousLagrangian = lagrangianGradient(gradient, rowGradients, multipliers);
            previousStep.assign(m_n, 0.0);
            for (std::size_t i = 0; i < m_n; ++i)
                previousStep[i] = accepted->y[i] - point.y[i];
            const bool isNowFeasible = m_m == 0 || accepted->maxConstraint <= m_settings.constraintViolationThreshold;
            smallChanges = isNowFeasible && m_scaled.isSmallObjectiveChange(point.objective, accepted->objective)
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
        return m_scaled.finish(point, OptimiserMethod::SequentialQuadratic, termination, iterations,
                               std::move(history));
    }

    void resetHessian()
    {
        std::fill(m_hessian.data(), m_hessian.data() + m_hessian.size(), 0.0);
        for (std::size_t i = 0; i < m_n; ++i)
            m_hessian[i * m_n + i] = 1.0;
    }

    [[nodiscard]] double merit(const ScaledPoint& point) const noexcept
    {
        double violation = 0.0;
        for (const double g : point.constraints)
            violation += std::max(g, 0.0);
        return point.objective + m_penalty * violation;
    }

    [[nodiscard]] std::vector<double> lagrangianGradient(const std::vector<double>& gradient,
                                                         std::span<const double> rowGradients,
                                                         const std::vector<double>& multipliers) const
    {
        std::vector<double> result = gradient;
        for (std::size_t j = 0; j < m_m; ++j)
            if (multipliers[j] != 0.0)
                for (std::size_t i = 0; i < m_n; ++i)
                    result[i] += multipliers[j] * rowGradients[j * m_n + i];
        return result;
    }

    // Powell's damped BFGS update of B with s the step and y the change of ∇L.
    void update(const std::vector<double>& s, const std::vector<double>& current, const std::vector<double>& previous)
    {
        std::vector<double> y(m_n);
        for (std::size_t i = 0; i < m_n; ++i)
            y[i] = current[i] - previous[i];
        std::vector<double> bs(m_n, 0.0);
        for (std::size_t i = 0; i < m_n; ++i)
            for (std::size_t k = 0; k < m_n; ++k)
                bs[i] += m_hessian[i * m_n + k] * s[k];
        const double sbs = dot(s, bs);
        if (!(sbs > 0.0))
            return;
        double sy = dot(s, y);
        if (sy < DAMPING_THRESHOLD * sbs)
        {
            const double theta = DAMPED_FRACTION * sbs / (sbs - sy);
            for (std::size_t i = 0; i < m_n; ++i)
                y[i] = theta * y[i] + (1.0 - theta) * bs[i];
            sy = dot(s, y);
        }
        if (!(sy > 0.0))
            return;
        for (std::size_t i = 0; i < m_n; ++i)
            for (std::size_t k = 0; k < m_n; ++k)
                m_hessian[i * m_n + k] += y[i] * y[k] / sy - bs[i] * bs[k] / sbs;
    }

    // The QP of the iteration over every constraint and the bounds; elastic when it has no solution.
    std::optional<QuadraticProgramSolution> solveSubproblem(const ScaledPoint& point,
                                                            const std::vector<double>& gradient,
                                                            std::span<const double> rowGradients)
    {
        for (const bool isElastic : {false, true})
        {
            const std::size_t columns = isElastic ? m_n + 1 : m_n;
            const std::size_t rowCount = m_m + 2 * m_n + (isElastic ? 1 : 0);
            base::MemoryBudget& budget = m_scaled.budget();
            base::BudgetedBuffer<double> metric(budget, columns * columns, "SQP subproblem metric");
            for (std::size_t i = 0; i < m_n; ++i)
                for (std::size_t k = 0; k < m_n; ++k)
                    metric[i * columns + k] = m_hessian[i * m_n + k];
            std::vector<double> linear(gradient.begin(), gradient.end());
            double gradientScale = 1.0;
            for (const double value : gradient)
                gradientScale = std::max(gradientScale, std::abs(value));
            if (isElastic)
            {
                metric[m_n * columns + m_n] = ELASTIC_CURVATURE;
                linear.push_back(ELASTIC_WEIGHT * gradientScale);
            }
            base::BudgetedBuffer<double> normals(budget, rowCount * columns, "SQP subproblem rows");
            std::vector<double> rhs(rowCount, 0.0);
            // −∇g_j·d (+ z) ≥ g_j.
            for (std::size_t j = 0; j < m_m; ++j)
            {
                for (std::size_t i = 0; i < m_n; ++i)
                    normals[j * columns + i] = -rowGradients[j * m_n + i];
                if (isElastic)
                    normals[j * columns + m_n] = 1.0;
                rhs[j] = point.constraints[j];
            }
            // d_i ≥ lower_i − y_i and −d_i ≥ y_i − upper_i.
            for (std::size_t i = 0; i < m_n; ++i)
            {
                normals[(m_m + 2 * i) * columns + i] = 1.0;
                rhs[m_m + 2 * i] = m_scaled.lower()[i] - point.y[i];
                normals[(m_m + 2 * i + 1) * columns + i] = -1.0;
                rhs[m_m + 2 * i + 1] = point.y[i] - m_scaled.upper()[i];
            }
            if (isElastic)
                normals[(rowCount - 1) * columns + m_n] = 1.0; // z ≥ 0
            QuadraticProgramSolution solution =
                QuadraticProgram::solve(metric.span(), linear, normals.span(), rhs, budget);
            if (solution.status == QuadraticProgramStatus::Optimal)
                return solution;
        }
        return std::nullopt;
    }

    ScaledProblem m_scaled;
    const OptimiserSettings& m_settings;
    std::size_t m_n;
    std::size_t m_m;
    base::BudgetedBuffer<double> m_hessian;
    double m_penalty = 1.0;
};

} // namespace

OptimiserResult SequentialQuadratic::run(const OptimisationProblem& problem, std::span<const double> start,
                                         const OptimiserSettings& settings, base::MemoryBudget& budget)
{
    Run run(problem, start, settings, budget);
    return run.run();
}

} // namespace oos::design
