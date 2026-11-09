#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/optimiser/OptimisationProblem.hpp"
#include "oos/design/optimiser/OptimiserResult.hpp"
#include "oos/design/optimiser/OptimiserSettings.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace oos::design
{

// A point of an optimisation: its variables (scaled), objective and constraints.
struct ScaledPoint
{
    std::vector<double> y;
    double objective = 0.0;
    std::vector<double> constraints;
    double maxConstraint = 0.0;
};

// The problem as the optimisers move in it: the variables scaled, y_i = x_i / s_i, with s_i = |x_i| at the start
// (ISCAL ≥ 0; 1 for a variable at zero, and every s_i = 1 with ISCAL = −1). Sizing variables span decades (an area of
// 0.1 and a thickness of 10); in y each starts at ±1, so that one step length, one move limit and the norm of a
// direction mean the same for all of them (Vanderplaats, Numerical Optimization Techniques for Engineering Design,
// 4th ed., §6.4 "Scaling"; for example ∇f = (2.83, 1.00) at x = (1, 2) gives the steepest descent direction
// −(1, 0.707) in y).
//
// It counts the calls to the problem and gives the result in x. The constraint gradients it hands out live in a
// budgeted buffer (rows × variables grows with the model).
//
// Thread-safety: one optimiser run. Ownership: refers to the problem and the budget, which must outlive it.
class ScaledProblem
{
public:
    ScaledProblem(const OptimisationProblem& problem, const OptimiserSettings& settings, std::span<const double> start,
                  base::MemoryBudget& budget);

    [[nodiscard]] std::size_t variableCount() const noexcept { return m_scale.size(); }
    [[nodiscard]] std::size_t constraintCount() const noexcept { return m_constraintCount; }
    [[nodiscard]] const std::vector<double>& lower() const noexcept { return m_lower; }
    [[nodiscard]] const std::vector<double>& upper() const noexcept { return m_upper; }
    [[nodiscard]] const std::vector<double>& start() const noexcept { return m_start; }
    [[nodiscard]] const OptimiserSettings& settings() const noexcept { return m_settings; }
    [[nodiscard]] base::MemoryBudget& budget() const noexcept { return m_budget; }

    // The point y with its objective and constraints (one function evaluation).
    [[nodiscard]] ScaledPoint evaluate(std::vector<double> y);
    // ∇f in y, and the gradients in y of the constraints `rows`, row-major (rows.size() × n), into `rows`-sized
    // storage this object owns until the next call (one gradient evaluation).
    void gradients(const ScaledPoint& point, std::span<const std::size_t> rows, std::vector<double>& objectiveGradient,
                   std::span<const double>& constraintGradients);

    // y clipped into the bounds.
    void clip(std::vector<double>& y) const noexcept;

    // The result at `point`, in x: active constraints (g ≥ CT) and bounds, feasibility (max g ≤ CTMIN), counts.
    [[nodiscard]] OptimiserResult finish(const ScaledPoint& point, OptimiserMethod method,
                                         OptimiserTermination termination, std::size_t iterations,
                                         std::vector<OptimiserIteration> history) const;

    // The iteration record of a point.
    [[nodiscard]] OptimiserIteration record(std::size_t iteration, const ScaledPoint& point) const noexcept;

    // Whether |Δf| ≤ DABOBJ or |Δf| ≤ DELOBJ·|f| (the convergence test of one iteration).
    [[nodiscard]] bool isSmallObjectiveChange(double before, double after) const noexcept;

private:
    [[nodiscard]] std::vector<double> toX(std::span<const double> y) const;

    const OptimisationProblem& m_problem;
    OptimiserSettings m_settings;
    base::MemoryBudget& m_budget;
    std::size_t m_constraintCount = 0;
    std::vector<double> m_scale;
    std::vector<double> m_lower;
    std::vector<double> m_upper;
    std::vector<double> m_start;
    double m_absoluteChange = 0.0;
    std::size_t m_functionEvaluations = 0;
    std::size_t m_gradientEvaluations = 0;
    base::BudgetedBuffer<double> m_rowGradients;
};

} // namespace oos::design
