#pragma once

#include <cstddef>
#include <span>

namespace oos::design
{

// A constrained minimisation problem as the optimisers of this folder see it (the plug-in point between them and what
// they optimise: the ApproximateProblem of a design cycle, a problem of the caller, the analytic problems of the
// tests):
//
//     minimise f(x)  subject to  g_j(x) ≤ 0 (j = 1 … m)  and  lower ≤ x ≤ upper,
//
// the constraints normalised (the normalised g of ConstraintNormalisation, so that CT, CTMIN and GMAX keep their
// meaning). The optimisers call it once per trial point (evaluate) and once per iteration for the gradients of the
// constraints they need (the active and violated ones, as DesignOptimParams IGMAX = 0 asks), never per variable.
//
// Thread-safety: the optimisers call it from one thread. Ownership: an interface; it owns nothing of the optimiser.
class OptimisationProblem
{
public:
    OptimisationProblem() = default;
    OptimisationProblem(const OptimisationProblem&) = default;
    OptimisationProblem(OptimisationProblem&&) = default;
    OptimisationProblem& operator=(const OptimisationProblem&) = default;
    OptimisationProblem& operator=(OptimisationProblem&&) = default;
    virtual ~OptimisationProblem() = default;

    [[nodiscard]] virtual std::size_t variableCount() const = 0;
    [[nodiscard]] virtual std::size_t constraintCount() const = 0;
    // The bounds of x, one value per variable; lower ≤ upper, both finite.
    [[nodiscard]] virtual std::span<const double> lowerBounds() const = 0;
    [[nodiscard]] virtual std::span<const double> upperBounds() const = 0;

    // f(x), and g(x) written into `constraints` (constraintCount() values).
    [[nodiscard]] virtual double evaluate(std::span<const double> x, std::span<double> constraints) const = 0;

    // ∇f(x) into `objectiveGradient` (n values), and the gradients of the constraints `rows` into
    // `constraintGradients`, row-major: row k is ∇g_rows[k] (rows.size() × n values).
    virtual void gradients(std::span<const double> x, std::span<double> objectiveGradient,
                           std::span<const std::size_t> rows, std::span<double> constraintGradients) const = 0;
};

} // namespace oos::design
