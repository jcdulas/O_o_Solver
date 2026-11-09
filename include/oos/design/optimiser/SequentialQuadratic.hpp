#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/optimiser/OptimisationProblem.hpp"
#include "oos/design/optimiser/OptimiserResult.hpp"
#include "oos/design/optimiser/OptimiserSettings.hpp"

#include <span>

namespace oos::design
{

// Sequential quadratic programming (METHOD = 3), the Han-Powell method (M. J. D. Powell, "A fast algorithm for
// nonlinearly constrained optimization calculations", Numerical Analysis, Lecture Notes in Mathematics 630, 1978;
// Nocedal and Wright, Numerical Optimization, 2nd ed., Algorithm 18.3), in the scaled variables of ScaledProblem.
// Each iteration solves
//
//     minimise ½·dᵀ·B·d + ∇f·d  subject to  g_j + ∇g_j·d ≤ 0  and  lower − y ≤ d ≤ upper − y
//
// by QuadraticProgram (Goldfarb-Idnani, written here), B the quasi-Newton approximation of the Hessian of the
// Lagrangian (identity at the start; Powell's damped BFGS update, which keeps it positive definite), elastic with one
// more variable z ≥ 0 when the linearised constraints are inconsistent. The step α·d is taken by backtracking on the
// exact penalty merit φ = f + μ·Σ max(0, g_j), with μ ≥ 1.5 × the largest multiplier (Armijo condition with η = 0.1).
// It stops when the step vanishes at a feasible point (Kuhn-Tucker), as FeasibleDirections does on the objective
// (DABOBJ, DELOBJ, ITRMOP), or after ITMAX iterations. The quasi-Newton matrix is reset to the identity when the line
// search fails, and the run ends when it fails again.
//
// Thread-safety: stateless. Determinism: serial, in a fixed order.
class SequentialQuadratic
{
public:
    // Throws as FeasibleDirections::run.
    [[nodiscard]] static OptimiserResult run(const OptimisationProblem& problem, std::span<const double> start,
                                             const OptimiserSettings& settings, base::MemoryBudget& budget);
};

} // namespace oos::design
