#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/optimiser/OptimisationProblem.hpp"
#include "oos/design/optimiser/OptimiserResult.hpp"
#include "oos/design/optimiser/OptimiserSettings.hpp"

#include <span>

namespace oos::design
{

// The modified method of feasible directions (METHOD = 1), Vanderplaats' method (G. N. Vanderplaats,
// "An efficient feasible directions algorithm for design synthesis", AIAA Journal 22(11), 1984; Numerical
// Optimization Techniques for Engineering Design, 4th ed., §5.4-5.6), in the scaled variables of ScaledProblem.
//
// Each iteration takes the gradients of the active (CT ≤ g ≤ CTMIN) and violated (g > CTMIN) constraints only and
// finds a direction S, its largest component 1:
// - no active or violated constraint: steepest descent, or the Fletcher-Reeves conjugate direction after another
//   unconstrained iteration, S = −∇f + (|∇f|²/|∇f_prev|²)·S_prev (restarted every n iterations);
// - active constraints only: the modified method's push-off factor 0, S = argmin ½|S|² + ∇f·S subject to ∇g_j·S ≤ 0
//   on the active constraints and the bounds (the projection of −∇f on the cone of feasible directions, solved by
//   QuadraticProgram); |S| below the Kuhn-Tucker tolerance ends the optimisation;
// - violated constraints: Zoutendijk's direction problem with push-off as CONMIN solves it (G. N. Vanderplaats, NASA
//   TM X-62282, 1973: THETA, PHI; subroutine CNMN05 of its source), minimise ∇f·S − Φ·β subject to
//   ∇g_j·S + θ_j·β ≤ 0, θ_j = (1 − g_j/CT)² at most 50 and scaled so that the largest is 1, −1 ≤ S_i ≤ 1, the
//   gradients normalised, Φ = 5 multiplied by 10 each time no direction reduces the violation, at most 1000, and
//   back to 5 at a feasible point (solved by LinearProgram); none at Φ = 1000 ends it at the best compromise.
// The line search goes along S from a first step that changes the objective by 10 % (ADS DLOBJ1) or moves the
// largest scaled variable by 0.2 (ADS DX2), doubling while the point improves and bisecting the first step that does
// not, a quadratic fit of the objective when it passed its minimum. With active constraints, a trial point is brought
// back onto g = 0 of the binding ones (a positive multiplier of the projection) and of those it pushed above zero, by
// least-norm Newton steps with their gradients at the start of the search, dependent gradients dropped (the
// "modified" part: the direction follows the boundary's tangent, the restoration returns to it, Rosen's gradient
// projection). A trial point is accepted when every constraint is at most CTMIN (from a feasible start) or the
// largest violation decreased (from an infeasible one). Along a search, a constraint the start had slightly outside
// (0 < g ≤ CTMIN) is held at its start value, not brought to g = 0 (the step would otherwise pay for a violation it
// did not make, and no step would improve the start); a Kuhn-Tucker point, or a point no step improves, left slightly
// outside is restored onto g = 0 (then returned, or the optimisation goes on from it).
//
// It stops when the objective changes by less than DABOBJ or DELOBJ·|f| for ITRMOP consecutive feasible iterations, on
// the Kuhn-Tucker test (|S| below STOL), when no step improves the point, or after ITMAX iterations (the defaults of
// ADS, OptimiserSettings).
//
// Thread-safety: stateless. Determinism: serial, in a fixed order.
class FeasibleDirections
{
public:
    // Throws OptimisationError (inconsistent problem sizes, infinite bounds, a non-finite objective) and
    // MemoryBudgetError.
    [[nodiscard]] static OptimiserResult run(const OptimisationProblem& problem, std::span<const double> start,
                                             const OptimiserSettings& settings, base::MemoryBudget& budget);
};

} // namespace oos::design
