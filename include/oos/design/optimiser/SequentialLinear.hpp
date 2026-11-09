#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/optimiser/OptimisationProblem.hpp"
#include "oos/design/optimiser/OptimiserResult.hpp"
#include "oos/design/optimiser/OptimiserSettings.hpp"

#include <span>

namespace oos::design
{

// Sequential linear programming (METHOD = 2; Vanderplaats, Numerical Optimization Techniques for Engineering Design,
// 4th ed., §6.2-6.3: SLP with move limits), in the scaled variables of ScaledProblem. Each iteration linearises the
// objective and every constraint at y and solves
//
//     minimise ∇f·d  subject to  g_j + ∇g_j·d ≤ 0  and  max(lower − y, −δ) ≤ d ≤ min(upper − y, δ)
//
// by LinearProgram (its own simplex), δ the move limits of the iteration (0.2 at the start in scaled units, ADS's
// RMVLMZ for this strategy, NASA CR-177985, Table 6). When the linearised constraints have no solution within δ, the
// elastic problem minimises ∇f·d + M·z with g_j + ∇g_j·d ≤ z, z ≥ 0 (the least violation, then the objective). A step
// is accepted when it decreases the exact penalty merit φ = f + μ·Σ max(0, g_j) (μ at least twice the largest
// multiplier of the LP); δ follows Fletcher's trust region rule (a quarter after a poor agreement or a rejected step,
// twice after a good one at the limit) and is halved for a variable whose step changed sign. It stops as
// FeasibleDirections does (DABOBJ, DELOBJ, ITRMOP), when the move limits or the step vanish, or after ITMAX
// iterations.
//
// Thread-safety: stateless. Determinism: serial, in a fixed order.
class SequentialLinear
{
public:
    // Throws as FeasibleDirections::run.
    [[nodiscard]] static OptimiserResult run(const OptimisationProblem& problem, std::span<const double> start,
                                             const OptimiserSettings& settings, base::MemoryBudget& budget);
};

} // namespace oos::design
