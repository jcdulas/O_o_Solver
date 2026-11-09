#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/optimiser/OptimisationProblem.hpp"
#include "oos/design/optimiser/OptimiserResult.hpp"
#include "oos/design/optimiser/OptimiserSettings.hpp"

#include <span>

namespace oos::design
{

// The optimiser of DesignOptimParams METHOD: FeasibleDirections (1), SequentialLinear (2), SequentialQuadratic (3).
// METHOD = 0 (automatic, the default) takes the modified method of feasible directions: the format gives no rule for
// the automatic choice, and this method suits the small, few-variable problems of a sizing run.
//
// Thread-safety: stateless.
class Optimiser
{
public:
    // The method that `method` runs (Automatic resolved).
    [[nodiscard]] static OptimiserMethod resolve(OptimiserMethod method) noexcept;
    // Runs the optimiser of `settings.method` from `start`. Throws as FeasibleDirections::run.
    [[nodiscard]] static OptimiserResult run(const OptimisationProblem& problem, std::span<const double> start,
                                             const OptimiserSettings& settings, base::MemoryBudget& budget);
};

} // namespace oos::design
