#pragma once

#include "oos/design/optimiser/OptimiserSettings.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace oos::design
{

// Why an approximate optimisation stopped.
enum class OptimiserTermination : std::uint8_t
{
    ObjectiveConverged, // the objective changed by less than DABOBJ or DELOBJ for ITRMOP iterations, feasible
    KuhnTucker,         // no usable feasible direction: the Kuhn-Tucker conditions hold within their tolerance
    NoProgress,         // the step became too small to change the design
    IterationLimit,     // ITMAX iterations
    Infeasible          // no move reduces the violation further: the best compromise point is returned
};

[[nodiscard]] std::string_view optimiserTerminationName(OptimiserTermination termination) noexcept;

// One iteration of an optimiser, as the iteration history prints them (IPRINT ≥ 2).
struct OptimiserIteration
{
    std::size_t iteration = 0;
    double objective = 0.0;
    double maxConstraint = 0.0;
    std::size_t activeConstraints = 0;   // CT ≤ g ≤ CTMIN
    std::size_t violatedConstraints = 0; // g > CTMIN
};

// The result of an approximate optimisation: the point, its objective and constraints, why it stopped, and the counts
// of the calls to the problem (each evaluate() a function evaluation, each gradients() a gradient evaluation).
struct OptimiserResult
{
    OptimiserMethod method = OptimiserMethod::FeasibleDirections; // the optimiser that ran
    std::vector<double> x;
    double objective = 0.0;
    std::vector<double> constraints;
    double maxConstraint = 0.0; // the largest g (−∞ without constraints)
    bool isFeasible = true;     // maxConstraint ≤ CTMIN
    OptimiserTermination termination = OptimiserTermination::ObjectiveConverged;
    std::size_t iterations = 0;
    std::size_t functionEvaluations = 0;
    std::size_t gradientEvaluations = 0;
    std::vector<OptimiserIteration> history;
    std::vector<std::size_t> activeConstraints; // CT ≤ g at the end
    // The variables at a bound at the end: −(i + 1) at the lower bound, i + 1 at the upper one.
    std::vector<std::int64_t> activeBounds;
};

} // namespace oos::design
