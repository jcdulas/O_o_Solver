#pragma once

#include "oos/design/optimiser/OptimisationParameters.hpp"
#include "oos/design/optimiser/OptimiserDefaults.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace oos::design
{

// The optimiser of an approximate problem (DOPTPRM METHOD).
enum class OptimiserMethod : std::uint8_t
{
    Automatic,          // METHOD = 0: the modified method of feasible directions
    FeasibleDirections, // METHOD = 1: Vanderplaats' modified method of feasible directions
    SequentialLinear,   // METHOD = 2: sequential linear programming
    SequentialQuadratic // METHOD = 3: sequential quadratic programming
};

[[nodiscard]] std::string_view optimiserMethodName(OptimiserMethod method) noexcept;
// "mfd", "slp", "sqp", "automatic" (also "feasible_directions", "sequential_linear", "sequential_quadratic"); nothing
// for another name.
[[nodiscard]] std::optional<OptimiserMethod> findOptimiserMethod(std::string_view name) noexcept;

// The settings of one approximate optimisation, from DesignOptimParams (CT, CTMIN, DABOBJ, DELOBJ, ISCAL, METHOD) and
// the internal controls of the optimisers, which the input format does not give. Every default is the one of the ADS
// optimizer level (OptimiserDefaults.hpp: CT, CTMIN, DABOBJ, DELOBJ, ITMAX, ITRMOP, STOL of NASA CR-177985, Tables 6
// and 8).
struct OptimiserSettings
{
    // ITMAX: the iterations of one approximate optimisation (ADS: 40).
    static constexpr std::size_t DEFAULT_ITERATION_LIMIT = optimiser_defaults::ITERATION_LIMIT;
    // ITRMOP: the consecutive iterations of small objective change that end it (ADS: 3).
    static constexpr std::size_t DEFAULT_CONVERGED_ITERATIONS = optimiser_defaults::CONVERGED_ITERATIONS;
    // STOL: the Kuhn-Tucker tolerance on the search direction (ADS: 0.001).
    static constexpr double DEFAULT_KUHN_TUCKER_TOLERANCE = optimiser_defaults::KUHN_TUCKER_TOLERANCE;

    OptimiserMethod method = OptimiserMethod::Automatic;
    // CT: a constraint above it is active.
    double constraintActiveThreshold = optimiser_defaults::ACTIVE_CONSTRAINT_THRESHOLD;
    // CTMIN: a constraint above it is violated.
    double constraintViolationThreshold = optimiser_defaults::VIOLATED_CONSTRAINT_THRESHOLD;
    // DABOBJ; nothing: |F0|/1000 (at least 1.0E-10), F0 the objective at the start.
    std::optional<double> givenAbsoluteObjectiveChange;
    double relativeObjectiveChange = optimiser_defaults::ITERATION_RELATIVE_OBJECTIVE_CHANGE; // DELOBJ
    bool isScaled = true; // ISCAL ≥ 0: the variables scaled by their magnitude at the start (ADS scales by default)
    std::size_t iterationLimit = DEFAULT_ITERATION_LIMIT;
    std::size_t convergedIterations = DEFAULT_CONVERGED_ITERATIONS;
    double kuhnTuckerTolerance = DEFAULT_KUHN_TUCKER_TOLERANCE;

    // The settings of a DesignOptimParams: METHOD, CT, CTMIN, DABOBJ (when given), DELOBJ, ISCAL.
    [[nodiscard]] static OptimiserSettings fromParameters(const OptimisationParameters& parameters);
    // DABOBJ for a start of objective `initialObjective`.
    [[nodiscard]] double absoluteObjectiveChange(double initialObjective) const noexcept;
};

} // namespace oos::design
