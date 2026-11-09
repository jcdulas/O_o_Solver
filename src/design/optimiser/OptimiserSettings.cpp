#include "oos/design/optimiser/OptimiserSettings.hpp"

namespace oos::design
{

std::string_view optimiserMethodName(OptimiserMethod method) noexcept
{
    switch (method)
    {
    case OptimiserMethod::Automatic:
        return "automatic";
    case OptimiserMethod::FeasibleDirections:
        return "mfd";
    case OptimiserMethod::SequentialLinear:
        return "slp";
    case OptimiserMethod::SequentialQuadratic:
        return "sqp";
    }
    return "";
}

std::optional<OptimiserMethod> findOptimiserMethod(std::string_view name) noexcept
{
    if (name == "automatic" || name == "auto")
        return OptimiserMethod::Automatic;
    if (name == "mfd" || name == "feasible_directions")
        return OptimiserMethod::FeasibleDirections;
    if (name == "slp" || name == "sequential_linear")
        return OptimiserMethod::SequentialLinear;
    if (name == "sqp" || name == "sequential_quadratic")
        return OptimiserMethod::SequentialQuadratic;
    return std::nullopt;
}

OptimiserSettings OptimiserSettings::fromParameters(const OptimisationParameters& parameters)
{
    OptimiserSettings settings;
    settings.method = static_cast<OptimiserMethod>(parameters.integer(DesignOptimParams::OptimizationMethod));
    settings.constraintActiveThreshold = parameters.value(DesignOptimParams::ConstraintActiveThreshold);
    settings.constraintViolationThreshold = parameters.value(DesignOptimParams::ConstraintViolationThreshold);
    if (parameters.isSet(DesignOptimParams::AbsoluteObjectiveChange))
        settings.givenAbsoluteObjectiveChange = parameters.value(DesignOptimParams::AbsoluteObjectiveChange);
    settings.relativeObjectiveChange = parameters.value(DesignOptimParams::RelativeObjectiveChange);
    settings.isScaled = parameters.integer(DesignOptimParams::DesignVarScaling) >= 0;
    return settings;
}

double OptimiserSettings::absoluteObjectiveChange(double initialObjective) const noexcept
{
    return givenAbsoluteObjectiveChange ? *givenAbsoluteObjectiveChange
                                        : optimiser_defaults::absoluteChangeOf(initialObjective);
}

} // namespace oos::design
