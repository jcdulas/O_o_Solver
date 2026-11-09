#include "oos/design/approximation/MoveLimits.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace oos::design
{

MoveLimitParameters MoveLimitParameters::fromParameters(const OptimisationParameters& parameters)
{
    const double fallback =
        defaultMoveLimit(approximationKindOfCode(parameters.integer(DesignOptimParams::ApproximationMethod)));
    const auto givenOr = [&](DesignOptimParams parameter)
    { return parameters.isSet(parameter) ? parameters.value(parameter) : fallback; };
    MoveLimitParameters result;
    result.designVarMoveLimit = givenOr(DesignOptimParams::DesignVarMoveLimit);
    result.minDesignVarChange = parameters.value(DesignOptimParams::MinDesignVarChange);
    result.propertyMoveLimit = givenOr(DesignOptimParams::PropertyMoveLimit);
    result.minPropertyChange = parameters.value(DesignOptimParams::MinPropertyChange);
    return result;
}

double MoveLimitParameters::defaultMoveLimit(ApproximationKind kind) noexcept
{
    return kind == ApproximationKind::Direct ? optimiser_defaults::DIRECT_MOVE_LIMIT
                                             : optimiser_defaults::CONVEX_MOVE_LIMIT;
}

Range MoveLimits::variableRange(double x0, double xlb, double xub, std::optional<double> delxv) const noexcept
{
    if (!m_parameters.isEnabled)
        return {xlb, xub};
    const double move =
        std::max(delxv.value_or(m_parameters.designVarMoveLimit) * std::abs(x0), m_parameters.minDesignVarChange);
    return {std::max(xlb, x0 - move), std::min(xub, x0 + move)};
}

Range MoveLimits::propertyRange(double p0, std::optional<double> pmin, std::optional<double> pmax) const noexcept
{
    constexpr double INF = std::numeric_limits<double>::infinity();
    Range range{pmin.value_or(-INF), pmax.value_or(INF)};
    if (!m_parameters.isEnabled)
        return range;
    const double move = std::max(m_parameters.propertyMoveLimit * std::abs(p0), m_parameters.minPropertyChange);
    range.lower = std::max(range.lower, p0 - move);
    range.upper = std::min(range.upper, p0 + move);
    return range;
}

} // namespace oos::design
