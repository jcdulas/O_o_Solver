#include "oos/design/constraints/ConstraintNormalisation.hpp"

#include <cmath>

namespace oos::design
{

double ConstraintNormalisation::scale(double bound) const noexcept
{
    return std::abs(bound) > m_constraintScale ? std::abs(bound) : m_constraintScale;
}

double ConstraintNormalisation::normalise(BoundSide side, double bound, double value) const noexcept
{
    return side == BoundSide::Upper ? (value - bound) / scale(bound) : (bound - value) / scale(bound);
}

double ConstraintNormalisation::derivative(BoundSide side, double bound) const noexcept
{
    return (side == BoundSide::Upper ? 1.0 : -1.0) / scale(bound);
}

double ConstraintNormalisation::value(BoundSide side, double bound, double normalised) const noexcept
{
    return side == BoundSide::Upper ? bound + normalised * scale(bound) : bound - normalised * scale(bound);
}

} // namespace oos::design
