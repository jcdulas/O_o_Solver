#pragma once

#include "oos/design/optimiser/OptimiserDefaults.hpp"

#include <cstdint>

namespace oos::design
{

// The side of a bound.
enum class BoundSide : std::uint8_t
{
    Lower,
    Upper
};

// The normalised constraint (DCONSTR bounds; DesignOptimParams GSCAL):
//
//     lower bound L:  g = (L − r) / GNORM,      upper bound U:  g = (r − U) / GNORM,
//     GNORM = |bound| when |bound| > GSCAL, GSCAL otherwise,
//
// so that g ≤ 0 is satisfied, g = r/U − 1 and g = 1 − r/L for positive bounds away from zero, and a bound near zero
// (dividing by it would blow the value up) is measured against GSCAL instead of itself. The
// same form normalises the bounds of the dependent design variables and of the designed properties in the
// approximate problem.
//
// Thread-safety: immutable.
class ConstraintNormalisation
{
public:
    // The default of DesignOptimParams GSCAL, O_o_Solver's choice: the smallest scale factor of ADS, SCLMIN = 0.001
    // (NASA CR-177985, Table 6), a floor on a scale that would otherwise vanish (OptimiserDefaults.hpp).
    static constexpr double DEFAULT_CONSTRAINT_SCALE = optimiser_defaults::MINIMUM_SCALE;

    explicit ConstraintNormalisation(double constraintScale = DEFAULT_CONSTRAINT_SCALE) noexcept
        : m_constraintScale(constraintScale)
    {
    }

    [[nodiscard]] double constraintScale() const noexcept { return m_constraintScale; }
    // GNORM of a bound.
    [[nodiscard]] double scale(double bound) const noexcept;
    // g of a value r against a bound.
    [[nodiscard]] double normalise(BoundSide side, double bound, double value) const noexcept;
    // dg/dr: 1/GNORM for an upper bound, −1/GNORM for a lower one.
    [[nodiscard]] double derivative(BoundSide side, double bound) const noexcept;
    // The value r whose normalised constraint is g.
    [[nodiscard]] double value(BoundSide side, double bound, double normalised) const noexcept;

private:
    double m_constraintScale;
};

} // namespace oos::design
