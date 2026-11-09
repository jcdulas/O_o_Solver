#pragma once

#include <cstddef>
#include <limits>

namespace oos::linalg
{

// The roots asked of the eigenvalue solver, in eigenvalue units (λ = ω², the solution sequence converts the EIGRL
// frequencies V1, V2 in cycles per unit time: λ = (2π·f)²). The combinations of the EIGRL fields V1, V2, ND:
//
//   lowerBound, upperBound, modeCount = 0   every root in [lowerBound, upperBound]
//   upperBound = +∞, modeCount > 0         the lowest modeCount roots above lowerBound
//   upperBound finite, modeCount > 0       the lowest modeCount roots of [lowerBound, upperBound]
//
// A lower bound at or below 0 means "from the lowest root": the pencil of a vibration problem (K and M positive
// semi-definite) has no root below 0, and the rigid-body roots of a free structure are zeros rounded to either sign,
// which must all be found. A stiffness with negative roots (negative springs) then has them reported as well.
//
// For a buckling pencil (SpectralTransform::Buckling) the roots are load factors of either sign and the request reads
// (EIGRL V1, V2, ND in load-factor units): every root in
// [lowerBound, upperBound], or, with modeCount > 0, the modeCount roots of smallest magnitude there (an infinite bound
// leaves that side open: the default request with ND alone gives the ND roots nearest 0, of both signs). A request
// without a mode count needs both bounds finite.
struct EigenRequest
{
    double lowerBound = -std::numeric_limits<double>::infinity();
    double upperBound = std::numeric_limits<double>::infinity();
    // 0: every root of the range (upperBound must then be finite).
    std::size_t modeCount = 0;

    // True when the roots are sought from the lowest one (lowerBound <= 0).
    [[nodiscard]] bool startsFromLowest() const noexcept { return !(lowerBound > 0.0); }
    [[nodiscard]] bool hasUpperBound() const noexcept { return upperBound < std::numeric_limits<double>::infinity(); }
};

} // namespace oos::linalg
