#pragma once

#include <cmath>
#include <limits>

namespace oos::elements
{

// The value of an undefined margin of safety (no stress limit on the material, or a zero stress): the smallest
// denormal double, bits 0x1, the marker the result tables use for such a margin; a printed report shows it blank. The
// scale rule of the comparison reads it as zero.
inline constexpr double UNDEFINED_MARGIN = std::numeric_limits<double>::denorm_min();

// The margin of safety of a normal stress against the IsoMat limits (fields ST, SC, which serve only the margins of
// safety of some elements): MS = ST/σ − 1 in tension, |SC|/|σ| − 1 in compression; undefined when
// the limit of that sign is not given (0) or the stress is zero.
[[nodiscard]] inline double normalStressMargin(double stress, double tensionLimit, double compressionLimit)
{
    if (stress > 0.0 && tensionLimit > 0.0)
        return tensionLimit / stress - 1.0;
    if (stress < 0.0 && compressionLimit != 0.0)
        return std::abs(compressionLimit) / std::abs(stress) - 1.0;
    return UNDEFINED_MARGIN;
}

// The margin of safety of a shear stress against the IsoMat limit SS: MS = SS/|τ| − 1; undefined when SS is not given
// or the stress is zero.
[[nodiscard]] inline double shearStressMargin(double stress, double shearLimit)
{
    if (stress != 0.0 && shearLimit > 0.0)
        return shearLimit / std::abs(stress) - 1.0;
    return UNDEFINED_MARGIN;
}

} // namespace oos::elements
