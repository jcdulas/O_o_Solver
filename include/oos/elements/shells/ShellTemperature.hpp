#pragma once

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The temperature change of the shell of batch row `row` (the thermal strain α (T − T0) in linear statics):
// - T: the PlateTemperatureLoad of the element (its mean TBAR and gradient TPRIME, uniform) when the load set has one,
// else the
//   temperatures of its grid points (TEMP, or the DefaultTemperatureLoad default), at the corners, with no gradient;
// - T0: the same from the TEMPERATURE(INITIAL) set when the subcase has one (a grid point it leaves out takes TREF),
//   else the reference temperature of the section (TREF of MID1, of the CompProp for a laminate).
// The change at the centre (`deltaT`) is the mean of the corner changes (the value of the bilinear or linear
// interpolation there).
// A corner grid point without a load temperature (no DefaultTemperatureLoad, no TEMP) is a problem on the field
// TEMPERATURE(LOAD). Without a temperature load (ElementTemperatures::hasTemperatureLoad false: a DEFORM alone) the
// change is zero.
[[nodiscard]] Resolution<ShellTemperatureChange> resolveShellTemperature(const ElementBatchView& view, std::int64_t row,
                                                                         const ElementTemperatures& temperatures,
                                                                         std::span<const std::int32_t> cornerRows,
                                                                         const ShellSection& section);

} // namespace oos::elements
