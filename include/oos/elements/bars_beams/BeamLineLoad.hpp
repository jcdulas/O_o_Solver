#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/bars_beams/BeamGeometry.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <optional>
#include <span>
#include <variant>

namespace oos::elements
{

// A BeamDistributedLoad on a bar or beam resolved into the element system:
//
// - TYPE: FX, FY, FZ a force along the basic axes, FXE, FYE, FZE along the element axes; MX ... MZE a moment, same
//   axes. The direction is written here in the element system.
// - SCALE: LE (X1, X2 distances from end A, P per unit length), FR (X1, X2 fractions of the length L between end A and
//   end B, P per unit length), LEPR and FRPR (P per unit of the length projected on a plane normal to the load
//   direction: per unit actual length, P·|sin α| with α the angle between the element axis and the load direction).
//   The projected forms apply to the loads along basic axes only (refused with an element axis).
// - X2 = X1: a concentrated load P1 at X1; otherwise a load varying linearly from P1 at X1 to P2 at X2,
//   0 ≤ X1 < X2 ≤ L (or 1). Distances are measured along the element between the end points (offsets included).
struct BeamLineLoad
{
    bool isMoment = false;
    base::Vec3 direction;    // the unit direction in the element system
    double start = 0.0;      // X1 as a distance from end A
    double end = 0.0;        // X2 as a distance from end A (end = start for a concentrated load)
    double startValue = 0.0; // P1 per unit actual length (a force or moment for a concentrated load)
    double endValue = 0.0;   // P2 per unit actual length
    bool isConcentrated = false;
};

// The force and moment of loads on a part of an element, the moment about a point of the axis.
struct LoadResultant
{
    base::Vec3 force;
    base::Vec3 moment;
};

// A position within this fraction of the length beyond 0 or L (beyond 1 for FR) is taken as the end itself: X2 = L
// written with fewer digits than the element length is computed with.
inline constexpr double BEAM_LOAD_POSITION_TOLERANCE = 1.0e-6;

// The load of `load` on the element of `geometry` in the element system, or the diagnostic of a load type, scale type
// or position the input format does not allow (card BeamDistributedLoad, ID the SID, the field).
[[nodiscard]] std::variant<BeamLineLoad, base::Diagnostic> resolveLineLoad(const ElementLineLoad& load,
                                                                           const BeamGeometry& geometry,
                                                                           std::string_view elementCard,
                                                                           std::int32_t elementId);

// The resultant of `load` on the part [from, to] of the axis, the moment about the point at `about` (distances from
// end A, element system). A concentrated load at `from` or `to` counts when `includeEnds` (the right part [x, L] of a
// cantilever) and only within [from, to) otherwise (the left part [0, x) of a recovery).
[[nodiscard]] LoadResultant loadResultant(const BeamLineLoad& load, double from, double to, double about,
                                          bool includeEnds);

// The sum over several loads.
[[nodiscard]] LoadResultant loadResultant(std::span<const BeamLineLoad> loads, double from, double to, double about,
                                          bool includeEnds);

} // namespace oos::elements
