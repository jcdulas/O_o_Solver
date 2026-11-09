#pragma once

#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/base/math/Vec3.hpp"

#include <array>

namespace oos::aero
{

// The resultant of a system of forces of the modelled configuration (boxes or structure), in the aerodynamic frame: the
// force, and the moment about the origin of the reference frame RCSID.
struct AeroResultant
{
    base::Vec3 force;
    base::Vec3 moment;
};

// The six non-dimensional coefficients of a SOL 144 trim (the stability derivatives, the monitor point totals),
// in the reference frame RCSID: CX, CY, CZ = F / (q·REFS) and CMX, CMY, CMZ = M / (q·REFS·REFB), M / (q·REFS·REFC),
// M / (q·REFS·REFB), the moment about the RCSID origin, for a resultant given per unit q.
//
// A half model (AEROS SYMXZ or SYMXY) gives half of the forces and moments of the whole configuration, the modelled
// part and its images: the image in a plane of symmetry key s is the mirrored system times s, whose resultant about
// the origin o is s·(mirror(F), mirror*(M about mirror(o))), mirror* the reflection of a moment (an axial vector:
// −mirror). The components the symmetry doubles are then those of the modelled part, those it cancels are zero (for
// SYMXZ = +1, CY, CMX, CMZ = 0).
//
// Thread-safety: stateless.
class ReferenceCoefficients
{
public:
    [[nodiscard]] static std::array<double, 6> of(const AeroReference& reference, const AeroResultant& resultant);
};

} // namespace oos::aero
