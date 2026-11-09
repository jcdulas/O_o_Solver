#pragma once

// The rate of a laminate (PCOMP, CompPropGlobal; Laminate.hpp) with respect to the thickness of one ply of its card,
// the designed field ("CompProp" or "CompPropGlobal", propId, ("ply_thickness", k)): what the shells' closed-form
// derivatives in a ply thickness differentiate along.
//
// The card's ply k gives the stack's plies j whose cardPly is k (one, or two under LAM = SYM), dt_j = 1, so dT = their
// count. The stack's z follow (design::DerivedProperties writes them so at a design point): z_bottom of the first ply
// is Z0, −T/2 when blank (dz = −dT/2) and fixed when given; each next ply starts at the top of the one below. The
// classical lamination theory (Jones §4.2) differentiated:
//
//   dA = Σ Q̄_j dt_j,  dB = Σ Q̄_j (z_j dz_j − z_{j−1} dz_{j−1}),  dD = Σ Q̄_j (z_j² dz_j − z_{j−1}² dz_{j−1}),
//
// z_j the top of ply j. The transverse shear H of the equilibrium distribution (Laminate.cpp: setTransverseShear,
// rational in the z of every ply through the neutral axis B/A, D* = D − B²/A and the shear energy's three-point Gauss
// sums) and each ply's shear at its top per unit force are differentiated forward through the same operations
// (a value with its rate per quantity), exact. LAM = MEM leaves dB = dD = 0, BEND dA = dB = 0, SMEAR D = A T²/12 (dD =
// dA T²/12 + A T dT/6), B = 0 and a rigid transverse shear. The mass per area Σ ρ_j t_j moves by Σ ρ_j dt_j.
//
// Not given (the caller differences): a field of another card or PID, an item out of the card's plies. The thermal
// resultants are not differentiated: under a temperature the shells refuse the closed forms (as for the AnisoMat2D
// terms).

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/DesignFieldUse.hpp"
#include "oos/elements/shells/Laminate.hpp"

#include <array>
#include <optional>
#include <vector>

namespace oos::elements
{

// The derivatives of a Laminate's stiffness and stack along one card ply's thickness, in the material frame of the
// element (as the Laminate); the vectors per stack ply, bounded by the plies of one property.
struct LaminateRate
{
    double thickness = 0.0;                                // dT
    double massPerArea = 0.0;                              // d(Σ ρ t)
    base::Mat3 a;                                          // dA
    base::Mat3 b;                                          // dB
    base::Mat3 d;                                          // dD
    std::array<double, 2> transverseShear{};               // dH_x, dH_y
    std::vector<double> plyThickness;                      // dt_j: 1 for the card ply's stack plies, else 0
    std::vector<double> zBottom;                           // dz of each ply's bottom
    std::vector<double> zTop;                              // dz of each ply's top
    std::vector<std::array<double, 2>> shearAtTopPerForce; // d(LaminatePly::shearAtTopPerForce)
};

// Whether a designed field is the thickness of a ply of a CompProp or CompPropGlobal ("ply_thickness", the item the
// card's ply from 0) named by its PID.
[[nodiscard]] bool isPlyThickness(const DesignedField& field) noexcept;

// The rate of `laminate` along the ply thickness `field`, or nothing when the field is not one of this laminate's
// card plies (another card type or PID, an item out of range).
[[nodiscard]] std::optional<LaminateRate> laminateThicknessRate(const Laminate& laminate, const DesignedField& field);

} // namespace oos::elements
