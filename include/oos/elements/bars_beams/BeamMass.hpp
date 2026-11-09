#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamSection.hpp"
#include "oos/elements/core/ElementFamily.hpp"

namespace oos::elements
{

// Whether the mass of a bar or beam carries the torsional inertia of its section about its axis.
enum class TorsionalInertia
{
    None,  // Bar2: no rotary or torsional inertia
    Polar, // Beam2: ρ·(I1 + I2) per unit length on θx
};

// The mass of a bar or beam over its end DOFs [u_A θ_A u_B θ_B] in the element system, with the mass per unit length
// m(x) = ρ·A(x) + NSM(x) + the NSM cards' share:
//
// - lumped (PARAM COUPMASS −1, the default): translations, m_A = ∫ m·(1 − x/L) dx at end A and m_B = ∫ m·x/L dx
//   at end B in each direction (m·L/2 at each end for a uniform bar); with TorsionalInertia::Polar also the torsional
//   inertia ∫ ρ·(I1 + I2)·(1 − x/L) dx on θx at end A and ∫ ρ·(I1 + I2)·x/L dx at end B (ρ·(I1 + I2)·L/2 each for a
//   uniform beam);
// - coupled (COUPMASS > 0): the consistent mass ∫ m·Nᵀ·N dx of the axial motion (linear shape functions) and of the
//   bending motion in both planes (Hermite cubics on v, θz and on w, −θy); with TorsionalInertia::Polar also the
//   consistent torsional inertia ∫ ρ·(I1 + I2)·Nᵀ·N dx with linear shape functions on θx (ρ·(I1 + I2)·L/6·[2 1; 1 2]
//   for a uniform beam).
//
// Modelling choices: no rotary inertia of the section in bending in either form, the NSM carrying no torsional inertia
// (a BeamProp NSI is refused); the pin flags do not change the mass (the end point carries the mass whatever its
// released DOFs). The 4-point Gauss rule used is exact for m(x) and I1 + I2 linear between stations.
[[nodiscard]] base::MatN<12> beamMass(const BeamSection& section, double density, double addedMassPerLength,
                                      double length, MassFormulation formulation, TorsionalInertia torsionalInertia);

} // namespace oos::elements
