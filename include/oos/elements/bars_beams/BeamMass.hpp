#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamSection.hpp"
#include "oos/elements/core/ElementFamily.hpp"

namespace oos::elements
{

// Whether the mass of a bar or beam carries the torsional inertia of its section about its axis.
enum class TorsionalInertia
{
    None,  // Bar2: no centroidal torsional inertia
    Polar, // Beam2: ρ·(I1 + I2) per unit length on θx
};

// Section mass over the physical end DOFs. Integrate material at the neutral axis and additional mass at its
// centre. Both carry bending rotary inertia, and Beam2 also carries polar inertia. The grid operator applies releases and rigid offsets.
[[nodiscard]] base::MatN<12> beamMass(const BeamSection& section, double density, double addedMassPerLength,
                                      double length, MassFormulation formulation, TorsionalInertia torsionalInertia);

} // namespace oos::elements
