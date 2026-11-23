#include "BeamMassIntegration.hpp"

namespace oos::elements
{
base::MatN<12> beamMass(const BeamSection& section, double density, double addedMassPerLength, double length,
                        MassFormulation formulation, TorsionalInertia torsionalInertia)
{
    base::MatN<12> mass;
    beam_mass::integrate(section, length,
                         [&](double xi, double weight)
                         {
                             const auto s = section.at(xi);
                             beam_mass::addStation(mass, xi, length, weight, formulation, torsionalInertia, density,
                                                   s.area, s.nonStructuralMass + addedMassPerLength, s.inertia1,
                                                   s.inertia2, s.productOfInertia, s.neutralAxis, s.massCentre);
                         });
    return mass;
}
} // namespace oos::elements
