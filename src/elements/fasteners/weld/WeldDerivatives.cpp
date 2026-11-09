#include "oos/elements/fasteners/weld/WeldElement.hpp"

namespace oos::elements
{

// The Derivatives facet of Weld: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse WeldElement::designFieldUse(const DesignedField& field) const
{
    // WeldProp: D sizes the connector beam (every computation); the material fields follow the common rules. The
    // thickness of the shells a side names (PSHELL T, the ply thicknesses of a CompProp or CompPropGlobal, whose total
    // the laminate summary gives) sets the beam length when the mean thickness (tA + tB)/2 exceeds |AB|
    // (weldBeamLength): the stiffness and the forces, not the mass (ρ·A·|AB|). Which welds read a property:
    // weldSideProperty.
    return findFieldUse(field, {{"ShellProp", "thickness", FieldUse::stiffnessAndRecovery()},
                                {"CompProp", "plyThickness", FieldUse::stiffnessAndRecovery()},
                                {"CompPropGlobal", "plyThickness", FieldUse::stiffnessAndRecovery()}});
}

} // namespace oos::elements
