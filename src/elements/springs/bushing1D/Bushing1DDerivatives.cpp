#include "oos/elements/springs/bushing1D/Bushing1DElement.hpp"

namespace oos::elements
{

// The Derivatives facet of Bushing1D: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse Bushing1DElement::designFieldUse(const DesignedField& field) const
{
    // BushProp1D: K is the axial stiffness (the stiffness and the force), C a damping (no static effect), M the
    // mass, SA and SE the stress and strain recovery coefficients (the recovery only).
    return findFieldUse(field, {{"BushProp1D", "stiffness", FieldUse::stiffnessAndRecovery()},
                                {"BushProp1D", "damping", FieldUse::none()},
                                {"BushProp1D", "mass", FieldUse::massOnly()},
                                {"BushProp1D", "stressFactor", FieldUse::recoveryOnly()},
                                {"BushProp1D", "strainFactor", FieldUse::recoveryOnly()}});
}

} // namespace oos::elements
