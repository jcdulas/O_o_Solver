#include "oos/elements/springs/bushing/BushingElement.hpp"

namespace oos::elements
{

// The Derivatives facet of Bushing: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse BushingElement::designFieldUse(const DesignedField& field) const
{
    // BushProp: K1-K6 are the stiffnesses (the stiffness and the forces), B1-B6 dampings and GE a structural
    // damping (no static effect), RCV the stress and strain recovery coefficients (the recovery only).
    return findFieldUse(field, {{"BushProp", "stiffness", FieldUse::stiffnessAndRecovery()},
                                {"BushProp", "damping", FieldUse::none()},
                                {"BushProp", "recovery", FieldUse::recoveryOnly()}});
}

} // namespace oos::elements
