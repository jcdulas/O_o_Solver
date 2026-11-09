#include "oos/elements/springs/springScalarProperty/SpringScalarPropertyElement.hpp"

namespace oos::elements
{

// The Derivatives facet of SpringScalarProperty: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SpringScalarPropertyElement::designFieldUse(const DesignedField& field) const
{
    // SpringProp: K is the spring rate (the stiffness and the spring force), S the stress coefficient (the recovery
    // only), GE a damping (no static effect).
    return findFieldUse(field, {{"SpringProp", "stiffness", FieldUse::stiffnessAndRecovery()},
                                {"SpringProp", "stressCoefficient", FieldUse::recoveryOnly()}});
}

} // namespace oos::elements
