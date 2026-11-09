#include "oos/elements/springs/springDofInline/SpringDofInlineElement.hpp"

namespace oos::elements
{

// The Derivatives facet of SpringDofInline: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SpringDofInlineElement::designFieldUse(const DesignedField& field) const
{
    // SpringDofInline: K is the spring rate (the stiffness and the spring force), S the stress coefficient (the recovery
    // only), GE a damping (no static effect).
    return findFieldUse(field, {{"SpringDofInline", "stiffness", FieldUse::stiffnessAndRecovery()},
                                {"SpringDofInline", "stressCoefficient", FieldUse::recoveryOnly()}});
}

} // namespace oos::elements
