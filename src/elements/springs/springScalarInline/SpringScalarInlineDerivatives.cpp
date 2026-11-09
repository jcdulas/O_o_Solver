#include "oos/elements/springs/springScalarInline/SpringScalarInlineElement.hpp"

namespace oos::elements
{

// The Derivatives facet of SpringScalarInline: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SpringScalarInlineElement::designFieldUse(const DesignedField& field) const
{
    // SpringScalarInline: K is the spring rate (the stiffness and the spring force).
    return findFieldUse(field, {{"SpringScalarInline", "stiffness", FieldUse::stiffnessAndRecovery()}});
}

} // namespace oos::elements
