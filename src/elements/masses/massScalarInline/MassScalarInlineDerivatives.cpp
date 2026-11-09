#include "oos/elements/masses/massScalarInline/MassScalarInlineElement.hpp"

namespace oos::elements
{

// The Derivatives facet of MassScalarInline: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse MassScalarInlineElement::designFieldUse(const DesignedField& field) const
{
    // MassScalarInline: M is a scalar mass; it changes the mass only.
    return findFieldUse(field, {{"MassScalarInline", "mass", FieldUse::massOnly()}});
}

} // namespace oos::elements
