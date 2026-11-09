#include "oos/elements/masses/massDofInline/MassDofInlineElement.hpp"

namespace oos::elements
{

// The Derivatives facet of MassDofInline: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse MassDofInlineElement::designFieldUse(const DesignedField& field) const
{
    // MassDofInline: M is a scalar mass; it changes the mass only.
    return findFieldUse(field, {{"MassDofInline", "mass", FieldUse::massOnly()}});
}

} // namespace oos::elements
