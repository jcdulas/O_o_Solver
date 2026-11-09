#include "oos/elements/masses/massDofProperty/MassDofPropertyElement.hpp"

namespace oos::elements
{

// The Derivatives facet of MassDofProperty: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse MassDofPropertyElement::designFieldUse(const DesignedField& field) const
{
    // MassProp: M is a scalar mass; it changes the mass only.
    return findFieldUse(field, {{"MassProp", "mass", FieldUse::massOnly()}});
}

} // namespace oos::elements
