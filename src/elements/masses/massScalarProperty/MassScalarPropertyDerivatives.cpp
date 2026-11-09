#include "oos/elements/masses/massScalarProperty/MassScalarPropertyElement.hpp"

namespace oos::elements
{

// The Derivatives facet of MassScalarProperty: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse MassScalarPropertyElement::designFieldUse(const DesignedField& field) const
{
    // MassProp: M is a scalar mass; it changes the mass only.
    return findFieldUse(field, {{"MassProp", "mass", FieldUse::massOnly()}});
}

} // namespace oos::elements
