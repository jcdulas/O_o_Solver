#include "oos/elements/solids/solidPenta/SolidPentaElement.hpp"

namespace oos::elements
{

// The Derivatives facet of SolidPenta: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SolidPentaElement::designFieldUse(const DesignedField& field) const
{
    // SolidProp has no real field; the material fields follow the common rules (commonFieldUse).
    return commonFieldUse(field);
}

} // namespace oos::elements
