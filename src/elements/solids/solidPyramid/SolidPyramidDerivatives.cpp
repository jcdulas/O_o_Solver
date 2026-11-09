#include "oos/elements/solids/solidPyramid/SolidPyramidElement.hpp"

namespace oos::elements
{

// The Derivatives facet of SolidPyramid: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SolidPyramidElement::designFieldUse(const DesignedField& field) const
{
    // SolidProp has no real field; the material fields follow the common rules (commonFieldUse).
    return commonFieldUse(field);
}

} // namespace oos::elements
