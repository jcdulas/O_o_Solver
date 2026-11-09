#include "oos/elements/rods/tube2/Tube2Element.hpp"

namespace oos::elements
{

// The Derivatives facet of Tube2: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse Tube2Element::designFieldUse(const DesignedField& field) const
{
    // TubeProp: OD and T give the area and the torsion constant, so they enter every computation; NSM the mass
    // only.
    return commonFieldUse(field);
}

} // namespace oos::elements
