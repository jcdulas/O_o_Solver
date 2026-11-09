#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"

namespace oos::elements
{

// The Derivatives facet of ShearPanel4: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse ShearPanel4Element::designFieldUse(const DesignedField& field) const
{
    // ShearPanelProp: T enters every computation; F1 and F2, the effectiveness of the extensional rods, the stiffness
    // and the recovery; NSM the mass only.
    return findFieldUse(field, {{"ShearPanelProp", "extensionalFactor1", FieldUse::stiffnessAndRecovery()},
                                {"ShearPanelProp", "extensionalFactor2", FieldUse::stiffnessAndRecovery()}});
}

} // namespace oos::elements
