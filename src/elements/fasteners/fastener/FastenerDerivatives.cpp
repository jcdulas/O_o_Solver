#include "oos/elements/fasteners/fastener/FastenerElement.hpp"

namespace oos::elements
{

// The Derivatives facet of Fastener: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse FastenerElement::designFieldUse(const DesignedField& field) const
{
    // FastenerProp: KT1-KT3 and KR1-KR3 are the stiffnesses (the stiffness and the forces), MASS the mass only, GE a
    // damping (no static effect); D, the diameter, takes no part in the computations.
    return findFieldUse(field, {{"FastenerProp", "translationalStiffness", FieldUse::stiffnessAndRecovery()},
                                {"FastenerProp", "rotationalStiffness", FieldUse::stiffnessAndRecovery()},
                                {"FastenerProp", "mass", FieldUse::massOnly()},
                                {"FastenerProp", "diameter", FieldUse::none()}});
}

} // namespace oos::elements
