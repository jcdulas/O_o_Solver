#include "oos/elements/springs/gap2/Gap2Element.hpp"

namespace oos::elements
{

// The Derivatives facet of Gap2: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse Gap2Element::designFieldUse(const DesignedField& field) const
{
    // GapProp: the real fields enter the linear stiffness of the gap at its initial opening (U0, KA, KB, KT) or
    // nothing a static analysis reads; every computation is kept for them all.
    return commonFieldUse(field);
}

} // namespace oos::elements
