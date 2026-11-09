#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"

namespace oos::elements
{

// The Derivatives facet of Beam2: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse Beam2Element::designFieldUse(const DesignedField& field) const
{
    // BeamProp: the section properties of the stations enter every computation; NSM the mass only; CW(A), CW(B) none
    // (no Beam2 has the warping DOFs SA, SB: Beam2Element.hpp).
    return findFieldUse(field, {{"BeamProp", "warpingCoefficient", FieldUse::none()}});
}

} // namespace oos::elements
