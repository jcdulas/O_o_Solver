#include "oos/elements/masses/pointMassInertia/PointMassInertiaElement.hpp"

namespace oos::elements
{

// The Derivatives facet of PointMassInertia: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse PointMassInertiaElement::designFieldUse(const DesignedField& field) const
{
    // PointMassInertia: M, the offset X1-X3 and the inertias I11-I33 change the mass only.
    return findFieldUse(field, {{"PointMassInertia", "mass", FieldUse::massOnly()},
                                {"PointMassInertia", "offset", FieldUse::massOnly()},
                                {"PointMassInertia", "inertia", FieldUse::massOnly()}});
}

} // namespace oos::elements
