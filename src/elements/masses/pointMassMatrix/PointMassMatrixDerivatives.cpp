#include "oos/elements/masses/pointMassMatrix/PointMassMatrixElement.hpp"

namespace oos::elements
{

// The Derivatives facet of PointMassMatrix: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse PointMassMatrixElement::designFieldUse(const DesignedField& field) const
{
    // PointMassMatrix: the terms of the 6 x 6 mass matrix change the mass only.
    return findFieldUse(field, {{"PointMassMatrix", "massMatrix", FieldUse::massOnly()}});
}

} // namespace oos::elements
