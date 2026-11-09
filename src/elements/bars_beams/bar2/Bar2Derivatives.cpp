#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

namespace oos::elements
{

// The Derivatives facet of Bar2: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse Bar2Element::designFieldUse(const DesignedField& field) const
{
    // BarProp: A, I1, I2, I12 and J enter the stiffness, the mass (the coupled mass reads the inertias) and the
    // recovery; K1, K2 (the column "shearFactor") the shear flexibility, so the stiffness and the recovery; the stress
    // recovery points C1-F2 (the columns "stressPointC" to "stressPointF") the recovery only; NSM the mass only.
    return findFieldUse(field, {{"BarProp", "shearFactor", FieldUse::stiffnessAndRecovery()},
                                {"BarProp", "stressPointC", FieldUse::recoveryOnly()},
                                {"BarProp", "stressPointD", FieldUse::recoveryOnly()},
                                {"BarProp", "stressPointE", FieldUse::recoveryOnly()},
                                {"BarProp", "stressPointF", FieldUse::recoveryOnly()}});
}

} // namespace oos::elements
