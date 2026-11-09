#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

namespace oos::elements
{

// The Derivatives facet of ShellTria6: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse ShellTria6Element::designFieldUse(const DesignedField& field) const
{
    // ShellProp: T, 12I/T**3 and TS/T enter the section (every computation); Z1 and Z2 are the fibre distances of
    // the stress recovery only; NSM the mass only. The element's ZOFFS, THETA or MCID and corner thicknesses enter
    // every computation. A pressure reads the element's geometry and offset only, never its property.
    // A AnisoMat2D stiffness term: the stiffness, the thermal loads and the recovery, not the mass (ρ and t only).
    constexpr FieldUse ANISO_MAT2D_TERM_USE{true, false, true, true, false};
    const FieldUse use = findFieldUse(field, {{"ShellProp", "fiberDistance1", FieldUse::recoveryOnly()},
                                              {"ShellProp", "fiberDistance2", FieldUse::recoveryOnly()},
                                              {"AnisoMat2D", "stiffness", ANISO_MAT2D_TERM_USE}});
    return field.card == card() ? use : use.withoutDistributedLoads();
}

bool ShellTria6Element::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, ElementMatrixBlock& block) const
{
    return higherOrderShellStiffnessRate<QuadraticTriangleShape>(view, row, field, block);
}

bool ShellTria6Element::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                            const DesignedField& field,
                                                            const NodalDisplacements& preload,
                                                            ElementMatrixBlock& block) const
{
    return higherOrderShellDifferentialRate<QuadraticTriangleShape>(view, row, field, preload, block);
}

bool ShellTria6Element::recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates) const
{
    requireOutput(output);
    return higherOrderShellRecoveryRates<QuadraticTriangleShape>(view, row, field, output, displacements, rates);
}

bool ShellTria6Element::layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& field, LayerOutput output,
                                               const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    return higherOrderShellLayerRates<QuadraticTriangleShape>(view, row, field, output, displacements, rates);
}

} // namespace oos::elements
