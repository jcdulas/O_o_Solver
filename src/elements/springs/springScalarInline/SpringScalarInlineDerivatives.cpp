#include "oos/elements/springs/springScalarInline/SpringScalarInlineElement.hpp"
#include "oos/elements/springs/SpringFieldDerivatives.hpp"

namespace oos::elements
{

// The Derivatives facet of SpringScalarInline: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SpringScalarInlineElement::designFieldUse(const DesignedField& field) const
{
    // SpringScalarInline: K is the spring rate (the stiffness and the spring force).
    return findFieldUse(field, {{"SpringScalarInline", "stiffness", FieldUse::stiffnessAndRecovery()}});
}

bool SpringScalarInlineElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementMatrixBlock& block) const
{
    const SpringField kind = springFieldOf(field, "SpringScalarInline", false);
    if (kind != SpringField::Stiffness)
        return false;
    const Columns columns(view.columns());
    return appendSpringStiffnessRate(row, resolve(view, columns, row).required().connection, kind, block);
}

bool SpringScalarInlineElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    const SpringField kind = springFieldOf(field, "SpringScalarInline", false);
    if (kind == SpringField::Other)
        return false;
    const Columns columns(view.columns());
    const Spring& spring = resolve(view, columns, row).required();
    return appendSpringRecoveryRate(view, row, spring.connection, kind, output, displacements,
        spring.stiffness, 0.0, rates);
}

bool SpringScalarInlineElement::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
        const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    // K = k s s^T is linear in the spring rate.
    return springFieldOf(field, "SpringScalarInline", false) == SpringField::Stiffness;
}

bool SpringScalarInlineElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        RecoverySlopes& rates) const
{
    requireOutput(output);
    const SpringField kind = springFieldOf(field, "SpringScalarInline", false);
    if (kind == SpringField::Other)
        return false;
    const Columns columns(view.columns());
    const Spring& spring = resolve(view, columns, row).required();
    return appendSpringSlopeRate(spring.connection, kind, output, spring.stiffness,
        0.0, rates);
}

bool SpringScalarInlineElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        const NodalDisplacements& /*displacements*/,
        ElementResultBlock& rates) const
{
    requireOutput(output);
    // At fixed displacement, force is linear in k and stress is linear in each of k and S.
    return appendSpringRecoverySecondRate(view, row, springFieldOf(field, "SpringScalarInline", false),
        output, rates);
}

bool SpringScalarInlineElement::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& first, const DesignedField& second,
                                              ElementOutput output, const NodalDisplacements& displacements,
                                              ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view.columns());
    const Spring spring = resolve(view, columns, row).required();
    const SpringField a = springFieldOf(first, "SpringScalarInline", false);
    const SpringField b = springFieldOf(second, "SpringScalarInline", false);
    return appendSpringRecoveryMixedRate(view, row, spring.connection, a, b, output, displacements, rates);
}

} // namespace oos::elements
