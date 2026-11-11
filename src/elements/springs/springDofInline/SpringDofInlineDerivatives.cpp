#include "oos/elements/springs/springDofInline/SpringDofInlineElement.hpp"
#include "oos/elements/springs/SpringFieldDerivatives.hpp"

namespace oos::elements
{

// The Derivatives facet of SpringDofInline: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SpringDofInlineElement::designFieldUse(const DesignedField& field) const
{
    // SpringDofInline: K is the spring rate (the stiffness and the spring force), S the stress coefficient (the recovery
    // only), GE a damping (no static effect).
    return findFieldUse(field, {{"SpringDofInline", "stiffness", FieldUse::stiffnessAndRecovery()},
        {"SpringDofInline", "stressCoefficient", FieldUse::recoveryOnly()}});
}

bool SpringDofInlineElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementMatrixBlock& block) const
{
    const SpringField kind = springFieldOf(field, "SpringDofInline", true);
    if (kind != SpringField::Stiffness)
        return false;
    const Columns columns(view.columns());
    return appendSpringStiffnessRate(row, resolve(view, columns, row).required().connection, kind, block);
}

bool SpringDofInlineElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    const SpringField kind = springFieldOf(field, "SpringDofInline", true);
    if (kind == SpringField::Other)
        return false;
    const Columns columns(view.columns());
    const Spring& spring = resolve(view, columns, row).required();
    return appendSpringRecoveryRate(view, row, spring.connection, kind, output, displacements,
        spring.stiffness, spring.stressCoefficient, rates);
}

bool SpringDofInlineElement::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
        const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    // K = k s s^T is linear in the spring rate.
    return springFieldOf(field, "SpringDofInline", true) == SpringField::Stiffness;
}

bool SpringDofInlineElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        RecoverySlopes& rates) const
{
    requireOutput(output);
    const SpringField kind = springFieldOf(field, "SpringDofInline", true);
    if (kind == SpringField::Other)
        return false;
    const Columns columns(view.columns());
    const Spring& spring = resolve(view, columns, row).required();
    return appendSpringSlopeRate(spring.connection, kind, output, spring.stiffness,
        spring.stressCoefficient, rates);
}

bool SpringDofInlineElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        const NodalDisplacements& /*displacements*/,
        ElementResultBlock& rates) const
{
    requireOutput(output);
    // At fixed displacement, force is linear in k and stress is linear in each of k and S.
    return appendSpringRecoverySecondRate(view, row, springFieldOf(field, "SpringDofInline", true),
        output, rates);
}

bool SpringDofInlineElement::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& first, const DesignedField& second,
                                              ElementOutput output, const NodalDisplacements& displacements,
                                              ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view.columns());
    const Spring spring = resolve(view, columns, row).required();
    return appendSpringRecoveryMixedRate(view, row, spring.connection, springFieldOf(first, "SpringDofInline", true),
                                          springFieldOf(second, "SpringDofInline", true), output, displacements, rates);
}

} // namespace oos::elements
