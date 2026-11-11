#include "oos/elements/springs/springDofProperty/SpringDofPropertyElement.hpp"
#include "oos/elements/springs/SpringFieldDerivatives.hpp"

namespace oos::elements
{

// The Derivatives facet of SpringDofProperty: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse SpringDofPropertyElement::designFieldUse(const DesignedField& field) const
{
    // SpringProp: K is the spring rate (the stiffness and the spring force), S the stress coefficient (the recovery
    // only), GE a damping (no static effect).
    return findFieldUse(field, {{"SpringProp", "stiffness", FieldUse::stiffnessAndRecovery()},
        {"SpringProp", "stressCoefficient", FieldUse::recoveryOnly()}});
}

bool SpringDofPropertyElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementMatrixBlock& block) const
{
    const SpringField kind = springFieldOf(field, "SpringProp", true);
    if (kind != SpringField::Stiffness)
        return false;
    const Columns columns(view);
    return appendSpringStiffnessRate(row, resolve(view, columns, row).required().connection, kind, block);
}

bool SpringDofPropertyElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    const SpringField kind = springFieldOf(field, "SpringProp", true);
    if (kind == SpringField::Other)
        return false;
    const Columns columns(view);
    const Spring& spring = resolve(view, columns, row).required();
    return appendSpringRecoveryRate(view, row, spring.connection, kind, output, displacements,
        spring.stiffness, spring.stressCoefficient, rates);
}

bool SpringDofPropertyElement::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
        const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    // K = k s s^T is linear in the spring rate.
    return springFieldOf(field, "SpringProp", true) == SpringField::Stiffness;
}

bool SpringDofPropertyElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        RecoverySlopes& rates) const
{
    requireOutput(output);
    const SpringField kind = springFieldOf(field, "SpringProp", true);
    if (kind == SpringField::Other)
        return false;
    const Columns columns(view);
    const Spring& spring = resolve(view, columns, row).required();
    return appendSpringSlopeRate(spring.connection, kind, output, spring.stiffness,
        spring.stressCoefficient, rates);
}

bool SpringDofPropertyElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
        const DesignedField& field, ElementOutput output,
        const NodalDisplacements& /*displacements*/,
        ElementResultBlock& rates) const
{
    requireOutput(output);
    // At fixed displacement, force is linear in k and stress is linear in each of k and S.
    return appendSpringRecoverySecondRate(view, row, springFieldOf(field, "SpringProp", true),
        output, rates);
}

bool SpringDofPropertyElement::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& first, const DesignedField& second,
                                              ElementOutput output, const NodalDisplacements& displacements,
                                              ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    const Spring spring = resolve(view, columns, row).required();
    return appendSpringRecoveryMixedRate(view, row, spring.connection, springFieldOf(first, "SpringProp", true),
                                          springFieldOf(second, "SpringProp", true), output, displacements, rates);
}

} // namespace oos::elements
