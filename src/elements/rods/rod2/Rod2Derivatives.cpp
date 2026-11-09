#include "oos/elements/rods/rod2/Rod2Element.hpp"

namespace oos::elements
{

// The Derivatives facet of Rod2: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse Rod2Element::designFieldUse(const DesignedField& field) const
{
    // RodProp: A enters every computation; J the torsional stiffness and the torque, C the torsional stress only,
    // NSM the mass only.
    return findFieldUse(field, {{"RodProp", "torsionConstant", FieldUse::stiffnessAndRecovery()},
                                {"RodProp", "torsionStressCoefficient", FieldUse::recoveryOnly()}});
}

// The closed forms of A and J (appendRodStiffnessRate, appendRodMassRate); every other field is differenced.
bool Rod2Element::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, ElementMatrixBlock& block) const
{
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return appendRodStiffnessRate(row, isOwn && field.column == "area", isOwn && field.column == "torsionConstant",
                                  block, [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

bool Rod2Element::massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                         const MassOptions& options, ElementMatrixBlock& block) const
{
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return appendRodMassRate(row, isOwn && field.column == "area", isOwn && field.column == "torsionConstant", options,
                             block, [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

bool Rod2Element::recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                        ElementOutput output, const NodalDisplacements& displacements,
                                        ElementResultBlock& rates) const
{
    requireOutput(output);
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return appendRodRowRates(view, row, isOwn && field.column == "area", isOwn && field.column == "torsionConstant",
                             output, displacements, rates,
                             [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

// The second derivatives of A and J (hasRodSecondDerivative): zero matrices and second rates, the
// slopes of the rates those of the forces of a unit section.
bool Rod2Element::stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                    const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return hasRodSecondDerivative(row, isOwn && field.column == "area", isOwn && field.column == "torsionConstant",
                                  [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

bool Rod2Element::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& field, const MassOptions& /*options*/,
                                               ElementMatrixBlock& /*block*/) const
{
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return hasRodSecondDerivative(row, isOwn && field.column == "area", isOwn && field.column == "torsionConstant",
                                  [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

bool Rod2Element::recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                        ElementOutput output, RecoverySlopes& rates) const
{
    requireOutput(output);
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return appendRodSlopeRates(row, isOwn && field.column == "area", isOwn && field.column == "torsionConstant", output,
                               rates, [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

bool Rod2Element::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, ElementOutput output,
                                              const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    // The thermal force P_T ∝ A too: the rows stay linear in A under a temperature change.
    static_cast<void>(displacements);
    const bool isOwn = field.card == "RodProp";
    const Columns columns(view);
    return appendRodRowSecondRates(view, row, isOwn && field.column == "area",
                                   isOwn && field.column == "torsionConstant", output, rates,
                                   [&](std::int64_t r) { return resolve(view, columns, r).required(); });
}

} // namespace oos::elements
