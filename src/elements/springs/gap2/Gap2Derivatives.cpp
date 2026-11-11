#include "oos/elements/springs/gap2/Gap2Element.hpp"

#include "oos/elements/core/MatrixKernels.hpp"

namespace oos::elements
{

namespace
{

bool isStiffnessField(const DesignedField& field)
{
    return field.card == "GapProp" && field.item == -1 &&
           (field.column == "closedStiffness" || field.column == "openStiffness" ||
            field.column == "transverseStiffness");
}

std::array<double, 3> stiffnessFactors(const Gap2Element::Gap& gap, const DesignedField& field)
{
    if (gap.isClosed && field.column == "closedStiffness")
        return {1.0, 0.0, 0.0};
    if (!gap.isClosed && field.column == "openStiffness")
        return {1.0, 0.0, 0.0};
    if (gap.isClosed && field.column == "transverseStiffness")
        return {0.0, 1.0, 1.0};
    return {};
}

std::array<ElementDof, 6> gapDofs(const Gap2Element::Gap& gap)
{
    return {ElementDof::grid(gap.rowA, 1), ElementDof::grid(gap.rowA, 2), ElementDof::grid(gap.rowA, 3),
            ElementDof::grid(gap.rowB, 1), ElementDof::grid(gap.rowB, 2), ElementDof::grid(gap.rowB, 3)};
}

} // namespace

// Differentiate the spring rates within the initial open or closed branch; the opening selects that branch.
FieldUse Gap2Element::designFieldUse(const DesignedField& field) const
{
    return findFieldUse(field, {{"GapProp", "closedStiffness", FieldUse::stiffnessAndRecovery()},
                                {"GapProp", "openStiffness", FieldUse::stiffnessAndRecovery()},
                                {"GapProp", "transverseStiffness", FieldUse::stiffnessAndRecovery()}});
}

bool Gap2Element::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                          const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!isStiffnessField(field))
        return false;
    const Columns columns(view);
    const Gap gap = resolve(view, columns, row).required();
    appendMatrix(block, row, gapDofs(gap), diagonalCongruence(stiffnessFactors(gap, field), gap.relativeMotion));
    return true;
}

bool Gap2Element::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                    const DesignedField& field, ElementOutput output,
                                    const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    if (!isStiffnessField(field))
        return false;
    const Columns columns(view);
    const Gap gap = resolve(view, columns, row).required();
    const auto dofs = gapDofs(gap);
    const auto factors = stiffnessFactors(gap, field);
    std::array<double, 8> values{};
    for (std::size_t component = 0; component < 3; ++component)
        for (std::size_t dof = 0; dof < dofs.size(); ++dof)
            values[component] += (component == 0 ? -1.0 : 1.0) * factors[component] *
                                 gap.relativeMotion(component, dof) * displacements.valueAt(dofs[dof]);
    rates.reset(outputColumns(output));
    rates.append(row, view.elementId(row), values);
    return true;
}

bool Gap2Element::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    return isStiffnessField(field);
}

bool Gap2Element::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
                                    const DesignedField& field, ElementOutput output, RecoverySlopes& rates) const
{
    requireOutput(output);
    if (!isStiffnessField(field))
        return false;
    const Columns columns(view);
    const Gap gap = resolve(view, columns, row).required();
    const auto dofs = gapDofs(gap);
    const auto factors = stiffnessFactors(gap, field);
    rates.reset(dofs, 8);
    const std::size_t resultRow = rates.appendRow({});
    for (std::size_t component = 0; component < 3; ++component)
        for (std::size_t dof = 0; dof < dofs.size(); ++dof)
            rates.slopes(resultRow, component)[dof] =
                (component == 0 ? -1.0 : 1.0) * factors[component] * gap.relativeMotion(component, dof);
    return true;
}

bool Gap2Element::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                         const DesignedField& field, ElementOutput output,
                                         const NodalDisplacements& /*displacements*/, ElementResultBlock& rates) const
{
    requireOutput(output);
    if (!isStiffnessField(field))
        return false;
    rates.reset(outputColumns(output));
    const std::array<double, 8> zeros{};
    rates.append(row, view.elementId(row), zeros);
    return true;
}

bool Gap2Element::stiffnessFieldMixedDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                               const DesignedField& first, const DesignedField& second,
                                               ElementMatrixBlock& /*block*/) const
{
    return isStiffnessField(first) && isStiffnessField(second);
}

bool Gap2Element::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                         const DesignedField& first, const DesignedField& second,
                                         ElementOutput output, const NodalDisplacements& displacements,
                                         ElementResultBlock& rates) const
{
    return isStiffnessField(second) && recoveryFieldSecondRates(view, row, first, output, displacements, rates);
}

} // namespace oos::elements
