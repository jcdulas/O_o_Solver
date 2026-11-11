#include "oos/elements/springs/bushing/BushingElement.hpp"

namespace oos::elements
{

namespace
{

enum class BushField
{
    Other,
    Stiffness,
    Mass,
    Recovery
};

BushField fieldOf(const DesignedField& field)
{
    if (field.card != "BushProp")
        return BushField::Other;
    if (field.column == "stiffness" && field.item >= 0 && field.item < 6)
        return BushField::Stiffness;
    if (field.column == "mass" && field.item == -1)
        return BushField::Mass;
    if (field.column == "recovery" && field.item >= 0 && field.item < 4)
        return BushField::Recovery;
    return BushField::Other;
}

std::span<const ElementDof> bushDofs(const BushingElement::Bush& bush, std::array<ElementDof, 12>& storage)
{
    for (std::size_t i = 0; i < 6; ++i)
    {
        storage[i] = ElementDof::grid(bush.rowA, static_cast<int>(i + 1));
        storage[6 + i] = ElementDof::grid(bush.rowB, static_cast<int>(i + 1));
    }
    return {storage.data(), bush.isGrounded ? 6u : 12u};
}

// Recovered rows are diag(c) H u. Return dc/dp for the selected property component.
std::array<double, 6> recoveryFactors(const BushingElement::Bush& bush, const DesignedField& field,
                                     ElementOutput output)
{
    std::array<double, 6> factors{};
    const BushField kind = fieldOf(field);
    if (kind == BushField::Stiffness)
    {
        const auto component = static_cast<std::size_t>(field.item);
        if (output == ElementOutput::Forces)
            factors[component] = 1.0;
        else if (output == ElementOutput::Stresses)
            factors[component] = bush.recovery[component < 3 ? 0 : 1];
    }
    else if (kind == BushField::Recovery)
        for (std::size_t component = 0; component < factors.size(); ++component)
        {
            const std::int32_t group = component < 3 ? 0 : 1;
            if (output == ElementOutput::Stresses && field.item == group)
                factors[component] = bush.stiffness[component];
            else if (output == ElementOutput::Strains && field.item == group + 2)
                factors[component] = 1.0;
        }
    return factors;
}

} // namespace

// Differentiate K = H^T diag(k) H and the recovered rows at fixed geometry.
FieldUse BushingElement::designFieldUse(const DesignedField& field) const
{
    // BushProp: K1-K6 are the stiffnesses (the stiffness and the forces), B1-B6 dampings and GE a structural
    // damping (no static effect), RCV the stress and strain recovery coefficients (the recovery only).
    return findFieldUse(field, {{"BushProp", "stiffness", FieldUse::stiffnessAndRecovery()},
                                {"BushProp", "mass", FieldUse::massOnly()},
                                {"BushProp", "damping", FieldUse::none()},
                                {"BushProp", "recovery", FieldUse::recoveryOnly()}});
}

bool BushingElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                             const DesignedField& field, ElementMatrixBlock& block) const
{
    if (fieldOf(field) != BushField::Stiffness)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 12> storage;
    const auto dofs = bushDofs(bush, storage);
    std::array<double, 144> matrix{};
    const auto component = static_cast<std::size_t>(field.item);
    for (std::size_t i = 0; i < dofs.size(); ++i)
        for (std::size_t j = 0; j < dofs.size(); ++j)
            matrix[i * dofs.size() + j] = bush.relativeMotion(component, i) * bush.relativeMotion(component, j);
    block.append(row, dofs, std::span<const double>(matrix.data(), dofs.size() * dofs.size()));
    return true;
}

bool BushingElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, const MassOptions& /*options*/,
                                        ElementMatrixBlock& block) const
{
    if (fieldOf(field) != BushField::Mass)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 12> storage;
    const auto dofs = bushDofs(bush, storage);
    std::array<double, 144> matrix{};
    const double share = bush.isGrounded ? 1.0 : 0.5;
    for (std::size_t i = 0; i < dofs.size(); ++i)
        if (i % 6 < 3)
            matrix[i * dofs.size() + i] = share;
    block.append(row, dofs, std::span<const double>(matrix.data(), dofs.size() * dofs.size()));
    return true;
}

bool BushingElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                       const DesignedField& field, ElementOutput output,
                                       const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    if (fieldOf(field) == BushField::Other)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 12> storage;
    const auto dofs = bushDofs(bush, storage);
    const auto factors = recoveryFactors(bush, field, output);
    std::array<double, 6> values{};
    for (std::size_t component = 0; component < values.size(); ++component)
        for (std::size_t i = 0; i < dofs.size(); ++i)
            values[component] +=
                factors[component] * bush.relativeMotion(component, i) * displacements.valueAt(dofs[i]);
    rates.reset(outputColumns(output));
    rates.append(row, view.elementId(row), values);
    return true;
}

bool BushingElement::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                   const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    return fieldOf(field) == BushField::Stiffness;
}

bool BushingElement::massFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                              const DesignedField& field, const MassOptions& /*options*/,
                                              ElementMatrixBlock& /*block*/) const
{
    return fieldOf(field) == BushField::Mass;
}

bool BushingElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
                                       const DesignedField& field, ElementOutput output, RecoverySlopes& rates) const
{
    requireOutput(output);
    if (fieldOf(field) == BushField::Other)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 12> storage;
    const auto dofs = bushDofs(bush, storage);
    const auto factors = recoveryFactors(bush, field, output);
    rates.reset(dofs, factors.size());
    const std::size_t resultRow = rates.appendRow({});
    for (std::size_t component = 0; component < factors.size(); ++component)
        for (std::size_t i = 0; i < dofs.size(); ++i)
            rates.slopes(resultRow, component)[i] = factors[component] * bush.relativeMotion(component, i);
    return true;
}

bool BushingElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                             const DesignedField& field, ElementOutput output,
                                             const NodalDisplacements& /*displacements*/,
                                             ElementResultBlock& rates) const
{
    requireOutput(output);
    if (fieldOf(field) == BushField::Other)
        return false;
    // Each recovered component is affine in one property component at fixed displacement.
    rates.reset(outputColumns(output));
    const std::array<double, 6> zeros{};
    rates.append(row, view.elementId(row), zeros);
    return true;
}

bool BushingElement::stiffnessFieldMixedDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                  const DesignedField& first, const DesignedField& second,
                                                  ElementMatrixBlock& /*block*/) const
{
    return fieldOf(first) != BushField::Other && fieldOf(second) != BushField::Other;
}

bool BushingElement::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                            const DesignedField& first, const DesignedField& second,
                                            ElementOutput output, const NodalDisplacements& displacements,
                                            ElementResultBlock& rates) const
{
    requireOutput(output);
    const BushField a = fieldOf(first), b = fieldOf(second);
    if (a == BushField::Other || b == BushField::Other)
        return false;
    std::array<double, 6> values{};
    if (output == ElementOutput::Stresses &&
        ((a == BushField::Stiffness && b == BushField::Recovery) ||
         (b == BushField::Stiffness && a == BushField::Recovery)))
    {
        const auto component = static_cast<std::size_t>(a == BushField::Stiffness ? first.item : second.item);
        const std::int32_t recovery = a == BushField::Recovery ? first.item : second.item;
        if (recovery == (component < 3 ? 0 : 1))
        {
            const Columns columns(view);
            const Bush bush = resolve(view, columns, row).required();
            std::array<ElementDof, 12> storage;
            const auto dofs = bushDofs(bush, storage);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                values[component] += bush.relativeMotion(component, i) * displacements.valueAt(dofs[i]);
        }
    }
    rates.reset(outputColumns(output));
    rates.append(row, view.elementId(row), values);
    return true;
}

} // namespace oos::elements
