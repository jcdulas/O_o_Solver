#include "oos/elements/springs/bushing1D/Bushing1DElement.hpp"

#include <array>
#include <vector>

namespace oos::elements
{
namespace
{

enum class BushField
{
    Other,
    Stiffness,
    Mass,
    StressFactor,
    StrainFactor
};

BushField fieldOf(const DesignedField& field)
{
    if (field.card != "BushProp1D" || field.item != -1)
        return BushField::Other;
    if (field.column == "stiffness")
        return BushField::Stiffness;
    if (field.column == "mass")
        return BushField::Mass;
    if (field.column == "stressFactor")
        return BushField::StressFactor;
    if (field.column == "strainFactor")
        return BushField::StrainFactor;
    return BushField::Other;
}

std::array<double, 7> recoveryFactors(const Bushing1DElement::Bush& bush, BushField field)
{
    switch (field)
    {
    case BushField::Stiffness:
        return {1.0, 0.0, 0.0, bush.stressCoefficient, 0.0, 0.0, 0.0};
    case BushField::StressFactor:
        return {0.0, 0.0, 0.0, bush.stiffness, 0.0, 0.0, 0.0};
    case BushField::StrainFactor:
        return {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    default:
        return {};
    }
}

} // namespace

// These field uses select the matrices and recovered rows affected by each property.
// The closed-form rates below avoid perturbing the property to compute those changes.
FieldUse Bushing1DElement::designFieldUse(const DesignedField& field) const
{
    // BushProp1D: K is the axial stiffness (the stiffness and the force), C a damping (no static effect), M the
    // mass, SA and SE the stress and strain recovery coefficients (the recovery only).
    return findFieldUse(field, {{"BushProp1D", "stiffness", FieldUse::stiffnessAndRecovery()},
                                {"BushProp1D", "damping", FieldUse::none()},
                                {"BushProp1D", "mass", FieldUse::massOnly()},
                                {"BushProp1D", "stressFactor", FieldUse::recoveryOnly()},
                                {"BushProp1D", "strainFactor", FieldUse::recoveryOnly()}});
}

bool Bushing1DElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, ElementMatrixBlock& block) const
{
    if (fieldOf(field) != BushField::Stiffness)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 6> storage;
    const auto dofs = translationDofs(bush, storage);
    std::vector<double> matrix(dofs.size() * dofs.size());
    // K = k h^T h, hence dK/dk = h^T h.
    for (std::size_t i = 0; i < dofs.size(); ++i)
        for (std::size_t j = 0; j < dofs.size(); ++j)
            matrix[i * dofs.size() + j] = bush.axialMap[i] * bush.axialMap[j];
    block.append(row, dofs, matrix);
    return true;
}

bool Bushing1DElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                            const DesignedField& field, const MassOptions& /*options*/,
                                            ElementMatrixBlock& block) const
{
    if (fieldOf(field) != BushField::Mass)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 6> storage;
    const auto dofs = translationDofs(bush, storage);
    std::vector<double> matrix(dofs.size() * dofs.size(), 0.0);
    // The element mass is m I when grounded, otherwise m I/2 on each end.
    const double share = bush.isGrounded ? 1.0 : 0.5;
    for (std::size_t i = 0; i < dofs.size(); ++i)
        matrix[i * dofs.size() + i] = share;
    block.append(row, dofs, matrix);
    return true;
}

bool Bushing1DElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                           const DesignedField& field, ElementOutput output,
                                           const NodalDisplacements& displacements,
                                           ElementResultBlock& rates) const
{
    requireOutput(output);
    const BushField kind = fieldOf(field);
    if (kind == BushField::Other)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 6> storage;
    const auto dofs = translationDofs(bush, storage);
    double axial = 0.0;
    for (std::size_t i = 0; i < dofs.size(); ++i)
        axial += bush.axialMap[i] * displacements.valueAt(dofs[i]);
    const auto factors = recoveryFactors(bush, kind);
    std::array<double, 7> values{};
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = factors[i] * axial;
    rates.reset(outputColumns(output));
    rates.append(row, view.elementId(row), values);
    return true;
}

bool Bushing1DElement::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                       const DesignedField& field,
                                                       ElementMatrixBlock& /*block*/) const
{
    return fieldOf(field) == BushField::Stiffness;
}

bool Bushing1DElement::massFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                  const DesignedField& field, const MassOptions& /*options*/,
                                                  ElementMatrixBlock& /*block*/) const
{
    return fieldOf(field) == BushField::Mass;
}

bool Bushing1DElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
                                           const DesignedField& field, ElementOutput output,
                                           RecoverySlopes& rates) const
{
    requireOutput(output);
    const BushField kind = fieldOf(field);
    if (kind == BushField::Other)
        return false;
    const Columns columns(view);
    const Bush bush = resolve(view, columns, row).required();
    std::array<ElementDof, 6> storage;
    const auto dofs = translationDofs(bush, storage);
    rates.reset(dofs, 7);
    const std::size_t resultRow = rates.appendRow({});
    const auto factors = recoveryFactors(bush, kind);
    for (std::size_t column = 0; column < factors.size(); ++column)
        for (std::size_t dof = 0; dof < dofs.size(); ++dof)
            rates.slopes(resultRow, column)[dof] = factors[column] * bush.axialMap[dof];
    return true;
}

bool Bushing1DElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, ElementOutput output,
                                                 const NodalDisplacements& /*displacements*/,
                                                 ElementResultBlock& rates) const
{
    requireOutput(output);
    const BushField kind = fieldOf(field);
    if (kind == BushField::Other)
        return false;
    // At fixed displacement every recovered field is affine in each parameter separately.
    rates.reset(outputColumns(output));
    const std::array<double, 7> zeros{};
    rates.append(row, view.elementId(row), zeros);
    return true;
}

bool Bushing1DElement::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& first, const DesignedField& second,
                                               ElementOutput output, const NodalDisplacements& displacements,
                                               ElementResultBlock& rates) const
{
    requireOutput(output);
    const BushField a = fieldOf(first), b = fieldOf(second);
    if (a == BushField::Other || b == BushField::Other)
        return false;
    std::array<double, 7> values{};
    if ((a == BushField::Stiffness && b == BushField::StressFactor) ||
        (b == BushField::Stiffness && a == BushField::StressFactor))
    {
        const Columns columns(view);
        const Bush bush = resolve(view, columns, row).required();
        std::array<ElementDof, 6> storage;
        const auto dofs = translationDofs(bush, storage);
        for (std::size_t i = 0; i < dofs.size(); ++i)
            values[3] += bush.axialMap[i] * displacements.valueAt(dofs[i]);
    }
    rates.reset(outputColumns(output));
    rates.append(row, view.elementId(row), values);
    return true;
}

} // namespace oos::elements
