#include "oos/elements/fasteners/fastener/FastenerElement.hpp"

#include "oos/elements/core/MatrixKernels.hpp"

namespace oos::elements
{

namespace
{

// The six spring rates precede the mass in this local index. -1 means an unsupported field.
int componentOf(const DesignedField& field)
{
    if (field.card != "FastenerProp")
        return -1;
    if (field.column == "mass" && field.item == -1)
        return 6;
    if (field.item < 0 || field.item >= 3)
        return -1;
    if (field.column == "translationalStiffness")
        return field.item;
    if (field.column == "rotationalStiffness")
        return field.item + 3;
    return -1;
}

} // namespace

// The interpolation and spring geometry are fixed for sizing derivatives: differentiate before condensation.
FieldUse FastenerElement::designFieldUse(const DesignedField& field) const
{
    // FastenerProp: KT1-KT3 and KR1-KR3 are the stiffnesses (the stiffness and the forces), MASS the mass only, GE a
    // damping (no static effect); D, the diameter, takes no part in the computations.
    return findFieldUse(field, {{"FastenerProp", "translationalStiffness", FieldUse::stiffnessAndRecovery()},
                                {"FastenerProp", "rotationalStiffness", FieldUse::stiffnessAndRecovery()},
                                {"FastenerProp", "mass", FieldUse::massOnly()},
                                {"FastenerProp", "diameter", FieldUse::none()}});
}

bool FastenerElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, ElementMatrixBlock& block) const
{
    const int component = componentOf(field);
    if (component < 0 || component >= 6)
        return false;
    const Columns columns(view);
    const Fastener fastener = resolve(view, columns, row).required();
    std::array<double, 6> factors{};
    factors[static_cast<std::size_t>(component)] = 1.0;
    const base::MatN<12> rate = diagonalCongruence(factors, fastener.relativeMotion);
    block.append(row, fastener.connection.dofs(), fastener.connection.condense(rate));
    return true;
}

bool FastenerElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                         const DesignedField& field, const MassOptions& /*options*/,
                                         ElementMatrixBlock& block) const
{
    if (componentOf(field) != 6)
        return false;
    const Columns columns(view);
    const Fastener fastener = resolve(view, columns, row).required();
    base::MatN<12> rate;
    for (std::size_t i = 0; i < 3; ++i)
    {
        rate(i, i) = 0.5;
        rate(6 + i, 6 + i) = 0.5;
    }
    block.append(row, fastener.connection.dofs(), fastener.connection.condense(rate));
    return true;
}

bool FastenerElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, ElementOutput output,
                                        const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    const int component = componentOf(field);
    if (component < 0)
        return false;
    const Columns columns(view);
    const Fastener fastener = resolve(view, columns, row).required();
    std::array<double, 6> values{};
    if (component < 6)
    {
        const auto relative = fastener.relativeMotion * fastener.connection.pointMotions(displacements);
        const auto index = static_cast<std::size_t>(component);
        values[index] = relative[index];
    }
    rates.reset(outputColumns(output));
    rates.append(row, view.elementId(row), values);
    return true;
}

bool FastenerElement::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                    const DesignedField& field, ElementMatrixBlock& /*block*/) const
{
    const int component = componentOf(field);
    return component >= 0 && component < 6;
}

bool FastenerElement::massFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                               const DesignedField& field, const MassOptions& /*options*/,
                                               ElementMatrixBlock& /*block*/) const
{
    return componentOf(field) == 6;
}

bool FastenerElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, ElementOutput output, RecoverySlopes& rates) const
{
    requireOutput(output);
    const int component = componentOf(field);
    if (component < 0)
        return false;
    const Columns columns(view);
    const Fastener fastener = resolve(view, columns, row).required();
    const auto dofs = fastener.connection.dofs();
    rates.reset(dofs, 6);
    const std::size_t resultRow = rates.appendRow({});
    if (component >= 6)
        return true;
    const auto index = static_cast<std::size_t>(component);
    const std::size_t n = dofs.size();
    for (std::size_t dof = 0; dof < n; ++dof)
        for (std::size_t motion = 0; motion < 12; ++motion)
            rates.slopes(resultRow, index)[dof] +=
                fastener.relativeMotion(index, motion) * fastener.connection.map[motion * n + dof];
    return true;
}

bool FastenerElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, ElementOutput output,
                                              const NodalDisplacements& /*displacements*/,
                                              ElementResultBlock& rates) const
{
    requireOutput(output);
    if (componentOf(field) < 0)
        return false;
    rates.reset(outputColumns(output));
    const std::array<double, 6> zeros{};
    rates.append(row, view.elementId(row), zeros);
    return true;
}

bool FastenerElement::stiffnessFieldMixedDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                   const DesignedField& first, const DesignedField& second,
                                                   ElementMatrixBlock& /*block*/) const
{
    return componentOf(first) >= 0 && componentOf(second) >= 0;
}

bool FastenerElement::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                            const DesignedField& first, const DesignedField& second,
                                            ElementOutput output, const NodalDisplacements& displacements,
                                            ElementResultBlock& rates) const
{
    return componentOf(second) >= 0 && recoveryFieldSecondRates(view, row, first, output, displacements, rates);
}

} // namespace oos::elements
