#include "oos/elements/bars_beams/BeamFieldDerivatives.hpp"
#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

namespace oos::elements
{

// Fixed-geometry sizing rates use the shared beam flexibility, mass and recovery derivatives. Unsupported
// fields return false so the design layer can retain its numerical fallback.
FieldUse Bar2Element::designFieldUse(const DesignedField& field) const
{
    if (field.card == "IsoMat" && (field.column == "youngsModulus" || field.column == "shearModulus"))
    {
        auto use = commonFieldUse(field);
        use.changesDistributedLoads = true;
        return use;
    }
    // BarProp: A, I1, I2, I12 and J enter the stiffness, the mass (the coupled mass reads the inertias) and the
    // recovery; K1, K2 (the column "shearFactor") the shear flexibility, so the stiffness and the recovery; the stress
    // recovery points C1-F2 (the columns "stressPointC" to "stressPointF") the recovery only; NSM the mass only.
    auto shearUse = FieldUse::stiffnessAndRecovery();
    shearUse.changesDistributedLoads = true; // Equivalent line loads depend on shear flexibility.
    return findFieldUse(field, {{"BarProp", "shearFactor", shearUse},
                                {"BarProp", "stressPointC", FieldUse::recoveryOnly()},
                                {"BarProp", "stressPointD", FieldUse::recoveryOnly()},
                                {"BarProp", "stressPointE", FieldUse::recoveryOnly()},
                                {"BarProp", "stressPointF", FieldUse::recoveryOnly()}});
}

bool Bar2Element::loadFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, const ElementLoadContext& loads,
                                        ElementVectorBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::loads(resolve(view, columns, row).required(), view, row, field, nullptr, loads, block);
}

bool Bar2Element::loadFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                             const DesignedField& first, const DesignedField& second,
                                             const ElementLoadContext& loads, ElementVectorBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::loads(resolve(view, columns, row).required(), view, row, first, &second, loads, block);
}

bool Bar2Element::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                          const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesStiffness) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::stiffness(resolve(view, columns, row).required(), view, row, field, nullptr, block);
}

bool Bar2Element::stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesStiffness) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::stiffness(resolve(view, columns, row).required(), view, row, field, &field, block);
}

bool Bar2Element::stiffnessFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& first, const DesignedField& second,
                                               ElementMatrixBlock& block) const
{
    if (!designFieldUse(first).changesStiffness || !designFieldUse(second).changesStiffness) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::stiffness(resolve(view, columns, row).required(), view, row, first, &second, block);
}

bool Bar2Element::massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                     const MassOptions& options, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesMass) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::mass(resolve(view, columns, row).required(), view, row, field, nullptr,
                                      options, TorsionalInertia::None, block);
}

bool Bar2Element::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                           const MassOptions& options, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesMass) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::mass(resolve(view, columns, row).required(), view, row, field, &field,
                                      options, TorsionalInertia::None, block);
}

bool Bar2Element::massFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                          const DesignedField& first, const DesignedField& second,
                                          const MassOptions& options, ElementMatrixBlock& block) const
{
    if (!designFieldUse(first).changesMass || !designFieldUse(second).changesMass) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::mass(resolve(view, columns, row).required(), view, row, first, &second,
                                      options, TorsionalInertia::None, block);
}

bool Bar2Element::recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                    ElementOutput output, const NodalDisplacements& displacements,
                                    ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::recovery(resolve(view, columns, row).required(), *this, view, row, field, nullptr,
                                          output, displacements, rates);
}

bool Bar2Element::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::recovery(resolve(view, columns, row).required(), *this, view, row, field, &field,
                                          output, displacements, rates);
}

bool Bar2Element::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                         const DesignedField& first, const DesignedField& second, ElementOutput output,
                                         const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::recovery(resolve(view, columns, row).required(), *this, view, row, first, &second,
                                          output, displacements, rates);
}

bool Bar2Element::recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                    ElementOutput output, RecoverySlopes& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::slopeRates(resolve(view, columns, row).required(), *this,
                                            view, row, field, output, rates);
}

bool Bar2Element::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, const NodalDisplacements& preload,
                                                      ElementMatrixBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::differential(resolve(view, columns, row).required(), view, row, field, nullptr,
                                              preload, block);
}

bool Bar2Element::differentialStiffnessFieldMixedDerivative(
    const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::differential(resolve(view, columns, row).required(), view, row, first, &second,
                                              preload, block);
}
} // namespace oos::elements
