#include "oos/elements/bars_beams/BeamFieldDerivatives.hpp"
#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"

namespace oos::elements
{

// Fixed-geometry sizing rates use the shared beam flexibility, mass and recovery derivatives. Unsupported
// fields return false so the design layer can retain its numerical fallback.
FieldUse Beam2Element::designFieldUse(const DesignedField& field) const
{
    if (field.card == "IsoMat" && (field.column == "youngsModulus" || field.column == "shearModulus"))
    {
        auto use = commonFieldUse(field);
        use.changesDistributedLoads = true;
        return use;
    }
    // BeamProp: the section properties of the stations enter every computation; NSM the mass only; CW(A), CW(B) none
    // (no Beam2 has the warping DOFs SA, SB: Beam2Element.hpp).
    if (field.card == "BeamProp" && field.column == "stationSection" && field.item >= 0 && field.item % 6 == 5)
        return FieldUse::massOnly();
    return findFieldUse(field, {{"BeamProp", "warpingCoefficient", FieldUse::none()},
                                {"BeamProp", "stationStress", FieldUse::recoveryOnly()},
                                {"BeamProp", "nsmCenter", FieldUse::massOnly()}});
}

bool Beam2Element::loadFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, const ElementLoadContext& loads,
                                        ElementVectorBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::loads(resolve(view, columns, row).required(), view, row, field, nullptr, loads, block);
}

bool Beam2Element::loadFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                             const DesignedField& first, const DesignedField& second,
                                             const ElementLoadContext& loads, ElementVectorBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::loads(resolve(view, columns, row).required(), view, row, first, &second, loads, block);
}

bool Beam2Element::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                          const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesStiffness) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::stiffness(resolve(view, columns, row).required(), view, row, field, nullptr, block);
}

bool Beam2Element::stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesStiffness) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::stiffness(resolve(view, columns, row).required(), view, row, field, &field, block);
}

bool Beam2Element::stiffnessFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& first, const DesignedField& second,
                                               ElementMatrixBlock& block) const
{
    if (!designFieldUse(first).changesStiffness || !designFieldUse(second).changesStiffness) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::stiffness(resolve(view, columns, row).required(), view, row, first, &second, block);
}

bool Beam2Element::massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                     const MassOptions& options, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesMass) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::mass(resolve(view, columns, row).required(), view, row, field, nullptr,
                                      options, TorsionalInertia::Polar, block);
}

bool Beam2Element::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                           const MassOptions& options, ElementMatrixBlock& block) const
{
    if (!designFieldUse(field).changesMass) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::mass(resolve(view, columns, row).required(), view, row, field, &field,
                                      options, TorsionalInertia::Polar, block);
}

bool Beam2Element::massFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                          const DesignedField& first, const DesignedField& second,
                                          const MassOptions& options, ElementMatrixBlock& block) const
{
    if (!designFieldUse(first).changesMass || !designFieldUse(second).changesMass) return true;
    const Columns columns(view);
    return BeamFieldDerivatives::mass(resolve(view, columns, row).required(), view, row, first, &second,
                                      options, TorsionalInertia::Polar, block);
}

bool Beam2Element::recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                    ElementOutput output, const NodalDisplacements& displacements,
                                    ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::recovery(resolve(view, columns, row).required(), *this, view, row, field, nullptr,
                                          output, displacements, rates);
}

bool Beam2Element::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::recovery(resolve(view, columns, row).required(), *this, view, row, field, &field,
                                          output, displacements, rates);
}

bool Beam2Element::recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                         const DesignedField& first, const DesignedField& second, ElementOutput output,
                                         const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::recovery(resolve(view, columns, row).required(), *this, view, row, first, &second,
                                          output, displacements, rates);
}

bool Beam2Element::recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                    ElementOutput output, RecoverySlopes& rates) const
{
    requireOutput(output);
    const Columns columns(view);
    return BeamFieldDerivatives::slopeRates(resolve(view, columns, row).required(), *this,
                                            view, row, field, output, rates);
}

bool Beam2Element::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, const NodalDisplacements& preload,
                                                      ElementMatrixBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::differential(resolve(view, columns, row).required(), view, row, field, nullptr,
                                              preload, block);
}

bool Beam2Element::differentialStiffnessFieldMixedDerivative(
    const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    return BeamFieldDerivatives::differential(resolve(view, columns, row).required(), view, row, first, &second,
                                              preload, block);
}
} // namespace oos::elements
