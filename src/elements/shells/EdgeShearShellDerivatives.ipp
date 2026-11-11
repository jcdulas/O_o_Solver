// The design derivatives of EdgeShearShellFamily (EdgeShearShellFamily.hpp): the closed forms by the complex step
// through the kernels, along the rates of the element's inputs (Direction). Included after EdgeShearShellRecovery.ipp.
//
// First derivatives use the complex-step chain rule (ShellScalar.hpp). Second derivatives in ShellProp T
// propagate the chain rule with SecondOrderScalar: no perturbed thickness or numerical step is needed.

#pragma once

#include "oos/elements/shells/SecondOrderScalar.hpp"

#include <limits>

namespace oos::elements
{

namespace edge_shear_detail
{

template <std::size_t R, std::size_t C>
ScalarMatrix<SecondOrderScalar, R, C> constantMatrix(const ScalarMatrix<double, R, C>& source)
{
    ScalarMatrix<SecondOrderScalar, R, C> result;
    for (std::size_t k = 0; k < source.values.size(); ++k)
        result.values[k] = source.values[k];
    return result;
}

// ShellProp thickness changes the geometric thicknesses; the constitutive values per unit thickness stay fixed.
inline PlateConstitutive<SecondOrderScalar> constantConstitutive(const PlateConstitutive<double>& source)
{
    PlateConstitutive<SecondOrderScalar> result;
    result.hasMembrane = source.hasMembrane;
    result.hasBending = source.hasBending;
    result.hasCoupling = source.hasCoupling;
    result.hasShearFlexibility = source.hasShearFlexibility;
    result.membrane = constantMatrix(source.membrane);
    result.bending = constantMatrix(source.bending);
    result.coupling = constantMatrix(source.coupling);
    result.transverseShear = constantMatrix(source.transverseShear);
    result.transverseShearCompliance = constantMatrix(source.transverseShearCompliance);
    result.bendingRatio = source.bendingRatio;
    result.shearRatio = source.shearRatio;
    result.youngX = source.youngX;
    result.youngY = source.youngY;
    result.shearModulusX = source.shearModulusX;
    result.shearModulusY = source.shearModulusY;
    return result;
}

template <std::size_t N>
std::array<SecondOrderScalar, N> secondOrderThickness(const std::array<double, N>& values,
                                                     const std::array<double, N>& rates)
{
    std::array<SecondOrderScalar, N> result;
    for (std::size_t k = 0; k < N; ++k)
        result[k] = {values[k], rates[k]};
    return result;
}

template <std::size_t N>
base::MatN<N> secondDerivativeOf(const ScalarMatrix<SecondOrderScalar, N>& matrix)
{
    base::MatN<N> result;
    for (std::size_t k = 0; k < matrix.values.size(); ++k)
        result.values[k] = matrix.values[k].second;
    return result;
}

template <typename Data, typename Direction>
[[nodiscard]] PlateConstitutive<Complex> steppedConstitutive(const Data& data, const Direction& direction)
{
    return plateConstitutiveOf(complexStepped(data.values, direction.section), data.materialAngle);
}

template <std::size_t N>
[[nodiscard]] std::array<Complex, N> steppedThickness(const std::array<double, N>& thickness,
                                                      const std::array<double, N>& rates)
{
    std::array<Complex, N> result{};
    for (std::size_t a = 0; a < N; ++a)
        result[a] = complexStepped(thickness[a], rates[a]);
    return result;
}

// The rows' rates from complex-stepped rows: the Linear columns' rates; a stress or strain row's von Mises along its
// components (planeVonMisesRate), its angle and principal values NaN.
inline void appendRowRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                           const std::vector<LinearRow<Complex>>& rows, ElementOutput output)
{
    constexpr double NONE = std::numeric_limits<double>::quiet_NaN();
    const ColumnForm vonMises = shellOutputColumnForms(output).back();
    for (const LinearRow<Complex>& linear : rows)
    {
        std::array<double, 8> rates{};
        for (std::size_t k = 0; k < 8; ++k)
            rates[k] = complexStepDerivative(linear.values[k]);
        if (output != ElementOutput::Forces)
        {
            const std::array<double, 3> values{linear.values[1].real(), linear.values[2].real(),
                                               linear.values[3].real()};
            const std::array<double, 3> componentRates{rates[1], rates[2], rates[3]};
            rates[4] = NONE;
            rates[5] = NONE;
            rates[6] = NONE;
            rates[7] = planeVonMisesRate(vonMises, values, componentRates);
        }
        block.append(row, elementId, linear.place, rates);
    }
}

// The complex-stepped centre values in the output frame and the fibres of an element along a direction, without
// temperature.
template <typename Shell, typename Data, typename Direction>
[[nodiscard]] std::pair<std::array<Complex, CENTRE_RECOVERY_ROWS>, std::array<Complex, 2>> steppedCentre(
    const Data& data, const Direction& direction, const std::array<double, Shell::DOFS>& q)
{
    const std::array<Complex, Shell::NODES> thickness = steppedThickness(data.thickness, direction.thickness);
    const std::array<Complex, 2> fibres = fibresOf(data.section, thickness);
    const CentreRecoveryOperator<Complex, Shell::DOFS> recovery = inOutputFrame(
        Shell::recovery(data.geometry, steppedConstitutive(data, direction), thickness, fibres, PlateThermalStrain{}),
        data.geometry.userAngle);
    return {centreValuesOf(recovery, q), fibres};
}

} // namespace edge_shear_detail

// The fields with a closed form: the ShellProp T when every corner follows it (the corner rates t_a/T), a AnisoMat2D
// stiffness term of the ShellProp's materials (shellSectionRate, in the material frame), a ply thickness of the
// laminate (laminateSectionRate: the section's rate at fixed T and the rate of T at every corner).
template <typename Shell>
std::optional<typename EdgeShearShellFamily<Shell>::Direction> EdgeShearShellFamily<Shell>::directionOf(
    const Data& data, const DesignedField& field) const
{
    constexpr double MATERIAL_FRAME = 0.0;
    Direction direction;
    direction.section = plateSectionRatesOf(data.values, ShellSectionRate{}, 0.0);
    if (isShellThickness(field))
    {
        if (!data.followsThickness)
            return std::nullopt;
        for (std::size_t a = 0; a < NODES; ++a)
            direction.thickness[a] = data.thickness[a] / data.section.thickness;
        return direction;
    }
    if (isAnisoMat2DStiffnessTerm(field))
    {
        const std::optional<ShellSectionRate> rate = shellSectionRate(data.section, field, MATERIAL_FRAME);
        if (!rate)
            return std::nullopt;
        direction.section = plateSectionRatesOf(data.values, *rate, 0.0);
        return direction;
    }
    if (isPlyThickness(field))
    {
        std::optional<LaminateSectionRate> rate = laminateSectionRate(data.section, field, MATERIAL_FRAME);
        if (!rate)
            return std::nullopt;
        direction.section = plateSectionRatesOf(data.values, rate->atFixedThickness, rate->density);
        direction.thickness.fill(rate->thickness);
        direction.laminate = std::move(rate);
        return direction;
    }
    return std::nullopt;
}

template <typename Shell>
bool EdgeShearShellFamily<Shell>::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                           const DesignedField& field, ElementMatrixBlock& block) const
{
    using namespace edge_shear_detail;
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const ScalarMatrix<Complex, DOFS> k = Shell::stiffness(data.geometry, steppedConstitutive(data, *direction),
                                                           steppedThickness(data.thickness, direction->thickness));
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(matrixOf(complexStepDerivativeOf(k))));
    return true;
}

// The mass moves with the thickness and the density only: the ShellProp T and a ply thickness.
template <typename Shell>
bool EdgeShearShellFamily<Shell>::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, const MassOptions& options,
                                                      ElementMatrixBlock& block) const
{
    using namespace edge_shear_detail;
    if (!isShellThickness(field) && !isPlyThickness(field))
        return false;
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const double added =
        options.addedNonStructuralMass.empty() ? 0.0 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
    const ScalarMatrix<Complex, DOFS> m =
        Shell::mass(data.geometry, complexStepped(data.values.density, direction->section.density),
                    Complex(data.values.nonStructuralMass + added, 0.0),
                    steppedThickness(data.thickness, direction->thickness), options.formulation);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.massToDisplacementSystems(matrixOf(complexStepDerivativeOf(m)), view.isMassOnOffsets()));
    return true;
}

// K_d is linear in the membrane forces: dK_d = K_d(dN), dN their complex-step rate at fixed displacements.
template <typename Shell>
bool EdgeShearShellFamily<Shell>::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                                       const DesignedField& field,
                                                                       const NodalDisplacements& preload,
                                                                       ElementMatrixBlock& block) const
{
    using namespace edge_shear_detail;
    if (preload.temperatures != nullptr)
        return false;
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const std::array<Complex, NODES> thickness = steppedThickness(data.thickness, direction->thickness);
    const CentreRecoveryOperator<Complex, DOFS> recovery =
        Shell::recovery(data.geometry, steppedConstitutive(data, *direction), thickness,
                        fibresOf(data.section, thickness), PlateThermalStrain{});
    const std::array<Complex, CENTRE_RECOVERY_ROWS> values =
        centreValuesOf(recovery, data.frames.localDisplacements(preload));
    std::array<double, 3> forceRate{};
    for (std::size_t k = 0; k < 3; ++k)
        forceRate[k] = complexStepDerivative(values[centre_row::MEMBRANE_FORCE + k]);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(Shell::differentialStiffness(data.geometry, forceRate)));
    return true;
}

template <typename Shell>
bool EdgeShearShellFamily<Shell>::differentialStiffnessFieldMixedDerivative(
    const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    using namespace edge_shear_detail;
    if (!isShellThickness(first) || !isShellThickness(second) || first.id != second.id ||
        preload.temperatures != nullptr)
        return ElementFamily::differentialStiffnessFieldMixedDerivative(view, row, first, second, preload, block);
    const Data data = resolved(view, row);
    const auto direction = directionOf(data, first);
    if (!direction)
        return false;
    const auto thickness = secondOrderThickness(data.thickness, direction->thickness);
    const auto recovery = Shell::recovery(data.geometry, constantConstitutive(data.constitutive), thickness,
                                          fibresOf(data.section, thickness), PlateThermalStrain{});
    const auto values = centreValuesOf(recovery, data.frames.localDisplacements(preload));
    std::array<double, 3> forceRate{};
    for (std::size_t k = 0; k < forceRate.size(); ++k)
        forceRate[k] = values[centre_row::MEMBRANE_FORCE + k].second;
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(Shell::differentialStiffness(data.geometry, forceRate)));
    return true;
}

// The rows' rates at fixed displacements (no temperature). A ply thickness gives the forces only (the homogeneous
// stresses and strains of a laminate under NOCOMPS read fibres the plies move: differenced).
template <typename Shell>
bool EdgeShearShellFamily<Shell>::recoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                                     const DesignedField& field, ElementOutput output,
                                                     const NodalDisplacements& displacements,
                                                     ElementResultBlock& rates) const
{
    using namespace edge_shear_detail;
    if (displacements.temperatures != nullptr)
        return false;
    requireOutput(output);
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction || (direction->laminate && output != ElementOutput::Forces))
        return false;
    const auto [values, fibres] = steppedCentre<Shell>(data, *direction, data.frames.localDisplacements(displacements));
    rates.reset(outputColumns(output));
    appendRowRates(rates, row, view.elementId(row),
                   linearRowsOf(values, fibres, data.section, output, compositeOutputOf(view.snapshot())), output);
    return true;
}

// The ply rows' rates along a ply thickness (no temperature): the centre state and its complex-step rates through the
// plies (ShellRecovery::appendLaminateRowRates).
template <typename Shell>
bool EdgeShearShellFamily<Shell>::layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                                          const DesignedField& field, LayerOutput output,
                                                          const NodalDisplacements& displacements,
                                                          ElementResultBlock& rates) const
{
    using namespace edge_shear_detail;
    if (!isPlyThickness(field) || displacements.temperatures != nullptr)
        return false;
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction || !direction->laminate)
        return false;
    const auto [values, fibres] = steppedCentre<Shell>(data, *direction, data.frames.localDisplacements(displacements));
    std::array<double, CENTRE_RECOVERY_ROWS> real{};
    std::array<double, CENTRE_RECOVERY_ROWS> rate{};
    for (std::size_t k = 0; k < CENTRE_RECOVERY_ROWS; ++k)
    {
        real[k] = values[k].real();
        rate[k] = complexStepDerivative(values[k]);
    }
    const ShellPlaceState centre = centreStateOf(real, meanOf(data.thickness), {});
    ShellPlaceState centreRates = centreStateOf(rate, 0.0, {});
    centreRates.thickness = meanOf(direction->thickness);
    rates.reset(layerOutputColumns(output));
    appendLaminateRowRates(rates, row, view.elementId(row), centre, centreRates, outputSectionOf(data, data.geometry),
                           direction->laminate->laminate, output, compositeOutputOf(view.snapshot()));
    return true;
}

template <typename Shell>
bool EdgeShearShellFamily<Shell>::stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                                 const DesignedField& field,
                                                                 ElementMatrixBlock& block) const
{
    using namespace edge_shear_detail;
    if (!isShellThickness(field))
        return false;
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const auto second = secondDerivativeOf(Shell::stiffness(data.geometry, constantConstitutive(data.constitutive),
        secondOrderThickness(data.thickness, direction->thickness)));
    appendMatrix(block, row, data.frames.dofs(), data.frames.toDisplacementSystems(second));
    return true;
}

template <typename Shell>
bool EdgeShearShellFamily<Shell>::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                            const DesignedField& field, const MassOptions& options,
                                                            ElementMatrixBlock& block) const
{
    using namespace edge_shear_detail;
    if (!isShellThickness(field))
        return false;
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const double added =
        options.addedNonStructuralMass.empty() ? 0.0 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
    const auto second = secondDerivativeOf(Shell::mass(data.geometry, SecondOrderScalar(data.values.density),
        SecondOrderScalar(data.values.nonStructuralMass + added),
        secondOrderThickness(data.thickness, direction->thickness), options.formulation));
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.massToDisplacementSystems(second, view.isMassOnOffsets()));
    return true;
}

// The slopes of the rows' rates in T: the rates at the unit local displacements (exact: the rates of the Linear
// columns are linear in q, the fibre's rate constant).
template <typename Shell>
bool EdgeShearShellFamily<Shell>::recoverySlopeRates(const ElementBatchView& view, std::int64_t row,
                                                     const DesignedField& field, ElementOutput output,
                                                     RecoverySlopes& rates) const
{
    using namespace edge_shear_detail;
    if (!isShellThickness(field))
        return false;
    requireOutput(output);
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    const std::array<Complex, NODES> thickness = steppedThickness(data.thickness, direction->thickness);
    const std::array<Complex, 2> fibres = fibresOf(data.section, thickness);
    const CentreRecoveryOperator<Complex, DOFS> recovery = inOutputFrame(
        Shell::recovery(data.geometry, steppedConstitutive(data, *direction), thickness, fibres, PlateThermalStrain{}),
        data.geometry.userAngle);
    shellRecoverySlopes<NODES>(
        data.frames, shellOutputColumnForms(output), outputColumns(output),
        [&](const std::array<double, DOFS>& q, ElementResultBlock& block)
        {
            appendRowRates(block, row, elementId,
                           linearRowsOf(centreValuesOf(recovery, q), fibres, data.section, output, composite), output);
        },
        rates);
    return true;
}

// Exact second thickness rates at fixed displacement. Nonlinear columns are chained by the design layer.
template <typename Shell>
bool EdgeShearShellFamily<Shell>::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                                           const DesignedField& field, ElementOutput output,
                                                           const NodalDisplacements& displacements,
                                                           ElementResultBlock& rates) const
{
    using namespace edge_shear_detail;
    constexpr double NONE = std::numeric_limits<double>::quiet_NaN();
    if (!isShellThickness(field) || displacements.temperatures != nullptr)
        return false;
    requireOutput(output);
    const Data data = resolved(view, row);
    const std::optional<Direction> direction = directionOf(data, field);
    if (!direction)
        return false;
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::array<double, DOFS> q = data.frames.localDisplacements(displacements);
    const auto thickness = secondOrderThickness(data.thickness, direction->thickness);
    const auto fibres = fibresOf(data.section, thickness);
    const auto recovery = inOutputFrame(Shell::recovery(data.geometry, constantConstitutive(data.constitutive),
        thickness, fibres, PlateThermalStrain{}), data.geometry.userAngle);
    const auto centre = centreValuesOf(recovery, q);
    rates.reset(outputColumns(output));
    for (const auto& linear : linearRowsOf(centre, fibres, data.section, output, composite))
    {
        std::array<double, 8> values{};
        for (std::size_t k = 0; k < 8; ++k)
            values[k] = linear.values[k].second;
        if (output != ElementOutput::Forces)
        {
            values[0] = 0.0;
            for (std::size_t k = 4; k < 8; ++k)
                values[k] = NONE;
        }
        rates.append(row, view.elementId(row), linear.place, values);
    }
    return true;
}

} // namespace oos::elements
