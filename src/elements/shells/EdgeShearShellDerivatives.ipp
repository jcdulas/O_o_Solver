// The design derivatives of EdgeShearShellFamily (EdgeShearShellFamily.hpp): the closed forms by the complex step
// through the kernels, along the rates of the element's inputs (Direction). Included after EdgeShearShellRecovery.ipp.
//
// First derivatives: the kernel evaluated in complex numbers at inputs x + i h ẋ gives Im f / h = f′·ẋ, exact to
// round-off (ShellScalar.hpp). Second derivatives in the ShellProp T: the 4-point central rule on complex-step first
// derivatives, f″ ≈ [−f′(T + 2δ) + 8 f′(T + δ) − 8 f′(T − δ) + f′(T − 2δ)] / (12 δ), δ = 1e-3 T (truncation error
// of order δ⁴ f⁽⁵⁾/30, about 1e-12 relative for these polynomial-like functions of T; round-off about 1e-13).

#pragma once

#include <limits>

namespace oos::elements
{

namespace edge_shear_detail
{

// δ / T of the second derivatives in T.
inline constexpr double SECOND_DERIVATIVE_STEP = 1.0e-3;
// The 4-point rule: the shifts in δ and their weights (over 12 δ).
inline constexpr std::array<double, 4> SECOND_DERIVATIVE_SHIFTS{-2.0, -1.0, 1.0, 2.0};
inline constexpr std::array<double, 4> SECOND_DERIVATIVE_WEIGHTS{1.0, -8.0, 8.0, -1.0};

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

// The element with every corner thickness moved by `shift` (in units of the ShellProp T): the corners follow T.
template <typename Data>
[[nodiscard]] Data withThicknessShift(Data data, double shift)
{
    for (double& t : data.thickness)
        t *= 1.0 + shift / data.section.thickness;
    return data;
}

// The second derivative in T of the values `firstRate(data)` gives (the first rates of an element as a vector), by the
// 4-point rule.
template <typename Data, typename FirstRate>
[[nodiscard]] std::vector<double> secondRateInThickness(const Data& data, FirstRate&& firstRate)
{
    const double delta = SECOND_DERIVATIVE_STEP * data.section.thickness;
    std::vector<double> result;
    for (std::size_t s = 0; s < SECOND_DERIVATIVE_SHIFTS.size(); ++s)
    {
        const std::vector<double> rates = firstRate(withThicknessShift(data, SECOND_DERIVATIVE_SHIFTS[s] * delta));
        if (result.empty())
            result.assign(rates.size(), 0.0);
        for (std::size_t k = 0; k < rates.size(); ++k)
            result[k] += SECOND_DERIVATIVE_WEIGHTS[s] * rates[k];
    }
    for (double& value : result)
        value /= 12.0 * delta;
    return result;
}

template <std::size_t N>
[[nodiscard]] std::vector<double> flattened(const ScalarMatrix<double, N>& m)
{
    return std::vector<double>(m.values.begin(), m.values.end());
}

template <std::size_t N>
[[nodiscard]] base::MatN<N> matrixFrom(const std::vector<double>& values)
{
    base::MatN<N> m;
    std::copy(values.begin(), values.end(), m.values.begin());
    return m;
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
    const std::vector<double> second = secondRateInThickness(
        data,
        [&](const Data& shifted)
        {
            return flattened(
                complexStepDerivativeOf(Shell::stiffness(shifted.geometry, steppedConstitutive(shifted, *direction),
                                                         steppedThickness(shifted.thickness, direction->thickness))));
        });
    appendMatrix(block, row, data.frames.dofs(), data.frames.toDisplacementSystems(matrixFrom<DOFS>(second)));
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
    const std::vector<double> second =
        secondRateInThickness(data,
                              [&](const Data& shifted)
                              {
                                  return flattened(complexStepDerivativeOf(Shell::mass(
                                      shifted.geometry, Complex(shifted.values.density, 0.0),
                                      Complex(shifted.values.nonStructuralMass + added, 0.0),
                                      steppedThickness(shifted.thickness, direction->thickness), options.formulation)));
                              });
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.massToDisplacementSystems(matrixFrom<DOFS>(second), view.isMassOnOffsets()));
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

// The rows' second rates in T at fixed displacements (no temperature): the Linear columns by the 4-point rule on
// their complex-step rates; the fibre distance's is zero (a blank Z is ∓t̄/2, linear in T); the other columns NaN.
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
    std::vector<ResultPlace> places; // the rows' places, bounded by two rows
    const std::vector<double> second = secondRateInThickness(
        data,
        [&](const Data& shifted)
        {
            const auto [values, fibres] = steppedCentre<Shell>(shifted, *direction, q);
            std::vector<double> flat;
            places.clear();
            for (const LinearRow<Complex>& linear : linearRowsOf(values, fibres, shifted.section, output, composite))
            {
                places.push_back(linear.place);
                for (const Complex& value : linear.values)
                    flat.push_back(complexStepDerivative(value));
            }
            return flat;
        });
    rates.reset(outputColumns(output));
    for (std::size_t r = 0; r < places.size(); ++r)
    {
        std::array<double, 8> values{};
        for (std::size_t k = 0; k < 8; ++k)
            values[k] = second[8 * r + k];
        if (output != ElementOutput::Forces)
        {
            values[0] = 0.0;
            for (std::size_t k = 4; k < 8; ++k)
                values[k] = NONE;
        }
        rates.append(row, view.elementId(row), places[r], values);
    }
    return true;
}

} // namespace oos::elements
