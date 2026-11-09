// The facets of EdgeShearShellFamily (EdgeShearShellFamily.hpp): the resolution of an element, its check, stiffness,
// mass, loads and differential stiffness. Included by the translation unit of each element
// (shellQuad4/ShellQuad4Element.cpp, shellTria3/ShellTria3Element.cpp), which instantiates the family once.

#pragma once

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/EdgeShearShellFamily.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::elements
{

namespace edge_shear_detail
{

// O_o_Reader gives MCID −1 when the field holds THETA.
inline constexpr std::int32_t NO_MAT_FRAME_ID = -1;
// An MCID whose x axis projects on the element plane to less than this (for a unit axis) has no direction there.
inline constexpr double MATERIAL_FRAME_MIN_PROJECTION = 1.0e-8;

template <std::size_t N>
[[nodiscard]] base::MatN<N> matrixOf(const ScalarMatrix<double, N>& k)
{
    base::MatN<N> m;
    m.values = k.values;
    return m;
}

template <typename T, std::size_t N>
[[nodiscard]] T meanOf(const std::array<T, N>& values)
{
    T sum{};
    for (const T& value : values)
        sum += value;
    return sum / static_cast<double>(N);
}

// The fibre distances Z1, Z2 at the centre: as given on the ShellProp, or ∓t̄/2 of the mean thickness when blank.
template <typename T, std::size_t N>
[[nodiscard]] std::array<T, 2> fibresOf(const ShellSection& section, const std::array<T, N>& thickness)
{
    const T half = 0.5 * meanOf(thickness);
    return {section.hasFiberDistance1 ? T(section.fiberDistance1) : -half,
            section.hasFiberDistance2 ? T(section.fiberDistance2) : half};
}

// The free thermal strain of an element (PlateConstitutive.hpp) at its mean thickness.
template <typename Data>
[[nodiscard]] PlateThermalStrain thermalStrainOf(const Data& data, const ShellTemperatureChange& temperature)
{
    return plateThermalStrainOf(data.section, data.materialAngle, data.constitutive, temperature,
                                meanOf(data.thickness));
}

// The centre recovery operator of the kernel turned from the element frame to the frame at `angle` from it (the
// output frame): the forces, moments and stresses as stresses, the strains and curvatures as engineering strains,
// the transverse shear forces as a vector.
template <typename T, std::size_t COLUMNS>
[[nodiscard]] ScalarMatrix<T, CENTRE_RECOVERY_ROWS, COLUMNS> inOutputFrame(
    const ScalarMatrix<T, CENTRE_RECOVERY_ROWS, COLUMNS>& element, double angle)
{
    constexpr std::size_t DOFS = COLUMNS - 1;
    const Matrix3<double> stress = stressRotation(angle);
    const Matrix3<double> strain = engineeringStrainRotation(angle);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    ScalarMatrix<T, CENTRE_RECOVERY_ROWS, COLUMNS> output;
    const auto rotate = [&](std::size_t first, const Matrix3<double>& rotation)
    {
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                for (std::size_t k = 0; k <= DOFS; ++k)
                    output(first + i, k) += rotation(i, j) * element(first + j, k);
    };
    rotate(centre_row::MEMBRANE_FORCE, stress);
    rotate(centre_row::MOMENT, stress);
    rotate(centre_row::STRESS_FIBER1, stress);
    rotate(centre_row::STRESS_FIBER2, stress);
    rotate(centre_row::MEMBRANE_STRAIN, strain);
    rotate(centre_row::CURVATURE, strain);
    for (std::size_t k = 0; k <= DOFS; ++k)
    {
        const T qx = element(centre_row::SHEAR_FORCE, k);
        const T qy = element(centre_row::SHEAR_FORCE + 1, k);
        output(centre_row::SHEAR_FORCE, k) = c * qx + s * qy;
        output(centre_row::SHEAR_FORCE + 1, k) = -s * qx + c * qy;
    }
    return output;
}

// The centre values R·[q; 1] of a recovery operator at the local displacements q.
template <typename T, std::size_t COLUMNS>
[[nodiscard]] std::array<T, CENTRE_RECOVERY_ROWS> centreValuesOf(
    const ScalarMatrix<T, CENTRE_RECOVERY_ROWS, COLUMNS>& recovery, const std::array<double, COLUMNS - 1>& q)
{
    constexpr std::size_t DOFS = COLUMNS - 1;
    std::array<T, CENTRE_RECOVERY_ROWS> values{};
    for (std::size_t i = 0; i < CENTRE_RECOVERY_ROWS; ++i)
    {
        T sum = recovery(i, DOFS);
        for (std::size_t k = 0; k < DOFS; ++k)
            sum += recovery(i, k) * q[k];
        values[i] = sum;
    }
    return values;
}

} // namespace edge_shear_detail

template <typename Shell>
EdgeShearShellFamily<Shell>::Columns::Columns(const ElementBatchView& view)
    : connectivity(view),
      sections(view),
      theta(view.columns().values<double>("materialAngle")),
      matFrameId(view.columns().values<std::int32_t>("matFrameId")),
      offset(view.columns().values<double>("offset")),
      thicknessFlag(view.columns().values<std::int32_t>("thicknessFlag")),
      cornerThickness(view.columns().fixedLists<double>("cornerThickness"))
{
}

template <typename Shell>
Resolution<typename EdgeShearShellFamily<Shell>::Data> EdgeShearShellFamily<Shell>::resolve(
    const ElementBatchView& view, const Columns& columns, std::int64_t row) const
{
    using Result = Resolution<Data>;
    const auto r = static_cast<std::size_t>(row);
    Resolution<ShellCorners<NODES>> corners = resolveShellCorners<NODES>(view, columns.connectivity, row);
    if (!corners)
        return Result::failure(std::move(corners.problem));
    Resolution<ShellSection> section = resolveShellSection(view, columns.sections, row);
    if (!section)
        return Result::failure(std::move(section.problem));
    auto geometry = Shell::buildGeometry(corners.value->positions);
    if (!geometry.problem.empty())
        return Result::failure(elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, geometry.problemField,
                                              std::move(geometry.problem)));

    Data data;
    data.geometry = geometry.geometry;
    data.section = *section.value;
    data.gridIds = corners.value->ids;
    Resolution<std::array<double, NODES>> thickness = resolveCornerThicknesses<NODES>(
        view, row, data.section, columns.thicknessFlag[r], columns.cornerThickness[row]);
    if (!thickness)
        return Result::failure(std::move(thickness.problem));
    data.thickness = *thickness.value;
    data.followsThickness =
        cornersFollowThickness(data.section, columns.thicknessFlag[r], columns.cornerThickness[row]);

    // The bending card decides how the shear-correction moduli are read (PlateConstitutive.hpp).
    std::optional<ShellMaterial> bendingMaterial;
    if (data.section.hasBending && !data.section.isLaminate())
    {
        Resolution<ShellMaterial> material = resolveShellMaterial(
            view, columns.sections.materials, row, data.section.materialIds[1], "ShellProp.bendingMatId");
        if (!material)
            return Result::failure(std::move(material.problem));
        bendingMaterial = *material.value;
    }

    // The material axis from the element x axis: THETA from the user x axis (G1→G2), or the projection of MCID's x
    // axis on the element plane, measured and applied in the element frame.
    if (columns.matFrameId[r] == edge_shear_detail::NO_MAT_FRAME_ID)
    {
        data.materialAngle = data.geometry.userAngle + columns.theta[r] * std::numbers::pi / 180.0;
    }
    else
    {
        Resolution<input::CoordinateFrame> system =
            resolveRectangularFrame(view, row, columns.matFrameId[r], "matFrameId");
        if (!system)
            return Result::failure(std::move(system.problem));
        const base::Mat3& e = data.geometry.axes;
        const base::Mat3& axes = system.value->axes;
        const double alongX = axes(0, 0) * e(0, 0) + axes(1, 0) * e(1, 0) + axes(2, 0) * e(2, 0);
        const double alongY = axes(0, 0) * e(0, 1) + axes(1, 0) * e(1, 1) + axes(2, 0) * e(2, 1);
        if (std::hypot(alongX, alongY) <= edge_shear_detail::MATERIAL_FRAME_MIN_PROJECTION)
            return Result::failure(elementProblem(
                view, row, problem_code::DEGENERATE_GEOMETRY, "matFrameId",
                base::concatenate({"expected the x axis of coordinate system ", std::to_string(columns.matFrameId[r]),
                                   " to cross the element plane at an angle, found it normal to the element; give "
                                   "another ShellQuad4.matFrameId or a ShellQuad4.materialAngle"})));
        data.materialAngle = std::atan2(alongY, alongX);
    }
    data.values = plateSectionValuesOf(data.section, bendingMaterial ? &*bendingMaterial : nullptr);
    data.constitutive = plateConstitutiveOf(data.values, data.materialAngle);
    if (std::optional<std::string> problem = Shell::integrationProblem(data.geometry, data.thickness))
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node1", std::move(*problem)));
    if (std::optional<std::string> problem = offsetSectionProblem(data.section, columns.offset[r]))
        return Result::failure(elementProblem(view, row, problem_code::INVALID_FIELD, "offset", std::move(*problem)));
    if (std::optional<std::string> problem = offsetDefinitionProblem(view.snapshot(), columns.offset[r]))
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, "offset", std::move(*problem)));
    data.frames = ShellNodeFrames<NODES>::build(corners.value->rows, corners.value->cdFrames, data.geometry.axes,
                                                columns.offset[r]);
    return Result::success(std::move(data));
}

template <typename Shell>
typename EdgeShearShellFamily<Shell>::Data EdgeShearShellFamily<Shell>::resolved(const ElementBatchView& view,
                                                                                 std::int64_t row) const
{
    const Columns columns(view);
    return resolve(view, columns, row).required();
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    checkRows(view, wholeBatch(view), sink);
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::checkRows(const ElementBatchView& view, ElementRange range,
                                            base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        Resolution<Data> resolution = resolve(view, columns, row);
        if (!resolution)
            sink.report(std::move(resolution.problem));
    }
}

// The design fields: ShellProp Z1 and Z2 move the stress fibres only, a AnisoMat2D stiffness term the stiffness,
// thermal loads and recovery (not the mass, which reads ρ and the thickness), NSM the mass only (the shared rules of
// core/DesignFieldUse); a pressure reads the element's own geometry only.
template <typename Shell>
FieldUse EdgeShearShellFamily<Shell>::designFieldUse(const DesignedField& field) const
{
    constexpr FieldUse ANISO_MAT2D_TERM_USE{true, false, true, true, false};
    const FieldUse use = findFieldUse(field, {{"ShellProp", "fiberDistance1", FieldUse::recoveryOnly()},
                                              {"ShellProp", "fiberDistance2", FieldUse::recoveryOnly()},
                                              {"AnisoMat2D", "stiffness", ANISO_MAT2D_TERM_USE}});
    return field.card == card() ? use : use.withoutDistributedLoads();
}

// The kernel's K in the element frame, carried to the displacement systems of the grid points through ZOFFS.
template <typename Shell>
void EdgeShearShellFamily<Shell>::computeStiffness(const ElementBatchView& view, ElementRange range,
                                                   ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ScalarMatrix<double, DOFS> k = Shell::stiffness(data.geometry, data.constitutive, data.thickness);
        appendMatrix(block, row, data.frames.dofs(), data.frames.toDisplacementSystems(edge_shear_detail::matrixOf(k)));
    }
}

// The kernel's translational mass, ρ t + NSM per unit area (the ShellProp's NSM and the NSM cards' share; ρ of MID1, of
// MID2 without MID1, ShellSection), through the offset only under ModelParams OFFDEF LROFF or ELMZ.
template <typename Shell>
void EdgeShearShellFamily<Shell>::computeMass(const ElementBatchView& view, ElementRange range,
                                              const MassOptions& options, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const std::span<const double> addedMass = options.addedNonStructuralMass;
        const double added = addedMass.empty() ? 0.0 : addedMass[static_cast<std::size_t>(row)];
        const ScalarMatrix<double, DOFS> m =
            Shell::mass(data.geometry, data.values.density, data.values.nonStructuralMass + added, data.thickness,
                        options.formulation);
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.massToDisplacementSystems(edge_shear_detail::matrixOf(m), view.isMassOnOffsets()));
    }
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::checkPressureLoads(const ElementBatchView& view,
                                                     std::span<const ElementPressure> pressures,
                                                     base::DiagnosticSink& sink) const
{
    checkShellPressures(view, pressures, sink);
}

// The kernel's pressure loads (PLOAD4, and ShellPressureLoad as a uniform FacePressureLoad), without the offset unless
// ModelParams OFFDEF puts the loads on it.
template <typename Shell>
void EdgeShearShellFamily<Shell>::computePressureLoads(const ElementBatchView& view,
                                                       std::span<const ElementPressure> pressures,
                                                       ElementVectorBlock& block) const
{
    const Columns columns(view);
    for (const ElementPressure& pressure : pressures)
    {
        const Data data = resolve(view, columns, pressure.elementRow).required();
        const PressureDirection direction = pressureDirectionOf(view, pressure, data.geometry.axes);
        if (!direction.problem.empty())
            throw std::invalid_argument(base::concatenate(
                {std::string(card()), " ", std::to_string(view.elementId(pressure.elementRow)), ": ", direction.problem,
                 " (the loads facet was called on a pressure checkPressureLoads() refuses: a bug "
                 "of O_o_Solver, report it)"}));
        const std::array<double, DOFS> loads =
            Shell::pressureLoads(data.geometry, pressure.cornerPressures, direction.inElementFrame);
        block.append(pressure.elementRow, data.frames.dofs(),
                     data.frames.loadsToDisplacementSystems(loads, view.areLoadsOnOffsets()));
    }
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::checkThermalLoads(const ElementBatchView& view,
                                                    const ElementTemperatures& temperatures,
                                                    base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        Resolution<Data> data = resolve(view, columns, row);
        if (!data)
            continue; // check() reports it
        Resolution<ShellTemperatureChange> change =
            resolveShellTemperature(view, row, temperatures, data.value->frames.gridRows, data.value->section);
        if (!change)
            sink.report(std::move(change.problem));
    }
}

// The kernel's thermal loads of the element's mean temperature change and gradient, through the offset (see the
// header: consistent with the stiffness, whatever ModelParams OFFDEF).
template <typename Shell>
void EdgeShearShellFamily<Shell>::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                                      const ElementTemperatures& temperatures,
                                                      ElementVectorBlock& block) const
{
    constexpr bool THROUGH_OFFSET = true;
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellTemperatureChange change =
            resolveShellTemperature(view, row, temperatures, data.frames.gridRows, data.section).required();
        if (change.isZero())
            continue;
        const PlateThermalStrain thermal = edge_shear_detail::thermalStrainOf(data, change);
        const std::array<double, DOFS> loads =
            Shell::thermalLoads(data.geometry, data.constitutive, data.thickness, thermal);
        block.append(row, data.frames.dofs(), data.frames.loadsToDisplacementSystems(loads, THROUGH_OFFSET));
    }
}

// The membrane forces of the preload recovered by the kernel (element frame, the thermal force removed), then the
// kernel's membrane-stress differential stiffness, carried to the displacement systems through ZOFFS.
template <typename Shell>
void EdgeShearShellFamily<Shell>::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                               const NodalDisplacements& preload,
                                                               ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellTemperatureChange temperature =
            preload.temperatures == nullptr
                ? ShellTemperatureChange{}
                : resolveShellTemperature(view, row, *preload.temperatures, data.frames.gridRows, data.section)
                      .required();
        const CentreRecoveryOperator<double, DOFS> recovery = Shell::recovery(
            data.geometry, data.constitutive, data.thickness, edge_shear_detail::fibresOf(data.section, data.thickness),
            edge_shear_detail::thermalStrainOf(data, temperature));
        const std::array<double, CENTRE_RECOVERY_ROWS> values =
            edge_shear_detail::centreValuesOf(recovery, data.frames.localDisplacements(preload));
        const std::array<double, 3> force{values[centre_row::MEMBRANE_FORCE], values[centre_row::MEMBRANE_FORCE + 1],
                                          values[centre_row::MEMBRANE_FORCE + 2]};
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.toDisplacementSystems(Shell::differentialStiffness(data.geometry, force)));
    }
}

} // namespace oos::elements
