// The recovery facets of EdgeShearShellFamily (EdgeShearShellFamily.hpp): the rows of the centre, in the user frame,
// from the kernel's centre recovery operator; the rows of the corners after them, from its corner recovery operators,
// for an output asked with them (RecoveryOutput::hasCorners) of a kernel that has them (the ShellQuad4); the plies of
// a laminate from the centre state. Included after EdgeShearShellFamily.ipp (the element translation units include
// the three files in order).

#pragma once

#include "oos/elements/shells/ShellRecoverySlopes.hpp"

#include <limits>

namespace oos::elements
{

namespace edge_shear_detail
{

// The first value of the curvature row of a STRCUR strain table, by the convention of the result tables.
inline constexpr double CURVATURE_ROW_FLAG = -1.0;

// One row of an element output in its Linear columns (ShellRecovery's forms): the eight forces, or the first column
// (fibre distance, curvature flag) and the three components of a stress or strain row.
template <typename T>
struct LinearRow
{
    ResultPlace place;
    std::array<T, 8> values{};
};

template <typename T>
[[nodiscard]] std::array<T, 3> triple(const std::array<T, CENTRE_RECOVERY_ROWS>& values, std::size_t first)
{
    return {values[first], values[first + 1], values[first + 2]};
}

// The rows of `output` at one place (`nodeId`: 0 for the centre, the grid point of a corner) from its values in the
// output frame: the forces (the moments as the force table writes them, M = D κ); the stresses at Z1 and Z2 (layers
// 1, 2); the strains at Z1 and Z2, ε(z) = ε_m − z κ; STRCUR the membrane strains (layer 1, flag 0) and the curvatures
// (layer 2, flag −1). A laminate writes its stresses and strains here only when PARAM NoCompositeStress asks for the
// homogeneous element.
template <typename T>
[[nodiscard]] std::vector<LinearRow<T>> linearRowsOf(const std::array<T, CENTRE_RECOVERY_ROWS>& values,
                                                     const std::array<T, 2>& fibres, const ShellSection& section,
                                                     ElementOutput output, CompositeOutput composite,
                                                     std::int32_t nodeId = 0)
{
    std::vector<LinearRow<T>> rows; // at most two
    if (output == ElementOutput::Forces)
    {
        LinearRow<T> row;
        row.place.nodeId = nodeId;
        for (std::size_t k = 0; k < 8; ++k)
            row.values[k] = values[centre_row::MEMBRANE_FORCE + k];
        rows.push_back(row);
        return rows;
    }
    if (section.isLaminate() && composite == CompositeOutput::PliesOnly)
        return rows;
    const std::array<T, 3> strain = triple(values, centre_row::MEMBRANE_STRAIN);
    const std::array<T, 3> curvature = triple(values, centre_row::CURVATURE);
    if (output == ElementOutput::StrainCurvatures)
    {
        rows.push_back({{nodeId, 1, 0.0}, {T(0.0), strain[0], strain[1], strain[2]}});
        rows.push_back({{nodeId, 2, 0.0}, {T(CURVATURE_ROW_FLAG), curvature[0], curvature[1], curvature[2]}});
        return rows;
    }
    for (std::size_t layer = 0; layer < 2; ++layer)
    {
        const T& z = fibres[layer];
        LinearRow<T> row;
        row.place = {nodeId, static_cast<std::int32_t>(layer + 1), 0.0};
        row.values[0] = z;
        const std::array<T, 3> stress =
            triple(values, layer == 0 ? centre_row::STRESS_FIBER1 : centre_row::STRESS_FIBER2);
        for (std::size_t k = 0; k < 3; ++k)
            row.values[1 + k] = output == ElementOutput::Strains ? strain[k] - z * curvature[k] : stress[k];
        rows.push_back(row);
    }
    return rows;
}

// Appends the rows: the forces as they are; a stress or strain row completed by shellStressRow or shellStrainRow
// (ShellRowValues::All), or with its columns that are not Linear at zero (LinearOnly: the rows of the slopes).
inline void appendLinearRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                             const std::vector<LinearRow<double>>& rows, ElementOutput output, ShellRowValues values)
{
    for (const LinearRow<double>& linear : rows)
    {
        if (output == ElementOutput::Forces || values == ShellRowValues::LinearOnly)
        {
            block.append(row, elementId, linear.place, linear.values);
            continue;
        }
        const std::array<double, 3> components{linear.values[1], linear.values[2], linear.values[3]};
        const std::array<double, 8> full = output == ElementOutput::Stresses
                                               ? shellStressRow(linear.values[0], components)
                                               : shellStrainRow(linear.values[0], components);
        block.append(row, elementId, linear.place, full);
    }
}

// The centre state for the plies of a laminate (ShellRecovery::appendLaminateRows) in the output frame: total
// membrane strains and curvatures, the shear forces, and the moments as resultants ∫σ z dz (−M of the force table).
inline ShellPlaceState centreStateOf(const std::array<double, CENTRE_RECOVERY_ROWS>& values, double thickness,
                                     const ShellTemperatureChange& temperature)
{
    ShellPlaceState state;
    state.thickness = thickness;
    state.membraneStrain = triple(values, centre_row::MEMBRANE_STRAIN);
    state.curvature = triple(values, centre_row::CURVATURE);
    state.membraneForce = triple(values, centre_row::MEMBRANE_FORCE);
    for (std::size_t k = 0; k < 3; ++k)
        state.moment[k] = -values[centre_row::MOMENT + k];
    state.shearForce = {values[centre_row::SHEAR_FORCE], values[centre_row::SHEAR_FORCE + 1]};
    state.temperature = temperature;
    return state;
}

// The values of the corners of an element at its local displacements, in the output frame, with the fibre distances
// Z1 and Z2 of each corner: as given on the ShellProp, or ∓t/2 of the thickness of the corner when blank.
template <std::size_t NODES>
struct CornerValues
{
    std::array<std::array<double, CENTRE_RECOVERY_ROWS>, NODES> values{};
    std::array<std::array<double, 2>, NODES> fibres{};
};

template <std::size_t NODES>
[[nodiscard]] std::array<std::array<double, 2>, NODES> cornerFibresOf(const ShellSection& section,
                                                                      const std::array<double, NODES>& thickness)
{
    std::array<std::array<double, 2>, NODES> fibres{};
    for (std::size_t i = 0; i < NODES; ++i)
        fibres[i] = {section.hasFiberDistance1 ? section.fiberDistance1 : -0.5 * thickness[i],
                     section.hasFiberDistance2 ? section.fiberDistance2 : 0.5 * thickness[i]};
    return fibres;
}

// Whether `output` gets the corner rows of the kernel `Shell`.
template <typename Shell>
[[nodiscard]] constexpr bool takesCorners(const RecoveryOutput& output) noexcept
{
    return Shell::HAS_CORNER_RECOVERY && !output.isLayer && output.hasCorners;
}

template <typename Shell>
[[nodiscard]] bool takeCorners(std::span<const RecoveryOutput> outputs) noexcept
{
    for (const RecoveryOutput& output : outputs)
        if (takesCorners<Shell>(output))
            return true;
    return false;
}

// The corner recovery operators of the kernel in the output frame (a kernel without corner recovery has no such
// call: takesCorners is then false for every output).
template <typename Shell, typename Data>
[[nodiscard]] std::array<CentreRecoveryOperator<double, Shell::DOFS>, Shell::NODES> cornerOperatorsOf(
    const Data& data, const std::array<std::array<double, 2>, Shell::NODES>& fibres, const PlateThermalStrain& thermal)
{
    std::array<CentreRecoveryOperator<double, Shell::DOFS>, Shell::NODES> operators{};
    if constexpr (Shell::HAS_CORNER_RECOVERY)
    {
        operators = Shell::cornerRecovery(data.geometry, data.constitutive, data.thickness, fibres, thermal);
        for (CentreRecoveryOperator<double, Shell::DOFS>& corner : operators)
            corner = inOutputFrame(corner, data.geometry.userAngle);
    }
    return operators;
}

// Appends the rows of `output` of the corners of an element (the grid points `gridIds`, in their order).
template <std::size_t NODES>
void appendCornerRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                      const CornerValues<NODES>& corners, const std::array<std::int32_t, NODES>& gridIds,
                      const ShellSection& section, ElementOutput output, CompositeOutput composite,
                      ShellRowValues values)
{
    for (std::size_t i = 0; i < NODES; ++i)
        appendLinearRows(block, row, elementId,
                         linearRowsOf(corners.values[i], corners.fibres[i], section, output, composite, gridIds[i]),
                         output, values);
}

// The section of an element written in its output frame (the ply angles of a laminate from that frame's x axis).
template <typename Data, typename Geometry>
[[nodiscard]] ShellSection outputSectionOf(const Data& data, const Geometry& geometry)
{
    return data.section.rotatedToElementFrame(data.materialAngle - geometry.userAngle);
}

} // namespace edge_shear_detail

template <typename Shell>
std::vector<std::string> EdgeShearShellFamily<Shell>::outputColumns(ElementOutput output) const
{
    return shellOutputColumns(output);
}

template <typename Shell>
std::vector<std::string> EdgeShearShellFamily<Shell>::layerOutputColumns(LayerOutput output) const
{
    return laminateOutputColumns(output);
}

template <typename Shell>
std::vector<ColumnForm> EdgeShearShellFamily<Shell>::outputColumnForms(ElementOutput output) const
{
    return shellOutputColumnForms(output);
}

template <typename Shell>
std::vector<ColumnForm> EdgeShearShellFamily<Shell>::layerOutputColumnForms(LayerOutput output) const
{
    return laminateOutputColumnForms(output);
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                          const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                                const NodalDisplacements& displacements,
                                                ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

// Every output from one resolution and one centre recovery per element, the thermal strains of the subcase
// subtracted (the stresses and forces; the strains are total); the corner operators of the element are formed when an
// output takes its corner rows.
template <typename Shell>
void EdgeShearShellFamily<Shell>::recoverOutputs(const ElementBatchView& view, ElementRange range,
                                                 std::span<const RecoveryOutput> outputs,
                                                 const NodalDisplacements& displacements,
                                                 std::span<ElementResultBlock> blocks) const
{
    using namespace edge_shear_detail;
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const bool hasElementOutput = resetShellRecoveryBlocks(outputs, blocks);
    const bool hasCorners = takeCorners<Shell>(outputs);
    const Columns columns(view);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        if (!hasElementOutput && !data.section.isLaminate())
            continue;
        const ShellTemperatureChange temperature =
            displacements.temperatures == nullptr
                ? ShellTemperatureChange{}
                : resolveShellTemperature(view, row, *displacements.temperatures, data.frames.gridRows, data.section)
                      .required();
        const std::array<double, 2> fibres = fibresOf(data.section, data.thickness);
        const PlateThermalStrain thermal = thermalStrainOf(data, temperature);
        const CentreRecoveryOperator<double, DOFS> recovery =
            inOutputFrame(Shell::recovery(data.geometry, data.constitutive, data.thickness, fibres, thermal),
                          data.geometry.userAngle);
        const std::array<double, DOFS> q = data.frames.localDisplacements(displacements);
        const std::array<double, CENTRE_RECOVERY_ROWS> values = centreValuesOf(recovery, q);
        CornerValues<NODES> corners;
        if (hasCorners)
        {
            corners.fibres = cornerFibresOf(data.section, data.thickness);
            const auto operators = cornerOperatorsOf<Shell>(data, corners.fibres, thermal);
            for (std::size_t i = 0; i < NODES; ++i)
                corners.values[i] = centreValuesOf(operators[i], q);
        }
        const std::int32_t elementId = view.elementId(row);
        for (std::size_t k = 0; k < outputs.size(); ++k)
        {
            if (outputs[k].isLayer)
            {
                appendLaminateRows(blocks[k], row, elementId,
                                   centreStateOf(values, meanOf(data.thickness), temperature),
                                   outputSectionOf(data, data.geometry), outputs[k].layerOutput, composite);
                continue;
            }
            appendLinearRows(blocks[k], row, elementId,
                             linearRowsOf(values, fibres, data.section, outputs[k].output, composite),
                             outputs[k].output, ShellRowValues::All);
            if (takesCorners<Shell>(outputs[k]))
                appendCornerRows(blocks[k], row, elementId, corners, data.gridIds, data.section, outputs[k].output,
                                 composite, ShellRowValues::All);
        }
    }
}

// The slopes of the rows: the recovery operator formed once, its rows at the unit local displacements
// (shellRecoverySlopes: exact, the rows are affine in q).
template <typename Shell>
bool EdgeShearShellFamily<Shell>::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                                 RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

template <typename Shell>
bool EdgeShearShellFamily<Shell>::layerRecoverySlopes(const ElementBatchView& view, std::int64_t row,
                                                      LayerOutput output, RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

template <typename Shell>
bool EdgeShearShellFamily<Shell>::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                          std::span<const RecoveryOutput> outputs,
                                                          std::span<RecoverySlopes> slopes) const
{
    using namespace edge_shear_detail;
    std::vector<std::vector<ColumnForm>> forms; // bounded by the outputs
    std::vector<std::vector<std::string>> columns;
    for (const RecoveryOutput& output : outputs)
    {
        if (!output.isLayer)
            requireOutput(output.output);
        forms.push_back(shellRecoveryOutputForms(output));
        columns.push_back(shellRecoveryOutputColumns(output));
    }
    const Data data = resolved(view, row);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    const std::array<double, 2> fibres = fibresOf(data.section, data.thickness);
    const CentreRecoveryOperator<double, DOFS> recovery =
        inOutputFrame(Shell::recovery(data.geometry, data.constitutive, data.thickness, fibres, PlateThermalStrain{}),
                      data.geometry.userAngle);
    const ShellSection outputSection = outputSectionOf(data, data.geometry);
    const double thickness = meanOf(data.thickness);
    const bool hasCorners = takeCorners<Shell>(outputs);
    CornerValues<NODES> corners;
    std::array<CentreRecoveryOperator<double, DOFS>, NODES> cornerOperators{};
    if (hasCorners)
    {
        corners.fibres = cornerFibresOf(data.section, data.thickness);
        cornerOperators = cornerOperatorsOf<Shell>(data, corners.fibres, PlateThermalStrain{});
    }
    shellRecoverySlopesOfOutputs<NODES>(
        data.frames, forms, columns,
        [&](const std::array<double, DOFS>& q, std::span<ElementResultBlock> blocks)
        {
            const std::array<double, CENTRE_RECOVERY_ROWS> values = centreValuesOf(recovery, q);
            if (hasCorners)
                for (std::size_t i = 0; i < NODES; ++i)
                    corners.values[i] = centreValuesOf(cornerOperators[i], q);
            for (std::size_t k = 0; k < outputs.size(); ++k)
            {
                if (outputs[k].isLayer)
                {
                    appendLaminateRows(blocks[k], row, elementId, centreStateOf(values, thickness, {}), outputSection,
                                       outputs[k].layerOutput, composite);
                    continue;
                }
                appendLinearRows(blocks[k], row, elementId,
                                 linearRowsOf(values, fibres, data.section, outputs[k].output, composite),
                                 outputs[k].output, ShellRowValues::LinearOnly);
                if (takesCorners<Shell>(outputs[k]))
                    appendCornerRows(blocks[k], row, elementId, corners, data.gridIds, data.section,
                                     outputs[k].output, composite, ShellRowValues::LinearOnly);
            }
        },
        slopes);
    return true;
}

template <typename Shell>
std::size_t EdgeShearShellFamily<Shell>::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SHELL_VECTORISED_RECOVERY_MIN_CASES;
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                                        std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    completeShellRows(output, blocks);
}

template <typename Shell>
void EdgeShearShellFamily<Shell>::completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                                             std::span<const ElementResultBlock> sources,
                                                             std::span<ElementResultBlock> blocks) const
{
    if (!isPlyIndexOutput(output))
    {
        completeLaminateRows(output, blocks);
        return;
    }
    const Columns columns(view);
    completeLaminateFailureRows(output, sources, blocks, [&](std::int64_t row, const auto& visit)
                                { visit(resolve(view, columns, row).required().section); });
}

} // namespace oos::elements
