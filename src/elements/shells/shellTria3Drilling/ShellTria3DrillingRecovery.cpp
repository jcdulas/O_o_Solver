#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellRecoverySlopes.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

namespace oos::elements
{

std::vector<std::string> ShellTria3DrillingElement::outputColumns(ElementOutput output) const
{
    return shellOutputColumns(output);
}

std::vector<std::string> ShellTria3DrillingElement::layerOutputColumns(LayerOutput output) const
{
    return laminateOutputColumns(output);
}

namespace
{

// The temperature change of an element for its recovery: none without a thermal load in the subcase.
template <typename Data>
[[nodiscard]] ShellTemperatureChange temperatureOf(const ElementBatchView& view, std::int64_t row, const Data& data,
                                                   const NodalDisplacements& displacements)
{
    if (displacements.temperatures == nullptr)
        return {};
    return resolveShellTemperature(view, row, *displacements.temperatures, data.frames.gridRows, data.section)
        .required();
}

} // namespace

// The rows of the centre (ShellTria3DrillingPlate::centreState), in the element frame (ShellRecovery).
void ShellTria3DrillingElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                               const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

void ShellTria3DrillingElement::recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

// Every output from one resolution, one plate and one centre state per element, as ShellQuad4DrillingElement's.
void ShellTria3DrillingElement::recoverOutputs(const ElementBatchView& view, ElementRange range,
                                      std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                      std::span<ElementResultBlock> blocks) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const bool hasElementOutput = resetShellRecoveryBlocks(outputs, blocks);
    const Columns columns(view);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        if (!hasElementOutput && !data.section.isLaminate())
            continue;
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 18> q = data.frames.localDisplacements(displacements);
        const std::array<ShellPlaceState, 1> states{
            plate.centreState(q, temperatureOf(view, row, data, displacements))};
        for (std::size_t k = 0; k < outputs.size(); ++k)
            appendShellOutputRows(blocks[k], row, view.elementId(row), states, data.section, outputs[k], composite,
                                  ShellRowValues::All);
    }
}

std::vector<ColumnForm> ShellTria3DrillingElement::outputColumnForms(ElementOutput output) const
{
    return shellOutputColumnForms(output);
}

std::vector<ColumnForm> ShellTria3DrillingElement::layerOutputColumnForms(LayerOutput output) const
{
    return laminateOutputColumnForms(output);
}

// The recovery matrices of the element (shellRecoverySlopes): the plate built once, its centre state at the unit local
// displacements (as ShellQuad4DrillingElement::recoverySlopes).
bool ShellTria3DrillingElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                      RecoverySlopes& slopes) const
{
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The centre state's parts that do not depend on the displacements, once for every unit displacement.
    const ShellTria3DrillingPlate::CentreOperators operators = plate.centreOperators();
    const ThermalResultants atRest = plate.centreThermal({});
    shellRecoverySlopes<3>(
        data.frames, shellOutputColumnForms(output), outputColumns(output),
        [&](const std::array<double, 18>& q, ElementResultBlock& block)
        {
            appendShellRows(block, row, elementId, plate.centreState(operators, atRest, q, {}), data.section, output,
                            composite, ShellRowValues::LinearOnly);
        },
        slopes);
    return true;
}

bool ShellTria3DrillingElement::layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                                           RecoverySlopes& slopes) const
{
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The centre state's parts that do not depend on the displacements, once for every unit displacement.
    const ShellTria3DrillingPlate::CentreOperators operators = plate.centreOperators();
    const ThermalResultants atRest = plate.centreThermal({});
    shellRecoverySlopes<3>(
        data.frames, laminateOutputColumnForms(output), layerOutputColumns(output),
        [&](const std::array<double, 18>& q, ElementResultBlock& block)
        {
            if (data.section.isLaminate())
                appendLaminateRows(block, row, elementId, plate.centreState(operators, atRest, q, {}), data.section,
                                   output, composite);
        },
        slopes);
    return true;
}

// The slopes of several outputs from one plate and one centre state per unit displacement (as
// ShellQuad4DrillingElement::recoverySlopesOfOutputs).
bool ShellTria3DrillingElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                               std::span<const RecoveryOutput> outputs,
                                               std::span<RecoverySlopes> slopes) const
{
    std::vector<std::vector<ColumnForm>> forms; // bounded by the outputs
    std::vector<std::vector<std::string>> columns;
    for (const RecoveryOutput& output : outputs)
    {
        if (!output.isLayer)
            requireOutput(output.output);
        forms.push_back(shellRecoveryOutputForms(output));
        columns.push_back(shellRecoveryOutputColumns(output));
    }
    const Columns snapshotColumns(view);
    const Data data = resolve(view, snapshotColumns, row).required();
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The centre state's parts that do not depend on the displacements, once for every unit displacement.
    const ShellTria3DrillingPlate::CentreOperators operators = plate.centreOperators();
    const ThermalResultants atRest = plate.centreThermal({});
    shellRecoverySlopesOfOutputs<3>(
        data.frames, forms, columns,
        [&](const std::array<double, 18>& q, std::span<ElementResultBlock> blocks)
        {
            const std::array<ShellPlaceState, 1> states{plate.centreState(operators, atRest, q, {})};
            for (std::size_t k = 0; k < outputs.size(); ++k)
                appendShellOutputRows(blocks[k], row, elementId, states, data.section, outputs[k], composite);
        },
        slopes);
    return true;
}

std::size_t ShellTria3DrillingElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SHELL_VECTORISED_RECOVERY_MIN_CASES;
}

void ShellTria3DrillingElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                             std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    completeShellRows(output, blocks);
}

// The ply stresses and strains completed in place, the failure indices made from the ply states (ShellRecovery).
void ShellTria3DrillingElement::completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                                  std::span<const ElementResultBlock> sources,
                                                  std::span<ElementResultBlock> blocks) const
{
    if (output != LayerOutput::FailureIndices)
    {
        completeLaminateRows(output, blocks);
        return;
    }
    const Columns columns(view);
    completeLaminateFailureRows(sources, blocks, [&](std::int64_t row, const auto& visit)
                                { visit(resolve(view, columns, row).required().section); });
}

} // namespace oos::elements
