#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellRecoverySlopes.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

namespace oos::elements
{

std::vector<std::string> ShellQuad4DrillingElement::outputColumns(ElementOutput output) const
{
    return shellOutputColumns(output);
}

std::vector<std::string> ShellQuad4DrillingElement::layerOutputColumns(LayerOutput output) const
{
    return laminateOutputColumns(output);
}

// The rows of the centre and of G1-G4 (ShellQuad4DrillingPlate::placeStates), in the element frame (ShellRecovery), the thermal
// strains of the subcase subtracted.
void ShellQuad4DrillingElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                               const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

// The ply rows of the laminates of the range at their centre (ShellRecovery::appendLaminateRows).
void ShellQuad4DrillingElement::recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

// Every output from one resolution, one plate and one set of place states per element (an element without a laminate
// has no layer rows: without an element output it is not formulated).
// PERFORMANCE: the rows at rest of a block of cases resolved and formulated every element again for each output
// (six for a laminate: 4.1 of the 10.2 s of 32 cases of the CompProp plate on one thread); the rows of each output are
// those of its recovery alone, bit for bit (the same states).
// Measured on the recovery of a homogeneous and of a CompProp plate meshed with this element.
void ShellQuad4DrillingElement::recoverOutputs(const ElementBatchView& view, ElementRange range,
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
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 24> q = data.frames.localDisplacements(displacements);
        const ShellTemperatureChange temperature =
            displacements.temperatures == nullptr
                ? ShellTemperatureChange{}
                : resolveShellTemperature(view, row, *displacements.temperatures, data.frames.gridRows, data.section)
                      .required();
        const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds, temperature);
        for (std::size_t k = 0; k < outputs.size(); ++k)
            appendShellOutputRows(blocks[k], row, view.elementId(row), states, data.section, outputs[k], composite,
                                  ShellRowValues::All);
    }
}

// The rows of recover() at each unit displacement: recoverOutputs' steps for one element (no temperature: the rest
// state of placeThermal), the plate and its place operators built once; placeStates(operators, thermal, q, ...) gives
// the bits of placeStates(q, ...) and localUnitDisplacements those of localDisplacements, so each block holds the bits
// of recover() under that displacement.
// PERFORMANCE: the rows of output::SectionLoads took a recovery per DOF, each resolving the element and building its
// plate again (24 per element, with a stiffness for its DOFs): 0.95 s of the 1.6 s of a design point of the stiffened
// panel (3 118 sections); here one resolution and one plate per element.
// Measured on a stiffened barrel panel model (a loads re-solve), on one thread.
bool ShellQuad4DrillingElement::recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                                  std::vector<ElementDof>& dofs,
                                                  std::vector<ElementResultBlock>& blocks) const
{
    requireOutput(output);
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const ShellQuad4DrillingPlate::PlaceOperators operators = plate.placeOperators();
    const ShellQuad4DrillingPlate::PlaceThermal atRest = plate.placeThermal(operators, ShellTemperatureChange{});
    const std::array<ElementDof, 24> elementDofs = data.frames.dofs();
    dofs.assign(elementDofs.begin(), elementDofs.end());
    blocks.resize(dofs.size());
    const std::int32_t elementId = view.elementId(row);
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        resetShellRecoveryBlocks(outputs, std::span<ElementResultBlock>(&blocks[d], 1));
        const std::array<double, 24> q = data.frames.localUnitDisplacements(dofs[d]);
        const std::array<ShellPlaceState, 5> states = plate.placeStates(operators, atRest, q, data.gridIds);
        appendShellOutputRows(blocks[d], row, elementId, states, data.section, outputs[0], composite,
                              ShellRowValues::All);
    }
    return true;
}

std::vector<ColumnForm> ShellQuad4DrillingElement::outputColumnForms(ElementOutput output) const
{
    return shellOutputColumnForms(output);
}

std::vector<ColumnForm> ShellQuad4DrillingElement::layerOutputColumnForms(LayerOutput output) const
{
    return laminateOutputColumnForms(output);
}

// The recovery matrices of the element (shellRecoverySlopes): the plate built once, its place states at the unit local
// displacements.
// PERFORMANCE: the derivatives of an element's items with respect to its displacements took two or more recoveries
// per DOF, each resolving the element and building its plate again (the membrane, the hybrid bending); here one
// resolution and one plate for every DOF, the rows exact.
// Measured on the design derivatives of a shell model.
bool ShellQuad4DrillingElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                      RecoverySlopes& slopes) const
{
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The place states' parts that do not depend on the displacements, once for every unit displacement.
    const ShellQuad4DrillingPlate::PlaceOperators operators = plate.placeOperators();
    const ShellQuad4DrillingPlate::PlaceThermal atRest = plate.placeThermal(operators, {});
    shellRecoverySlopes<4>(
        data.frames, shellOutputColumnForms(output), outputColumns(output),
        [&](const std::array<double, 24>& q, ElementResultBlock& block)
        {
            for (const ShellPlaceState& state : plate.placeStates(operators, atRest, q, data.gridIds))
                appendShellRows(block, row, elementId, state, data.section, output, composite,
                                ShellRowValues::LinearOnly);
        },
        slopes);
    return true;
}

bool ShellQuad4DrillingElement::layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                                           RecoverySlopes& slopes) const
{
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The place states' parts that do not depend on the displacements, once for every unit displacement.
    const ShellQuad4DrillingPlate::PlaceOperators operators = plate.placeOperators();
    const ShellQuad4DrillingPlate::PlaceThermal atRest = plate.placeThermal(operators, {});
    shellRecoverySlopes<4>(
        data.frames, laminateOutputColumnForms(output), layerOutputColumns(output),
        [&](const std::array<double, 24>& q, ElementResultBlock& block)
        {
            if (data.section.isLaminate())
                appendLaminateRows(block, row, elementId, plate.placeStates(operators, atRest, q, data.gridIds)[0],
                                   data.section, output, composite);
        },
        slopes);
    return true;
}

// The slopes of several outputs from one plate and one set of place states per unit displacement (the vectorised
// recovery), their rows without the columns that are not Linear (those get no slopes).
// PERFORMANCE: the plate and the 25 place states of the slopes were made again for each output (37 µs per output on
// one thread); here once for all of them, and no principal values computed; the place states' parts that do not
// depend on the displacements formed once (ShellQuad4DrillingPlate::PlaceOperators).
// Measured on the recovery of shell models with several outputs.
bool ShellQuad4DrillingElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The place states' parts that do not depend on the displacements, once for every unit displacement.
    const ShellQuad4DrillingPlate::PlaceOperators operators = plate.placeOperators();
    const ShellQuad4DrillingPlate::PlaceThermal atRest = plate.placeThermal(operators, {});
    shellRecoverySlopesOfOutputs<4>(
        data.frames, forms, columns,
        [&](const std::array<double, 24>& q, std::span<ElementResultBlock> blocks)
        {
            const std::array<ShellPlaceState, 5> states = plate.placeStates(operators, atRest, q, data.gridIds);
            for (std::size_t k = 0; k < outputs.size(); ++k)
                appendShellOutputRows(blocks[k], row, elementId, states, data.section, outputs[k], composite);
        },
        slopes);
    return true;
}

std::size_t ShellQuad4DrillingElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SHELL_VECTORISED_RECOVERY_MIN_CASES;
}

void ShellQuad4DrillingElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                             std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    completeShellRows(output, blocks);
}

// The ply stresses and strains completed in place, the failure indices made from the ply states (ShellRecovery).
void ShellQuad4DrillingElement::completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
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
