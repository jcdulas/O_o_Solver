#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

namespace oos::elements
{

std::vector<std::string> ShellTria6Element::outputColumns(ElementOutput output) const
{
    return shellOutputColumns(output);
}

std::vector<std::string> ShellTria6Element::layerOutputColumns(LayerOutput output) const
{
    return laminateOutputColumns(output);
}

// The rows of the centre and of the corner grid points (MindlinShell::placeStates), each in the tangent frame of its
// place (the element frame on a flat element: HigherOrderShell.hpp).
void ShellTria6Element::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                               const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    recoverHigherOrderShells<QuadraticTriangleShape>(view, range, output, displacements, block);
}

void ShellTria6Element::recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    recoverHigherOrderShellLayers<QuadraticTriangleShape>(view, range, output, displacements, block);
}

void ShellTria6Element::recoverOutputs(const ElementBatchView& view, ElementRange range,
                                      std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                      std::span<ElementResultBlock> blocks) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    recoverHigherOrderShellOutputs<QuadraticTriangleShape>(view, range, outputs, displacements, blocks);
}

bool ShellTria6Element::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                      RecoverySlopes& slopes) const
{
    requireOutput(output);
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    higherOrderShellSlopes<QuadraticTriangleShape>(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
    return true;
}

bool ShellTria6Element::layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                                           RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    higherOrderShellSlopes<QuadraticTriangleShape>(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
    return true;
}

bool ShellTria6Element::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                               std::span<const RecoveryOutput> outputs,
                                               std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    higherOrderShellSlopes<QuadraticTriangleShape>(view, row, outputs, slopes);
    return true;
}

void ShellTria6Element::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                             std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    completeShellRows(output, blocks);
}

void ShellTria6Element::completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                                  std::span<const ElementResultBlock> sources,
                                                  std::span<ElementResultBlock> blocks) const
{
    if (output != LayerOutput::FailureIndices)
    {
        completeLaminateRows(output, blocks);
        return;
    }
    completeHigherOrderShellFailureRows<QuadraticTriangleShape>(view, sources, blocks);
}

} // namespace oos::elements
