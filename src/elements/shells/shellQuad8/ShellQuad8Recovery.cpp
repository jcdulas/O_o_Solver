#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"

namespace oos::elements
{

std::vector<std::string> ShellQuad8Element::outputColumns(ElementOutput output) const
{
    return shellOutputColumns(output);
}

std::vector<std::string> ShellQuad8Element::layerOutputColumns(LayerOutput output) const
{
    return laminateOutputColumns(output);
}

// The rows of the centre and of the corner grid points (MindlinShell::placeStates), each in the tangent frame of its
// place (the element frame on a flat element: HigherOrderShell.hpp).
void ShellQuad8Element::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                               const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    recoverHigherOrderShells<SerendipityQuadShape>(view, range, output, displacements, block);
}

void ShellQuad8Element::recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    recoverHigherOrderShellLayers<SerendipityQuadShape>(view, range, output, displacements, block);
}

void ShellQuad8Element::recoverOutputs(const ElementBatchView& view, ElementRange range,
                                      std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                      std::span<ElementResultBlock> blocks) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    recoverHigherOrderShellOutputs<SerendipityQuadShape>(view, range, outputs, displacements, blocks);
}

bool ShellQuad8Element::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                      RecoverySlopes& slopes) const
{
    requireOutput(output);
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    higherOrderShellSlopes<SerendipityQuadShape>(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
    return true;
}

bool ShellQuad8Element::layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                                           RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    higherOrderShellSlopes<SerendipityQuadShape>(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
    return true;
}

bool ShellQuad8Element::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                               std::span<const RecoveryOutput> outputs,
                                               std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    higherOrderShellSlopes<SerendipityQuadShape>(view, row, outputs, slopes);
    return true;
}

void ShellQuad8Element::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                             std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    completeShellRows(output, blocks);
}

void ShellQuad8Element::completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                                  std::span<const ElementResultBlock> sources,
                                                  std::span<ElementResultBlock> blocks) const
{
    if (output != LayerOutput::FailureIndices)
    {
        completeLaminateRows(output, blocks);
        return;
    }
    completeHigherOrderShellFailureRows<SerendipityQuadShape>(view, sources, blocks);
}

} // namespace oos::elements
