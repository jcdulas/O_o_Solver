#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"

namespace oos::elements
{

// Forces P, T and stresses P/A, C·T/J with their margins against the IsoMat limits (rodOutputRow).
void Rod2InlineSectionElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const Columns columns(view);
    requireOutput(output);
    recoverRods(view, range, output, displacements, block,
                [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

// The recovery matrix of the rod in closed form (rodRecoverySlopes).
bool Rod2InlineSectionElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                            RecoverySlopes& slopes) const
{
    requireOutput(output);
    const Columns columns(view);
    rodRecoverySlopes(resolve(view, columns, row).required(), output, slopes);
    return true;
}

void Rod2InlineSectionElement::completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                                                   std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    const Columns columns(view);
    completeRodRows(output, blocks, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
