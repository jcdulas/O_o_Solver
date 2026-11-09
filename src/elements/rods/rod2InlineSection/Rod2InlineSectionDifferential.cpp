#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"

namespace oos::elements
{

// (P/L)·[D −D; −D D] on the translations, P the axial force of the preload (rodDifferentialStiffness).
void Rod2InlineSectionElement::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                          const NodalDisplacements& preload,
                                                          ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendRodDifferentialStiffness(view, range, preload, block,
                                   [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
