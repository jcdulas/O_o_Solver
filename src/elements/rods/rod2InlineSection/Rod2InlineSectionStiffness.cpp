#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"

namespace oos::elements
{

// Axial EA/L and torsional GJ/L along the axis, in the displacement systems of G1 and G2 (rodStiffness).
void Rod2InlineSectionElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                              ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendRodStiffness(range, block, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
