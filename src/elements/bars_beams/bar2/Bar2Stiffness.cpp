#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

namespace oos::elements
{

// The stiffness of each Bar2: BeamFlexibility (uniform section), the pin flags condensed, then carried through the
// offsets to the displacement systems of GA and GB (beamStiffness).
void Bar2Element::computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendBeamStiffness(range, block, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
