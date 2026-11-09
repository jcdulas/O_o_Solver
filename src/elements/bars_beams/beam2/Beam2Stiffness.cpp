#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"

namespace oos::elements
{

// The stiffness of each Beam2: BeamFlexibility (tapered: integrated along the stations), the pin flags condensed,
// then carried through the offsets to the displacement systems of GA and GB (beamStiffness).
void Beam2Element::computeStiffness(const ElementBatchView& view, ElementRange range,
                                       ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendBeamStiffness(range, block, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
