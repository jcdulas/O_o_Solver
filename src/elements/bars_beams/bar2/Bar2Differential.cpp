#include "oos/elements/bars_beams/BeamDifferential.hpp"
#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

namespace oos::elements
{

// The consistent differential stiffness of the axial force of the preload (BeamDifferential), pin flags condensed,
// carried through the offsets to the displacement systems of GA and GB.
void Bar2Element::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                  const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendBeamDifferentialStiffness(view, range, preload, block,
                                    [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
