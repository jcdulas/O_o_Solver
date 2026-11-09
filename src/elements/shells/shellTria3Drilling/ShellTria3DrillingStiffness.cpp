#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

namespace oos::elements
{

// K18 of ShellTria3DrillingPlate (drilling membrane, bending, edge shear) in the element frame, carried to the displacement
// systems of G1-G3 through the ZOFFS offset (ShellNodeFrames).
void ShellTria3DrillingElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                        ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        appendMatrix(block, row, data.frames.dofs(), data.frames.toDisplacementSystems(plate.stiffness()));
    }
}

} // namespace oos::elements
