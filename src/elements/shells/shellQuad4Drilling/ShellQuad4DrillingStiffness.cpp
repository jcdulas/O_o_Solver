#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

namespace oos::elements
{

// K24 of ShellQuad4DrillingPlate (drilling membrane, bending, edge shear; the warp offsets of a warped element) in the element
// frame, carried to the displacement systems of G1-G4 through the ZOFFS offset (ShellNodeFrames).
void ShellQuad4DrillingElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                        ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        appendMatrix(block, row, data.frames.dofs(), data.frames.toDisplacementSystems(plate.stiffness()));
    }
}

} // namespace oos::elements
