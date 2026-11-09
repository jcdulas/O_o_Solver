#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

namespace oos::elements
{

// K of MindlinShell (membrane, Mindlin bending and shear, coupling; nothing on the rotation about a director) carried
// to the displacement systems of the grid points through ZOFFS (ShellNodeFrames).
void ShellTria6Element::computeStiffness(const ElementBatchView& view, ElementRange range,
                                        ElementMatrixBlock& block) const
{
    appendHigherOrderShellStiffness<QuadraticTriangleShape>(view, range, block);
}

} // namespace oos::elements
