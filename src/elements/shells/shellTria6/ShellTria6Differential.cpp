#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

namespace oos::elements
{

// The membrane resultants of the preload at the centre on the differential stiffness of ShellDifferential
// (appendHigherOrderShellDifferential).
void ShellTria6Element::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    appendHigherOrderShellDifferential<QuadraticTriangleShape>(view, range, preload, block);
}

} // namespace oos::elements
