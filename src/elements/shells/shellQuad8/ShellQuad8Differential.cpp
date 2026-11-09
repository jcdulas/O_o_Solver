#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"

namespace oos::elements
{

// The membrane resultants of the preload at the centre on the differential stiffness of ShellDifferential
// (appendHigherOrderShellDifferential).
void ShellQuad8Element::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    appendHigherOrderShellDifferential<SerendipityQuadShape>(view, range, preload, block);
}

} // namespace oos::elements
