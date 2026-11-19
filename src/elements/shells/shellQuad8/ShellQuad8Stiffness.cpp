#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"

namespace oos::elements
{

// K of MindlinShell (membrane, Mindlin bending and shear, coupling; nothing on the rotation about a director) carried
// to the displacement systems of the grid points through ZOFFS (ShellNodeFrames).
void ShellQuad8Element::computeStiffness(const ElementBatchView& view, ElementRange range,
                                        ElementMatrixBlock& block) const
{
    appendHigherOrderShellStiffness<SerendipityQuadShape>(view, range, block);
}

bool ShellQuad8Element::computeStabilisationEnergies(const ElementBatchView& view, ElementRange range,
                                                     const NodalDisplacements& displacements,
                                                     std::span<double> energies) const
{
    higherOrderShellStabilisationEnergies<SerendipityQuadShape>(view, range, displacements, energies);
    return true;
}

} // namespace oos::elements
