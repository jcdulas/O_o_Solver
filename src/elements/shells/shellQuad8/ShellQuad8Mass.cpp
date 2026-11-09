#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"

namespace oos::elements
{

// The consistent mass of the shape functions (coupled, with rotary inertia) or the HRZ diagonal of its translational
// part (lumped), integrated with the full rule of the shape (MindlinShell::mass).
void ShellQuad8Element::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                   ElementMatrixBlock& block) const
{
    appendHigherOrderShellMass<SerendipityQuadShape>(view, range, options, block);
}

} // namespace oos::elements
