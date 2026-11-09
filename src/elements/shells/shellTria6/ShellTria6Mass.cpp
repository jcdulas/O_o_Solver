#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

namespace oos::elements
{

// The consistent mass of the shape functions (coupled, with rotary inertia) or the HRZ diagonal of its translational
// part (lumped), integrated with the full rule of the shape (MindlinShell::mass).
void ShellTria6Element::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                   ElementMatrixBlock& block) const
{
    appendHigherOrderShellMass<QuadraticTriangleShape>(view, range, options, block);
}

} // namespace oos::elements
