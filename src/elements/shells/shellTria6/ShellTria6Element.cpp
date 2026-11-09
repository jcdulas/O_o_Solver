#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

#include "oos/elements/shells/HigherOrderShell.hpp"

namespace oos::elements
{

void ShellTria6Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    checkHigherOrderShells<QuadraticTriangleShape>(view, sink);
}

} // namespace oos::elements
