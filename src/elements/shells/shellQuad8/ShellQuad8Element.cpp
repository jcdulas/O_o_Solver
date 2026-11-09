#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"

#include "oos/elements/shells/HigherOrderShell.hpp"

namespace oos::elements
{

void ShellQuad8Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    checkHigherOrderShells<SerendipityQuadShape>(view, sink);
}

} // namespace oos::elements
