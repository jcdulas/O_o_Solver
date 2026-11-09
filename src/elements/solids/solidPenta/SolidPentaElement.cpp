#include "oos/elements/solids/solidPenta/SolidPentaElement.hpp"

namespace oos::elements
{

// Every element of the batch resolves (grid points, SolidProp, material, shape) or is reported (SolidElementKernel).
void SolidPentaElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    KERNEL.check(view, sink);
}

} // namespace oos::elements
