#include "oos/elements/solids/solidPyramid/SolidPyramidElement.hpp"

namespace oos::elements
{

// Every element of the batch resolves (grid points, SolidProp, material, shape) or is reported (SolidElementKernel).
void SolidPyramidElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    KERNEL.check(view, sink);
}

} // namespace oos::elements
