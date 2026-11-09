#include "oos/elements/solids/solidHexa/SolidHexaElement.hpp"

namespace oos::elements
{

// The initial-stress stiffness of the centre stress of buildResponse (SolidElementKernel).
void SolidHexaElement::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                   const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    KERNEL.computeDifferentialStiffness(view, range, preload, block);
}

} // namespace oos::elements
