#include "oos/elements/solids/solidTetra/SolidTetraElement.hpp"

namespace oos::elements
{

// The initial-stress stiffness of the centre stress of buildResponse (SolidElementKernel).
void SolidTetraElement::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    KERNEL.computeDifferentialStiffness(view, range, preload, block);
}

} // namespace oos::elements
