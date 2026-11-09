#include "oos/elements/solids/solidTetra/SolidTetraElement.hpp"

namespace oos::elements
{

// The lumped (HRZ) or coupled mass ∫ ρ NᵀN dV in the displacement systems of the grid points (SolidElementKernel).
void SolidTetraElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                   ElementMatrixBlock& block) const
{
    KERNEL.computeMass(view, range, options, block);
}

} // namespace oos::elements
