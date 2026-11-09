#include "oos/elements/solids/solidHexa/SolidHexaElement.hpp"

namespace oos::elements
{

// The lumped (HRZ) or coupled mass ∫ ρ NᵀN dV in the displacement systems of the grid points (SolidElementKernel).
void SolidHexaElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                  ElementMatrixBlock& block) const
{
    KERNEL.computeMass(view, range, options, block);
}

} // namespace oos::elements
