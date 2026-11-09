#include "oos/elements/solids/SolidRecovery.hpp"
#include "oos/elements/solids/solidPenta/SolidPentaElement.hpp"

namespace oos::elements
{

std::vector<std::string> SolidPentaElement::outputColumns(ElementOutput output) const
{
    return solidOutputColumns(output);
}

// Stresses or strains at the centre and the corners, in the material system CORDM (SolidRecovery).
void SolidPentaElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                               const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    KERNEL.recover(view, range, output, displacements, block);
}

} // namespace oos::elements
