#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"

namespace oos::elements
{

// The mean shear flow of the preload (ShearPanel4Panel::meanFlow) on ShearPanel4Panel::differentialStiffness, in
// the displacement systems of G1-G4.
void ShearPanel4Element::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                         const NodalDisplacements& preload,
                                                         ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const double flow = data.panel.meanFlow(data.localDisplacements(preload));
        appendMatrix(block, row, data.dofs(), data.toDisplacementSystems(data.panel.differentialStiffness(flow)));
    }
}

} // namespace oos::elements
