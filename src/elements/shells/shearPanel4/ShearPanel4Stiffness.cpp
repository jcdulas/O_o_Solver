#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"

namespace oos::elements
{

// K = b bᵀ / f of the shear flows and the rods of F1, F2 (ShearPanel4Panel), in the displacement systems of G1-G4.
void ShearPanel4Element::computeStiffness(const ElementBatchView& view, ElementRange range,
                                             ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        appendMatrix(block, row, data.dofs(), data.toDisplacementSystems(data.panel.stiffness()));
    }
}

} // namespace oos::elements
