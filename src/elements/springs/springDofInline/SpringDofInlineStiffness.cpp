#include "oos/elements/springs/springDofInline/SpringDofInlineElement.hpp"

namespace oos::elements
{

// SpringDofInline: K·[1 −1; −1 1] on (u1, u2) in the displacement systems of the grid points, [K] for a grounded spring.
// No frame transformation: the spring acts on the DOF components as the user names them.
void SpringDofInlineElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                          ElementMatrixBlock& block) const
{
    const Columns columns(view.columns());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Spring& spring = resolve(view, columns, row).required();
        appendScalarMatrix(block, row, spring.connection, spring.stiffness);
    }
}

} // namespace oos::elements
