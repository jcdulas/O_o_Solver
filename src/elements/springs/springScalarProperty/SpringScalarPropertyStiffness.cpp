#include "oos/elements/springs/springScalarProperty/SpringScalarPropertyElement.hpp"

namespace oos::elements
{

// SpringScalarProperty: K·[1 −1; −1 1] on the scalar points (u1, u2), [K] for a grounded spring.
void SpringScalarPropertyElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                          ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Spring& spring = resolve(view, columns, row).required();
        appendScalarMatrix(block, row, spring.connection, spring.stiffness);
    }
}

} // namespace oos::elements
