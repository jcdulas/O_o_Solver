#include "oos/elements/springs/springScalarInline/SpringScalarInlineElement.hpp"

namespace oos::elements
{

// SpringScalarInline: K·[1 −1; −1 1] on the scalar points (u1, u2), [K] for a grounded spring.
void SpringScalarInlineElement::computeStiffness(const ElementBatchView& view, ElementRange range,
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
