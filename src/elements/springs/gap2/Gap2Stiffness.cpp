#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/springs/gap2/Gap2Element.hpp"

namespace oos::elements
{

// K = Hᵀ·diag(k_x, k_t, k_t)·H over the translations of GA and GB (Gap2Element.hpp).
void Gap2Element::computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Gap& gap = resolve(view, columns, row).required();
        const std::array<ElementDof, 6> dofs{ElementDof::grid(gap.rowA, 1), ElementDof::grid(gap.rowA, 2),
                                             ElementDof::grid(gap.rowA, 3), ElementDof::grid(gap.rowB, 1),
                                             ElementDof::grid(gap.rowB, 2), ElementDof::grid(gap.rowB, 3)};
        appendMatrix(block, row, dofs, diagonalCongruence(gap.stiffness, gap.relativeMotion));
    }
}

} // namespace oos::elements
