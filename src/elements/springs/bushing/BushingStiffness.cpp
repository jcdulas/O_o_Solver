#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/springs/bushing/BushingElement.hpp"

namespace oos::elements
{

namespace
{

// The first six columns of H: the map of a grounded Bushing, whose GB DOFs do not exist.
base::MatN<6, 6> columnsOfA(const base::MatN<6, 12>& h)
{
    base::MatN<6, 6> result;
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            result(i, j) = h(i, j);
    return result;
}

} // namespace

// K = Hᵀ·diag(K1..K6)·H over (GA 1-6, GB 1-6), or over GA 1-6 alone for a grounded Bushing.
void BushingElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                       ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Bush& bush = resolve(view, columns, row).required();
        const std::array<ElementDof, 6> dofsA = gridPointDofs(bush.rowA);
        if (bush.isGrounded)
        {
            appendMatrix(block, row, dofsA, diagonalCongruence(bush.stiffness, columnsOfA(bush.relativeMotion)));
            continue;
        }
        const std::array<ElementDof, 6> dofsB = gridPointDofs(bush.rowB);
        std::array<ElementDof, 12> dofs;
        for (std::size_t k = 0; k < 6; ++k)
        {
            dofs[k] = dofsA[k];
            dofs[6 + k] = dofsB[k];
        }
        appendMatrix(block, row, dofs, diagonalCongruence(bush.stiffness, bush.relativeMotion));
    }
}

} // namespace oos::elements
