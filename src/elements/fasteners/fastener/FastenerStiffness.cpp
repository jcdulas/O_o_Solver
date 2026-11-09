#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/fasteners/fastener/FastenerElement.hpp"

namespace oos::elements
{

// K = Mᵀ·(Hᵀ·diag(KT, KR)·H)·M over the grid points of the pierced elements (FastenerElement.hpp,
// FastenerConnection).
void FastenerElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                           ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Fastener fastener = resolve(view, columns, row).required();
        const base::MatN<12> spring = diagonalCongruence(fastener.stiffness, fastener.relativeMotion);
        block.append(row, fastener.connection.dofs(), fastener.connection.condense(spring));
    }
}

// FastenerProp MASS, half on the translations of A and half on those of B (basic), carried to the pierced elements:
// Mᵀ·m·M.
void FastenerElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& /*options*/,
                                      ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Fastener fastener = resolve(view, columns, row).required();
        if (fastener.mass == 0.0)
            continue;
        base::MatN<12> lumped;
        for (std::size_t c = 0; c < 3; ++c)
        {
            lumped(c, c) = 0.5 * fastener.mass;
            lumped(6 + c, 6 + c) = 0.5 * fastener.mass;
        }
        block.append(row, fastener.connection.dofs(), fastener.connection.condense(lumped));
    }
}

} // namespace oos::elements
