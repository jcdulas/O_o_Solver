#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/springs/bushing/BushingElement.hpp"

namespace oos::elements
{

// BushProp M lumped on the translations (BushingElement.hpp): M/2 at GA and at GB, M at GA for a grounded Bushing. A
// mass m·I on the translations is the same in every displacement system (Rᵀ·m·I·R = m·I), and the lumped and coupled
// forms coincide. A Bushing without mass appends nothing.
void BushingElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& /*options*/,
                                  ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Bush& bush = resolve(view, columns, row).required();
        if (bush.mass == 0.0)
            continue;
        const double share = bush.isGrounded ? bush.mass : 0.5 * bush.mass;
        base::MatN<6, 6> lumped;
        for (std::size_t c = 0; c < 3; ++c)
            lumped(c, c) = share;
        if (bush.isGrounded)
        {
            appendMatrix(block, row, gridPointDofs(bush.rowA), lumped);
            continue;
        }
        base::MatN<12, 12> both;
        for (std::size_t c = 0; c < 3; ++c)
        {
            both(c, c) = share;
            both(6 + c, 6 + c) = share;
        }
        const std::array<ElementDof, 6> dofsA = gridPointDofs(bush.rowA);
        const std::array<ElementDof, 6> dofsB = gridPointDofs(bush.rowB);
        std::array<ElementDof, 12> dofs;
        for (std::size_t k = 0; k < 6; ++k)
        {
            dofs[k] = dofsA[k];
            dofs[6 + k] = dofsB[k];
        }
        appendMatrix(block, row, dofs, both);
    }
}

} // namespace oos::elements
