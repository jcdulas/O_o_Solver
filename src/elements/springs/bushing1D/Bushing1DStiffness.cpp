#include "oos/elements/springs/bushing1D/Bushing1DElement.hpp"

#include <vector>

namespace oos::elements
{

// K = K·hᵀ·h over the translations of GA and GB (Bushing1DElement.hpp).
void Bushing1DElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                         ElementMatrixBlock& block) const
{
    const Columns columns(view);
    std::vector<double> matrix; // at most 6 x 6
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Bush& bush = resolve(view, columns, row).required();
        std::array<ElementDof, 6> storage;
        const std::span<const ElementDof> dofs = translationDofs(bush, storage);
        const std::size_t n = dofs.size();
        matrix.assign(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                matrix[i * n + j] = bush.stiffness * bush.axialMap[i] * bush.axialMap[j];
        block.append(row, dofs, matrix);
    }
}

// BushProp1D M lumped on the translations: M/2 at GA and at GB, M at GA when grounded (Bushing1DElement.hpp); m·I is
// the same in every displacement system.
void Bushing1DElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& /*options*/,
                                    ElementMatrixBlock& block) const
{
    const Columns columns(view);
    std::vector<double> matrix;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Bush& bush = resolve(view, columns, row).required();
        if (bush.mass == 0.0)
            continue;
        std::array<ElementDof, 6> storage;
        const std::span<const ElementDof> dofs = translationDofs(bush, storage);
        const std::size_t n = dofs.size();
        const double share = bush.isGrounded ? bush.mass : 0.5 * bush.mass;
        matrix.assign(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            matrix[i * n + i] = share;
        block.append(row, dofs, matrix);
    }
}

} // namespace oos::elements
