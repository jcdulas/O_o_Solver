#include "oos/elements/fasteners/weld/WeldElement.hpp"

namespace oos::elements
{

// K = Mᵀ·(Tᵀ·K_beam·T)·M over the grid points of the pierced elements (WeldElement.hpp, FastenerConnection).
void WeldElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                       ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Weld weld = resolve(view, columns, row).required();
        const base::MatN<12> overPoints = congruence(weld.beamStiffness, weld.toBeamEnds);
        block.append(row, weld.connection.dofs(), weld.connection.condense(overPoints));
    }
}

// ρ·A·L lumped half on the translations of A and half on those of B (basic), carried to the pierced elements: Mᵀ·m·M.
void WeldElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& /*options*/,
                                  ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Weld weld = resolve(view, columns, row).required();
        const double mass = weld.material.density * weld.section.stations[0].area * weld.distance;
        if (mass == 0.0)
            continue;
        base::MatN<12> lumped;
        for (std::size_t c = 0; c < 3; ++c)
        {
            lumped(c, c) = 0.5 * mass;
            lumped(6 + c, 6 + c) = 0.5 * mass;
        }
        block.append(row, weld.connection.dofs(), weld.connection.condense(lumped));
    }
}

} // namespace oos::elements
