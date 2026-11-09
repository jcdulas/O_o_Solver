#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"

namespace oos::elements
{

// A quarter of the panel's mass, (ρ T + NSM) A, on the translations of each corner (lumped and coupled alike: a
// design choice of this project), with the NSM cards' share.
void ShearPanel4Element::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                        ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        appendMatrix(block, row, data.dofs(), data.toDisplacementSystems(data.panel.mass(added)));
    }
}

} // namespace oos::elements
