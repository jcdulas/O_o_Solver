#include "oos/elements/masses/massScalarProperty/MassScalarPropertyElement.hpp"

namespace oos::elements
{

// MassScalarProperty: M·[1 −1; −1 1] on (u1, u2), [M] for a grounded end; lumped and coupled alike.
void MassScalarPropertyElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& /*options*/,
                                   ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Mass& mass = resolve(view, columns, row).required();
        appendScalarMatrix(block, row, mass.connection, mass.mass);
    }
}

} // namespace oos::elements
