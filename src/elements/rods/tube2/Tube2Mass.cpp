#include "oos/elements/rods/tube2/Tube2Element.hpp"

namespace oos::elements
{

// Mass per unit length ρA + NSM (PTUBE NSM, plus the NSM cards' share of MassOptions), lumped or coupled (rodMass).
void Tube2Element::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                  ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendRodMass(range, options, block, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
