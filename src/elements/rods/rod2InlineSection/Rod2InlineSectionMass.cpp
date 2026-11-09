#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"

namespace oos::elements
{

// Mass per unit length ρA + NSM (CONROD NSM, plus the NSM cards' share of MassOptions), lumped or coupled (rodMass).
void Rod2InlineSectionElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                         ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendRodMass(range, options, block, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
