#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"

namespace oos::elements
{

void Rod2InlineSectionElement::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                               base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkRodThermalLoads(view, temperatures, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

// The axial force of the free deformation of each rod, E A α ΔT + (E A / L)·D (rodThermalForce: its temperatures
// and its enforced elongation), outward at both ends.
void Rod2InlineSectionElement::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                                 const ElementTemperatures& temperatures,
                                                 ElementVectorBlock& block) const
{
    const Columns columns(view);
    appendRodThermalLoads(view, range, temperatures, block,
                          [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
