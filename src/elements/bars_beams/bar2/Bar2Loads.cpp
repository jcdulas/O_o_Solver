#include "oos/elements/bars_beams/BeamThermal.hpp"
#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

namespace oos::elements
{

void Bar2Element::checkLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                    base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkBeamLineLoads(view, loads, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

// BeamDistributedLoad: the exact fixed-end forces of the bar under the load with their sign changed (BeamFlexibility),
// the pin flags condensed, carried to the displacement systems of GA and GB.
void Bar2Element::computeLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                      ElementVectorBlock& block) const
{
    const Columns columns(view);
    appendBeamLineLoads(view, loads, block, [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

void Bar2Element::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                       base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkBeamThermalLoads(view, temperatures, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

// The load of the free thermal expansion, K·d_T, the pin flags condensed, carried to the displacement systems of GA
// and GB (BeamThermal).
void Bar2Element::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                         const ElementTemperatures& temperatures, ElementVectorBlock& block) const
{
    const Columns columns(view);
    appendBeamThermalLoads(view, range, temperatures, block,
                           [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
