#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"

namespace oos::elements
{

void ShellQuad8Element::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                          base::DiagnosticSink& sink) const
{
    checkShellPressures(view, pressures, sink);
}

// F_i = ∫ N_i p dA, p interpolated from the corner pressures, along +n or the direction of CID.
void ShellQuad8Element::computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                            ElementVectorBlock& block) const
{
    appendHigherOrderShellPressures<SerendipityQuadShape>(view, pressures, block);
}

void ShellQuad8Element::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                         base::DiagnosticSink& sink) const
{
    checkHigherOrderShellThermalLoads<SerendipityQuadShape>(view, temperatures, sink);
}

void ShellQuad8Element::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                           const ElementTemperatures& temperatures, ElementVectorBlock& block) const
{
    appendHigherOrderShellThermalLoads<SerendipityQuadShape>(view, range, temperatures, block);
}

} // namespace oos::elements
