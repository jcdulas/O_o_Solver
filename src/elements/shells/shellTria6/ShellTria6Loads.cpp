#include "oos/elements/shells/HigherOrderShell.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"

namespace oos::elements
{

void ShellTria6Element::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                          base::DiagnosticSink& sink) const
{
    checkShellPressures(view, pressures, sink);
}

// F_i = ∫ N_i p dA, p interpolated from the corner pressures, along +n or the direction of CID.
void ShellTria6Element::computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                            ElementVectorBlock& block) const
{
    appendHigherOrderShellPressures<QuadraticTriangleShape>(view, pressures, block);
}

void ShellTria6Element::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                         base::DiagnosticSink& sink) const
{
    checkHigherOrderShellThermalLoads<QuadraticTriangleShape>(view, temperatures, sink);
}

void ShellTria6Element::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                           const ElementTemperatures& temperatures, ElementVectorBlock& block) const
{
    appendHigherOrderShellThermalLoads<QuadraticTriangleShape>(view, range, temperatures, block);
}

} // namespace oos::elements
