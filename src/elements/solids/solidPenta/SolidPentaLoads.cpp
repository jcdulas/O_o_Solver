#include "oos/elements/solids/solidPenta/SolidPentaElement.hpp"

namespace oos::elements
{

// FacePressureLoad on a face of the element: the face from face_nodes, uniform or corner pressures, consistent nodal
// loads along the inward normal (SolidPressure).
void SolidPentaElement::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                          base::DiagnosticSink& sink) const
{
    KERNEL.checkPressureLoads(view, pressures, sink);
}

void SolidPentaElement::computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                            ElementVectorBlock& block) const
{
    KERNEL.computePressureLoads(view, pressures, block);
}

// Thermal loads of the grid point temperatures (SolidElementKernel: f = ∫ Bᵀ·D·ε_T dV).
void SolidPentaElement::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                         base::DiagnosticSink& sink) const
{
    KERNEL.checkThermalLoads(view, temperatures, sink);
}

void SolidPentaElement::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                           const ElementTemperatures& temperatures, ElementVectorBlock& block) const
{
    KERNEL.computeThermalLoads(view, range, temperatures, block);
}

} // namespace oos::elements
