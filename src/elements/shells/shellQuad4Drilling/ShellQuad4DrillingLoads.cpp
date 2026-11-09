#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::elements
{

void ShellQuad4DrillingElement::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                          base::DiagnosticSink& sink) const
{
    checkShellPressures(view, pressures, sink);
}

// p(ξ, η) = Σ N_i p_i: F_i = ∫ N_i p dA on the mean plane, along +n or the direction of CID (ShellQuad4DrillingPlate::
// pressureLoads), carried to the displacement systems of G1-G4 (through ZOFFS under ModelParams OFFDEF LROFF or ELMZ
// only: ShellNodeFrames::loadsToDisplacementSystems).
void ShellQuad4DrillingElement::computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                            ElementVectorBlock& block) const
{
    const Columns columns(view);
    for (const ElementPressure& pressure : pressures)
    {
        const Data data = resolve(view, columns, pressure.elementRow).required();
        const PressureDirection direction = pressureDirectionOf(view, pressure, data.geometry.axes);
        if (!direction.problem.empty())
            throw std::invalid_argument(base::concatenate(
                {"ShellQuad4 ", std::to_string(view.elementId(pressure.elementRow)), ": ", direction.problem,
                 " (the loads facet was called on a pressure checkPressureLoads() refuses: a bug "
                 "of O_o_Solver, report it)"}));
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 24> loads = data.frames.loadsToDisplacementSystems(
            plate.pressureLoads(pressure.cornerPressures, direction.inElementFrame), view.areLoadsOnOffsets());
        block.append(pressure.elementRow, data.frames.dofs(), loads);
    }
}

void ShellQuad4DrillingElement::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                         base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        Resolution<Data> data = resolve(view, columns, row);
        if (!data)
            continue; // check() reports it
        Resolution<ShellTemperatureChange> change =
            resolveShellTemperature(view, row, temperatures, data.value->frames.gridRows, data.value->section);
        if (!change)
            sink.report(std::move(change.problem));
    }
}

// The thermal loads of ShellQuad4DrillingPlate for the temperature change of each element, carried to the displacement systems.
void ShellQuad4DrillingElement::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                           const ElementTemperatures& temperatures, ElementVectorBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellTemperatureChange change =
            resolveShellTemperature(view, row, temperatures, data.frames.gridRows, data.section).required();
        if (change.isZero())
            continue;
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        block.append(row, data.frames.dofs(),
                     data.frames.loadsToDisplacementSystems(plate.thermalLoads(change), view.areLoadsOnOffsets()));
    }
}

} // namespace oos::elements
