#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::elements
{

void ShellTria3DrillingElement::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                          base::DiagnosticSink& sink) const
{
    checkShellPressures(view, pressures, sink);
}

// The pressure linear between P1, P2, P3 (P4 of the FacePressureLoad plays no part on a triangle),
// F_i = A_e (2 p_i + p_j + p_k)/12 along +n or the direction of CID (ShellTria3DrillingPlate::pressureLoads), carried to the
// displacement systems of G1-G3.
void ShellTria3DrillingElement::computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                            ElementVectorBlock& block) const
{
    const Columns columns(view);
    for (const ElementPressure& pressure : pressures)
    {
        const Data data = resolve(view, columns, pressure.elementRow).required();
        const PressureDirection direction = pressureDirectionOf(view, pressure, data.geometry.axes);
        if (!direction.problem.empty())
            throw std::invalid_argument(base::concatenate(
                {"ShellTria3 ", std::to_string(view.elementId(pressure.elementRow)), ": ", direction.problem,
                 " (the loads facet was called on a pressure checkPressureLoads() refuses: a bug "
                 "of O_o_Solver, report it)"}));
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 3> corners{pressure.cornerPressures[0], pressure.cornerPressures[1],
                                            pressure.cornerPressures[2]};
        block.append(pressure.elementRow, data.frames.dofs(),
                     data.frames.loadsToDisplacementSystems(plate.pressureLoads(corners, direction.inElementFrame),
                                                            view.areLoadsOnOffsets()));
    }
}

void ShellTria3DrillingElement::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
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

void ShellTria3DrillingElement::computeThermalLoads(const ElementBatchView& view, ElementRange range,
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
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        block.append(row, data.frames.dofs(),
                     data.frames.loadsToDisplacementSystems(plate.thermalLoads(change), view.areLoadsOnOffsets()));
    }
}

} // namespace oos::elements
