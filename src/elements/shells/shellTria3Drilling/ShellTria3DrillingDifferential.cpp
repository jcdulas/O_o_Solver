#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

namespace oos::elements
{

// The membrane resultants of the preload (ShellTria3DrillingPlate::centreState, the thermal resultant subtracted), then
// ShellTria3DrillingPlate::differentialStiffness in the element frame, carried to the displacement systems of G1-G3 through the
// ZOFFS offset (ShellNodeFrames). The ShellTria3Drilling takes the same matrix.
void ShellTria3DrillingElement::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 18> q = data.frames.localDisplacements(preload);
        const ShellTemperatureChange temperature =
            preload.temperatures == nullptr
                ? ShellTemperatureChange{}
                : resolveShellTemperature(view, row, *preload.temperatures, data.frames.gridRows, data.section)
                      .required();
        const ShellPlaceState centre = plate.centreState(q, temperature);
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.toDisplacementSystems(plate.differentialStiffness(centre.membraneForce)));
    }
}

} // namespace oos::elements
