#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

namespace oos::elements
{

// The membrane resultants of the preload at the centre (ShellQuad4DrillingPlate::placeStates, the thermal resultant subtracted),
// then ShellQuad4DrillingPlate::differentialStiffness in the element frame, carried to the displacement systems of G1-G4 through
// the ZOFFS offset (ShellNodeFrames). The ShellQuad4Drilling takes the same matrix (its membrane gives the resultants).
void ShellQuad4DrillingElement::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 24> q = data.frames.localDisplacements(preload);
        const ShellTemperatureChange temperature =
            preload.temperatures == nullptr
                ? ShellTemperatureChange{}
                : resolveShellTemperature(view, row, *preload.temperatures, data.frames.gridRows, data.section)
                      .required();
        const ShellPlaceState centre = plate.placeStates(q, data.gridIds, temperature)[0];
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.toDisplacementSystems(plate.differentialStiffness(centre.membraneForce)));
    }
}

} // namespace oos::elements
