#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

namespace oos::elements
{

// The lumped (HRZ) or coupled mass of ShellTria3DrillingPlate::mass, ρ t + NSM per unit area (PSHELL NSM and the NSM cards'
// share), carried to the displacement systems of G1-G3: through the offset of ZOFFS under ModelParams OFFDEF LROFF or
// ELMZ, at the grid points under ELMOFF, the default (ShellNodeFrames::massToDisplacementSystems).
void ShellTria3DrillingElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                   ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        appendMatrix(
            block, row, data.frames.dofs(),
            data.frames.massToDisplacementSystems(plate.mass(options.formulation, added), view.isMassOnOffsets()));
    }
}

} // namespace oos::elements
