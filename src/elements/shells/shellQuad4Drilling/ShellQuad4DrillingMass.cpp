#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

namespace oos::elements
{

// The lumped (row sums) or coupled mass of ShellQuad4DrillingPlate::mass, ρ t + NSM per unit area (PSHELL NSM and the NSM cards'
// share), carried to the displacement systems of G1-G4: through the offset of ZOFFS under ModelParams OFFDEF LROFF or
// ELMZ, at the grid points under ELMOFF, the default (ShellNodeFrames::massToDisplacementSystems).
void ShellQuad4DrillingElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                   ElementMatrixBlock& block) const
{
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        // The mass reads the geometry, the density and the thicknesses only: no plate is built
        // (ShellQuad4DrillingPlate::massOf).
        appendMatrix(
            block, row, data.frames.dofs(),
            data.frames.massToDisplacementSystems(
                ShellQuad4DrillingPlate::massOf(data.geometry, data.section, data.thickness, options.formulation, added),
                view.isMassOnOffsets()));
    }
}

} // namespace oos::elements
