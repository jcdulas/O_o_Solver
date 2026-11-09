#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"

namespace oos::elements
{

// The mass of each Beam2 (BeamMass: lumped or coupled with ρ·A(x) + NSM(x) per unit length and the torsional inertia
// ρ·(I1 + I2) of the section), through the offsets under ModelParams OFFDEF LROFF or ELMZ, at the grid points under
// ELMOFF, the default (beamGridMass).
void Beam2Element::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                  ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendBeamMass(range, options, TorsionalInertia::Polar, view.isMassOnOffsets(), block,
                   [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
