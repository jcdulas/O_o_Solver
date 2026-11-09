#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

namespace oos::elements
{

// The mass of each Bar2 (BeamMass: lumped or coupled with ρ·A + NSM per unit length; no torsional inertia),
// through the offsets under ModelParams OFFDEF LROFF or ELMZ, at the grid points under ELMOFF, the default
// (beamGridMass).
void Bar2Element::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                 ElementMatrixBlock& block) const
{
    const Columns columns(view);
    appendBeamMass(range, options, TorsionalInertia::None, view.isMassOnOffsets(), block,
                   [&](std::int64_t row) { return resolve(view, columns, row).required(); });
}

} // namespace oos::elements
