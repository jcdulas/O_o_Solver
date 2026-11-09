#pragma once

#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <array>

namespace oos::elements
{

// The thermal load of a bar or beam in linear statics (TEMPERATURE(LOAD), LineElemTemperatureLoad): the mean
// temperature change of the element, ΔT = T̄ − T̄0, gives the free axial strain ε_T = α·ΔT, and the thermal load is the
// force that holds the free expansion,
//     f_T = K·d_T,    d_T the end displacements of the free expansion (end B moving ε_T·L along the axis from end A),
// K the end stiffness of BeamFlexibility (the shear-flexible, tapered section of the element: f_T is the axial force
// E·A·ε_T of a uniform bar, its flexibility-weighted value along a tapered one), the pin flags condensed (a released
// axial DOF takes no thermal load) and carried to the grid points by the offsets and CD systems, as the line loads
// are. The recovery subtracts the free expansion: the end forces are K·(d − d_T), so σ = E·(ε − α·ΔT).
//
// An enforced elongation D of the element (DEFORM) is a free axial strain D/L added to α·ΔT: end B moves
// (α·ΔT + D/L)·L = α·ΔT·L + D from end A, the loads and the recovery are those above (the axial force of a uniform
// bar E·A·(ε − α·ΔT − D/L)). A subcase with a DEFORM and no TEMPERATURE(LOAD) has D/L alone.
//
// T̄: a LineElemTemperatureLoad of the element (temperatureTable gives TA and TB − TA: T̄ = (TA + TB)/2), else the mean
// of the temperatures of GA and GB (TEMP, DefaultTemperatureLoad); T̄0 the same from the TEMPERATURE(INITIAL) set, else
// the TREF of the IsoMat. Not available from temperatureTable: the gradients TP1, TP2 of LineElemTemperatureLoad (the
// thermal bending, which O_o_Reader does not give yet) and the stress-recovery temperatures TCA ... TFB.

// ΔT of the element of batch row `row`, or the diagnostic of a missing temperature (field TEMPERATURE(LOAD)).
[[nodiscard]] Resolution<double> resolveBeamTemperatureChange(const ElementBatchView& view, std::int64_t row,
                                                              const BeamData& beam,
                                                              const ElementTemperatures& temperatures);

// The free axial strain α·ΔT + D/L of the element under the initial strains of the displacements (0 without them;
// α·ΔT 0 without a temperature load, D/L 0 without an elongation). Throws std::invalid_argument for an element whose
// temperature resolveBeamTemperatureChange refuses.
[[nodiscard]] double beamThermalStrain(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                       const NodalDisplacements& displacements);

// D/L of the element: the free axial strain of its enforced elongation (0 when the DEFORM set gives it none).
[[nodiscard]] double beamDeformationStrain(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                           const ElementTemperatures& temperatures);

// K·d_T over the end DOFs (pin flags not condensed): the equivalent end loads of the free expansion.
[[nodiscard]] std::array<double, 12> beamThermalEndLoads(const BeamData& beam, double thermalStrain);
// The thermal load over the grid DOFs [GA 1-6, GB 1-6] (pin flags condensed, CD systems applied), through the offsets
// when `areLoadsOnOffsets` (MDLPRM OFFDEF LROFF, ELMZ), at the grid points by the rotations alone otherwise (ELMOFF:
// the thermal loads leave the offsets out).
[[nodiscard]] std::array<double, 12> beamGridThermalLoads(const BeamData& beam, double thermalStrain,
                                                          bool areLoadsOnOffsets);

// The batch loops of the two families; `resolve(row)` returns the Resolution<BeamData> of a row.
template <typename Resolve>
void checkBeamThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink, Resolve&& resolve)
{
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        if (!temperatures.hasTemperatureLoad)
            return; // the enforced elongations alone: nothing to resolve
        auto beam = resolve(row);
        if (!beam)
            continue; // check() reports the element
        Resolution<double> change = resolveBeamTemperatureChange(view, row, *beam.value, temperatures);
        if (!change)
            sink.report(std::move(change.problem));
    }
}

template <typename Resolve>
void appendBeamThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                            ElementVectorBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const BeamData beam = resolve(row);
        const double thermalStrain =
            temperatures.hasTemperatureLoad
                ? beam.material.expansion * resolveBeamTemperatureChange(view, row, beam, temperatures).required()
                : 0.0;
        const double strain = thermalStrain + beamDeformationStrain(view, row, beam, temperatures);
        if (strain == 0.0)
            continue;
        const std::array<double, 12> values = beamGridThermalLoads(beam, strain, view.areLoadsOnOffsets());
        const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
        block.append(row, dofs, values);
    }
}

} // namespace oos::elements
