#pragma once

#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <array>
#include <cstddef>

namespace oos::elements
{

// Whether the material reference temperature supplies the initial temperature of this element. An explicitly
// supplied initial temperature takes precedence, including a zero value.
[[nodiscard]] bool beamUsesReferenceTemperature(const ElementBatchView& view, std::int64_t row,
                                              const BeamData& beam, const ElementTemperatures& temperatures);

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
// the TREF of the IsoMat.
//
// The temperature gradients across the section of a LineElemTemperatureLoad (T'y along the element y axis and T'z
// along z, given at ends A and B and linear between them) bend the element. The temperature of the section is
// T̄ + T'y·(y − y_n) + T'z·(z − z_n), T̄ the temperature at the neutral axis (y_n, z_n), where the mean temperature of
// a section with a linear temperature is. Its free strain α·ΔT(y, z) is that of a plane section, with the curvatures
//     κ1 = v'' = −α·ΔT'y  (plane 1, x-y),     κ2 = w'' = −α·ΔT'z  (plane 2, x-z)
// (a fibre at +y that is hotter is longer: the element bends towards −y), ΔT' the gradient of the load set less that
// of the LineElemTemperatureLoad of the element in the TEMPERATURE(INITIAL) set, if any (a reference temperature has
// no gradient). The free deformation has no stress, so d_T gains the displacements of end B of the element axis in
// it (beamFreeCurvatureDisplacements) and the loads and the recovery stay f_T = K·d_T and K·(d − d_T): the bending
// moments recovered are those of the elastic curvatures, and the stresses at the recovery points, taken from the
// forces, are σ = E·(ε − α·ΔT(y, z)). Not read: the stress-recovery temperatures TCA ... TFB of the card (the
// temperature is linear across the section).

// The free curvatures of a bar or beam under temperature gradients across its section, at ends A and B (linear
// between them): `plane1` the free v'' (bending in the x-y plane), `plane2` the free w'' (x-z plane). `Scalar` is
// double, or MixedScalar where the rates of the design fields go with the values (BeamSizing).
template <typename Scalar>
struct BeamFreeCurvaturesOf
{
    std::array<Scalar, 2> plane1{};
    std::array<Scalar, 2> plane2{};
};
using BeamFreeCurvatures = BeamFreeCurvaturesOf<double>;

// The free deformation of a bar or beam under the initial strains of a subcase: the free axial strain α·ΔT + D/L at
// the neutral axis and the free curvatures of the temperature gradients.
struct BeamFreeStrains
{
    double axial = 0.0;
    BeamFreeCurvatures curvatures{};

    [[nodiscard]] bool hasCurvatures() const noexcept
    {
        return curvatures.plane1[0] != 0.0 || curvatures.plane1[1] != 0.0 || curvatures.plane2[0] != 0.0 ||
               curvatures.plane2[1] != 0.0;
    }
    [[nodiscard]] bool isZero() const noexcept { return axial == 0.0 && !hasCurvatures(); }
};

// The end DOFs [A 1-6, B 1-6] of end B that the free deformation moves.
inline constexpr std::size_t BEAM_END_B_AXIAL = 6;
inline constexpr std::size_t BEAM_END_B_ALONG_Y = 7;
inline constexpr std::size_t BEAM_END_B_ALONG_Z = 8;
inline constexpr std::size_t BEAM_END_B_ABOUT_Y = 10;
inline constexpr std::size_t BEAM_END_B_ABOUT_Z = 11;

// The displacements of end B of the element axis from end A, held, in the free deformation of the curvatures alone,
// over the end DOFs [A 1-6, B 1-6]. With κ1(x) and κ2(x) linear from end A to end B over the length L:
//     v_B  = ∫ (L − x)·κ1 dx = L²·(2·κ1A + κ1B)/6,       θz_B = ∫ κ1 dx = L·(κ1A + κ1B)/2,
//     w_B  = L²·(2·κ2A + κ2B)/6,                          θy_B = −∫ κ2 dx = −L·(κ2A + κ2B)/2,
//     u_B  = ∫ (y_n·κ1 + z_n·κ2) dx,
// u_B the stretch of the element axis, a fibre at (−y_n, −z_n) from the neutral axis the sections turn about (0 for a
// section whose neutral axis is the element axis). `neutralAxisAt(position)` gives (y_n, z_n) at a position (0 at end
// A, 1 at end B); it is linear between two stations, as κ1 and κ2 are, so Simpson's rule on each interval of the
// stations is exact.
template <typename Scalar, typename NeutralAxisAt>
[[nodiscard]] std::array<Scalar, 12> beamFreeCurvatureDisplacements(const BeamSection& section, double length,
                                                                    const BeamFreeCurvaturesOf<Scalar>& curvatures,
                                                                    NeutralAxisAt&& neutralAxisAt)
{
    const std::array<Scalar, 2>& plane1 = curvatures.plane1;
    const std::array<Scalar, 2>& plane2 = curvatures.plane2;
    std::array<Scalar, 12> ends{};
    ends[BEAM_END_B_ALONG_Y] = (plane1[0] * 2.0 + plane1[1]) * (length * length / 6.0);
    ends[BEAM_END_B_ABOUT_Z] = (plane1[0] + plane1[1]) * (0.5 * length);
    ends[BEAM_END_B_ALONG_Z] = (plane2[0] * 2.0 + plane2[1]) * (length * length / 6.0);
    ends[BEAM_END_B_ABOUT_Y] = (plane2[0] + plane2[1]) * (-0.5 * length);
    // y_n·κ1 + z_n·κ2 at a position.
    const auto stretchRateAt = [&](double position)
    {
        const std::array<Scalar, 2> neutralAxis = neutralAxisAt(position);
        const Scalar curvature1 = plane1[0] + (plane1[1] - plane1[0]) * position;
        const Scalar curvature2 = plane2[0] + (plane2[1] - plane2[0]) * position;
        return neutralAxis[0] * curvature1 + neutralAxis[1] * curvature2;
    };
    Scalar stretch{};
    for (std::size_t station = 0; station + 1 < section.count; ++station)
    {
        const double first = section.stations[station].position;
        const double last = section.stations[station + 1].position;
        const double sixth = (last - first) * length / 6.0;
        stretch = stretch + (stretchRateAt(first) + stretchRateAt(0.5 * (first + last)) * 4.0 + stretchRateAt(last)) *
                                sixth;
    }
    ends[BEAM_END_B_AXIAL] = stretch;
    return ends;
}

// ΔT of the element of batch row `row`, or the diagnostic of a missing temperature (field TEMPERATURE(LOAD)).
[[nodiscard]] Resolution<double> resolveBeamTemperatureChange(const ElementBatchView& view, std::int64_t row,
                                                              const BeamData& beam,
                                                              const ElementTemperatures& temperatures);

// ΔT' of the element: the gradients of its LineElemTemperatureLoad in the load set less those of its
// LineElemTemperatureLoad in the initial set, in the order of ElementTemperature::lineGradients (T'y at A and B, T'z
// at A and B); zeros for an element without that card and without a temperature load.
[[nodiscard]] std::array<double, 4> beamGradientChange(const ElementBatchView& view, std::int64_t row,
                                                       const ElementTemperatures& temperatures);

// The free curvatures −α·ΔT' of the element (zeros without a temperature load, a gradient or an expansion
// coefficient).
[[nodiscard]] BeamFreeCurvatures beamFreeCurvatures(const ElementBatchView& view, std::int64_t row,
                                                    const BeamData& beam, const ElementTemperatures& temperatures);

// The free axial strain α·ΔT + D/L of the element under the initial strains of the displacements (0 without them;
// α·ΔT 0 without a temperature load, D/L 0 without an elongation). Throws std::invalid_argument for an element whose
// temperature resolveBeamTemperatureChange refuses.
[[nodiscard]] double beamThermalStrain(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                       const NodalDisplacements& displacements);

// D/L of the element: the free axial strain of its enforced elongation (0 when the DEFORM set gives it none).
[[nodiscard]] double beamDeformationStrain(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                           const ElementTemperatures& temperatures);

// The free deformation of the element under the initial strains of the displacements: beamThermalStrain and
// beamFreeCurvatures (zero without them).
[[nodiscard]] BeamFreeStrains beamFreeStrains(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                              const NodalDisplacements& displacements);

// K·d_T over the end DOFs (pin flags not condensed): the equivalent end loads of the free deformation.
[[nodiscard]] std::array<double, 12> beamThermalEndLoads(const BeamData& beam, const BeamFreeStrains& freeStrains);
// The thermal load over the grid DOFs [GA 1-6, GB 1-6] (pin flags condensed, CD systems applied), through the offsets
// when `areLoadsOnOffsets` (ModelParams OffsetDefinition LoadRecoveryOffsets, ElemOffsetZ), at the grid points by the
// rotations alone otherwise (ElemOffsetsOnly: the thermal loads leave the offsets out).
[[nodiscard]] std::array<double, 12> beamGridThermalLoads(const BeamData& beam, const BeamFreeStrains& freeStrains,
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
        const BeamFreeStrains freeStrains{strain, beamFreeCurvatures(view, row, beam, temperatures)};
        if (freeStrains.isZero())
            continue;
        const std::array<double, 12> values = beamGridThermalLoads(beam, freeStrains, view.areLoadsOnOffsets());
        const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
        block.append(row, dofs, values);
    }
}

} // namespace oos::elements
