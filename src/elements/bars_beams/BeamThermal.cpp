#include "oos/elements/bars_beams/BeamThermal.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/bars_beams/BeamFlexibility.hpp"
#include "oos/elements/bars_beams/PinRelease.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace oos::elements
{

namespace
{

// The axial DOF of end B among the end DOFs [A 1-6, B 1-6].
constexpr std::size_t AXIAL_B = 6;

// T̄ of the element in one temperature set: its LineElemTemperatureLoad (TA and TB − TA), else the mean of its grid
// points; nothing when the set gives neither.
std::optional<double> meanTemperature(const ElementBatchView& view, std::int64_t row, const BeamGeometry& geometry,
                                      const TemperatureSet& set)
{
    if (const ElementTemperature* element = set.ofElement(view.elementId(row)))
        return element->mean + 0.5 * element->gradient;
    const std::optional<double> atA = set.atGridRow(geometry.rowA);
    const std::optional<double> atB = set.atGridRow(geometry.rowB);
    if (!atA || !atB)
        return std::nullopt;
    return 0.5 * (*atA + *atB);
}

} // namespace

Resolution<double> resolveBeamTemperatureChange(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                                const ElementTemperatures& temperatures)
{
    using Result = Resolution<double>;
    const std::optional<double> load = meanTemperature(view, row, beam.geometry, temperatures.load);
    if (!load)
    {
        const std::span<const std::int32_t> ids = view.snapshot().nodes().ids();
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "TemperatureSelection(LOAD)",
            base::concatenate(
                {"expected a LineElemTemperatureLoad of the element or temperatures at its grid points ",
                 std::to_string(ids[static_cast<std::size_t>(beam.geometry.rowA)]), " and ",
                 std::to_string(ids[static_cast<std::size_t>(beam.geometry.rowB)]),
                 " in the thermal load set, found none; give a LineElemTemperatureLoad, NodeTemperatureLoad or "
                 "DefaultTemperatureLoad"})));
    }
    double initial = beam.material.referenceTemperature;
    if (!temperatures.initial.isEmpty())
        initial = meanTemperature(view, row, beam.geometry, temperatures.initial).value_or(initial);
    return Result::success(*load - initial);
}

double beamDeformationStrain(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                             const ElementTemperatures& temperatures)
{
    const double elongation = temperatures.elongationOf(view.elementId(row));
    return elongation == 0.0 ? 0.0 : elongation / beam.geometry.length;
}

double beamThermalStrain(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                         const NodalDisplacements& displacements)
{
    if (displacements.temperatures == nullptr)
        return 0.0;
    const ElementTemperatures& temperatures = *displacements.temperatures;
    const double thermalStrain =
        !temperatures.hasTemperatureLoad || beam.material.expansion == 0.0
            ? 0.0
            : beam.material.expansion * resolveBeamTemperatureChange(view, row, beam, temperatures).required();
    // + 0 without an elongation: the bits of α·ΔT alone, as without a DEFORM set.
    return thermalStrain + beamDeformationStrain(view, row, beam, temperatures);
}

std::array<double, 12> beamThermalEndLoads(const BeamData& beam, double thermalStrain)
{
    std::array<double, 12> expansion{};
    expansion[AXIAL_B] = thermalStrain * beam.geometry.length;
    const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
    return flexibility.stiffness() * expansion;
}

std::array<double, 12> beamGridThermalLoads(const BeamData& beam, double thermalStrain, bool areLoadsOnOffsets)
{
    std::array<double, 12> endLoads = beamThermalEndLoads(beam, thermalStrain);
    if (beam.geometry.hasPinFlags())
    {
        const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
        const std::optional<PinRelease> release = PinRelease::make(flexibility.stiffness(), beam.geometry.isReleased);
        if (!release)
            throw std::invalid_argument(
                "bar or beam: pin flags on DOFs without stiffness (the caller skipped check(): a bug of O_o_Solver, "
                "report it)");
        endLoads = release->condenseLoads(endLoads);
    }
    // f_grid = Tᵀ·f_end (T without the offsets under ELMOFF).
    const base::MatN<12> transform =
        areLoadsOnOffsets ? beam.geometry.endTransform() : beam.geometry.rotationTransform();
    return transform.transposed() * endLoads;
}

} // namespace oos::elements
