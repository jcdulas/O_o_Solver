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

bool beamUsesReferenceTemperature(const ElementBatchView& view, std::int64_t row,
                                 const BeamData& beam, const ElementTemperatures& temperatures)
{
    return temperatures.hasTemperatureLoad &&
           !meanTemperature(view, row, beam.geometry, temperatures.initial).has_value();
}

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

std::array<double, 4> beamGradientChange(const ElementBatchView& view, std::int64_t row,
                                         const ElementTemperatures& temperatures)
{
    std::array<double, 4> change{};
    if (!temperatures.hasTemperatureLoad)
        return change;
    const std::int32_t elementId = view.elementId(row);
    if (const ElementTemperature* load = temperatures.load.ofElement(elementId))
        change = load->lineGradients;
    // A reference temperature and the temperatures of the grid points are uniform across the section: only a
    // LineElemTemperatureLoad of the initial set has gradients to subtract.
    if (const ElementTemperature* initial = temperatures.initial.ofElement(elementId))
        for (std::size_t k = 0; k < change.size(); ++k)
            change[k] -= initial->lineGradients[k];
    return change;
}

BeamFreeCurvatures beamFreeCurvatures(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                      const ElementTemperatures& temperatures)
{
    BeamFreeCurvatures curvatures;
    const double expansion = beam.material.expansion;
    if (!temperatures.hasTemperatureLoad || expansion == 0.0)
        return curvatures;
    const std::array<double, 4> change = beamGradientChange(view, row, temperatures);
    // κ1 = −α·ΔT'y, κ2 = −α·ΔT'z: the hotter side is the longer one.
    curvatures.plane1 = {-expansion * change[LINE_GRADIENT_Y_AT_A], -expansion * change[LINE_GRADIENT_Y_AT_B]};
    curvatures.plane2 = {-expansion * change[LINE_GRADIENT_Z_AT_A], -expansion * change[LINE_GRADIENT_Z_AT_B]};
    return curvatures;
}

BeamFreeStrains beamFreeStrains(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                const NodalDisplacements& displacements)
{
    BeamFreeStrains freeStrains;
    freeStrains.axial = beamThermalStrain(view, row, beam, displacements);
    if (displacements.temperatures != nullptr)
        freeStrains.curvatures = beamFreeCurvatures(view, row, beam, *displacements.temperatures);
    return freeStrains;
}

std::array<double, 12> beamThermalEndLoads(const BeamData& beam, const BeamFreeStrains& freeStrains)
{
    // d_T: end B from end A in the free deformation. Without curvatures it is the axial expansion alone, as it
    // always was (the same bits for a model without temperature gradients).
    std::array<double, 12> expansion{};
    if (freeStrains.hasCurvatures())
        expansion =
            beamFreeCurvatureDisplacements(beam.section, beam.geometry.length, freeStrains.curvatures,
                                           [&](double position) { return beam.section.at(position).neutralAxis; });
    expansion[AXIAL_B] += freeStrains.axial * beam.geometry.length;
    const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
    return flexibility.stiffness() * expansion;
}

std::array<double, 12> beamGridThermalLoads(const BeamData& beam, const BeamFreeStrains& freeStrains,
                                            bool areLoadsOnOffsets)
{
    std::array<double, 12> endLoads = beamThermalEndLoads(beam, freeStrains);
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
    // f_grid = Tᵀ·f_end (T without the offsets under ElemOffsetsOnly).
    const base::MatN<12> transform =
        areLoadsOnOffsets ? beam.geometry.endTransform() : beam.geometry.rotationTransform();
    return transform.transposed() * endLoads;
}

} // namespace oos::elements
