#include "oos/elements/bars_beams/BeamFormulation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/bars_beams/BeamFlexibility.hpp"
#include "oos/elements/bars_beams/PinRelease.hpp"

#include <optional>
#include <stdexcept>
#include <variant>

namespace oos::elements
{

namespace
{

// Tᵀ·K·T, made exactly symmetric: the product rounds (i, j) and (j, i) differently, and the assembler may read one
// triangle only.
base::MatN<12> symmetricCongruence(const base::MatN<12>& k, const base::MatN<12>& t)
{
    base::MatN<12> result = congruence(k, t);
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = i + 1; j < 12; ++j)
            result(i, j) = result(j, i) = 0.5 * (result(i, j) + result(j, i));
    return result;
}

} // namespace

bool hasStablePinFlags(const BeamData& beam)
{
    if (!beam.geometry.hasPinFlags())
        return true;
    const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
    return PinRelease::make(flexibility.stiffness(), beam.geometry.isReleased).has_value();
}

base::Diagnostic unstablePinFlags(const ElementBatchView& view, std::int64_t row, const BeamGeometry& geometry)
{
    bool isEndAPinned = false;
    for (std::size_t i = 0; i < 6; ++i)
        isEndAPinned = isEndAPinned || geometry.isReleased[i];
    return elementProblem(view, row, problem_code::INVALID_FIELD, isEndAPinned ? "pinFlagsA" : "pinFlagsB",
                          base::concatenate({"expected pin flags on degrees of freedom the element has stiffness in, "
                                             "without leaving a mechanism of the element (a ",
                                             view.card(),
                                             " must have stiffness in each released DOF), "
                                             "found a released DOF without stiffness or a free motion of the "
                                             "element; correct Bar2.pinFlagsA and Bar2.pinFlagsB or the "
                                             "section"}));
}

base::MatN<12> beamStiffness(const BeamData& beam)
{
    const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
    base::MatN<12> endStiffness = flexibility.stiffness();
    if (beam.geometry.hasPinFlags())
    {
        const std::optional<PinRelease> release = PinRelease::make(endStiffness, beam.geometry.isReleased);
        if (!release)
            throw std::invalid_argument(
                "bar or beam: pin flags on DOFs without stiffness (the caller skipped check(): a bug of O_o_Solver, "
                "report it)");
        endStiffness = release->condensedStiffness();
    }
    return symmetricCongruence(endStiffness, beam.geometry.endTransform());
}

base::MatN<12> beamGridMass(const BeamData& beam, double addedMassPerLength, MassFormulation formulation,
                            TorsionalInertia torsionalInertia, bool isMassOnOffsets)
{
    auto endMass = beamMass(beam.section, beam.material.density, addedMassPerLength, beam.geometry.length,
                            formulation, torsionalInertia);
    if (beam.geometry.hasPinFlags())
    {
        const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
        const auto release = PinRelease::make(flexibility.stiffness(), beam.geometry.isReleased).value();
        base::MatN<12> completion;
        for (std::size_t j = 0; j < 12; ++j)
        {
            std::array<double, 12> unit{};
            unit[j] = 1.0;
            const auto displacement = release.completeDisplacements(unit, {});
            for (std::size_t i = 0; i < 12; ++i)
                completion(i, j) = displacement[i];
        }
        endMass = symmetricCongruence(endMass, completion);
    }
    return symmetricCongruence(endMass,
                               isMassOnOffsets ? beam.geometry.endTransform() : beam.geometry.rotationTransform());
}

std::array<double, 12> beamGridLoads(const BeamData& beam, std::span<const BeamLineLoad> loads, bool areLoadsOnOffsets)
{
    const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
    std::array<double, 12> endLoads = flexibility.equivalentLoads(loads);
    if (beam.geometry.hasPinFlags())
    {
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

std::vector<BeamLineLoad> resolveLineLoadsOrThrow(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                                    const BeamGeometry& geometry, std::int64_t row)
{
    std::vector<BeamLineLoad> resolved;
    resolved.reserve(loads.size());
    for (const ElementLineLoad& load : loads)
    {
        if (row >= 0 && load.elementRow != row)
            continue;
        auto result = resolveLineLoad(load, geometry, view.card(), view.elementId(load.elementRow));
        if (const auto* problem = std::get_if<base::Diagnostic>(&result))
            throw std::invalid_argument(
                base::concatenate({base::formatDiagnostic(*problem),
                                   " (the loads facet was called on a line load checkLineLoads() refuses)"}));
        resolved.push_back(std::get<BeamLineLoad>(result));
    }
    return resolved;
}

} // namespace oos::elements
