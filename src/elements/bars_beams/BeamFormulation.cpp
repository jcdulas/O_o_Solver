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

// The end DOFs of the points at (0, y, z) from the end points, in the element system, rigidly attached to the section
// (it does not deform in its plane): u_p = u + θ × r = u − [r]×·θ, θ_p = θ, with r_A at end A and r_B at end B.
base::MatN<12> sectionPointTransform(const std::array<double, 2>& atA, const std::array<double, 2>& atB)
{
    base::MatN<12> t = base::MatN<12>::identity();
    const std::array<std::array<double, 2>, 2> points{atA, atB};
    for (std::size_t end = 0; end < 2; ++end)
    {
        const double y = points[end][0];
        const double z = points[end][1];
        const std::size_t first = 6 * end;
        // −[r]× for r = (0, y, z): rows (0, z, −y), (−z, 0, 0), (y, 0, 0).
        t(first + 0, first + 4) = z;
        t(first + 0, first + 5) = -y;
        t(first + 1, first + 3) = -z;
        t(first + 2, first + 3) = y;
    }
    return t;
}

// The mass of a Beam2 whose neutral axis or non-structural mass is off its axis, over the end DOFs: the structural
// mass ρ·A(x) (and the torsional inertia of the section) on the neutral axis, the non-structural mass (the BeamProp NSM
// and the NSM cards' share) at its centre of gravity (M1, M2), each carried rigidly to the axis of the shear centres by
// the offsets of its two ends (a BeamProp may set the neutral axis, the shear centre and the centre of gravity of the
// non-structural mass apart). Modelling choices: the structural mass at the neutral axis, the offsets those of the
// end stations (a tapered offset between them is carried by the end offsets), the NSM of a BeamShapeProp at the shear
// centre (its derived BeamProp has no M1, M2).
base::MatN<12> offsetSectionMass(const BeamSection& section, double density, double addedMassPerLength, double length,
                                 MassFormulation formulation, TorsionalInertia torsionalInertia)
{
    BeamSection structural = section;
    for (std::size_t s = 0; s < structural.count; ++s)
        structural.stations[s].nonStructuralMass = 0.0;
    const BeamStation& endA = section.stations[0];
    const BeamStation& endB = section.stations[section.count - 1];
    base::MatN<12> mass = congruence(beamMass(structural, density, 0.0, length, formulation, torsionalInertia),
                                     sectionPointTransform(endA.neutralAxis, endB.neutralAxis));
    // The non-structural part: density 0 leaves NSM(x) + the NSM cards' share, and no torsional inertia.
    mass += congruence(beamMass(section, 0.0, addedMassPerLength, length, formulation, torsionalInertia),
                       sectionPointTransform(endA.massCentre, endB.massCentre));
    return mass;
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
    const BeamSection& section = beam.section;
    const base::MatN<12> endMass = section.hasNeutralAxisOffset() || section.hasMassCentreOffset()
                                       ? offsetSectionMass(section, beam.material.density, addedMassPerLength,
                                                           beam.geometry.length, formulation, torsionalInertia)
                                       : beamMass(section, beam.material.density, addedMassPerLength,
                                                  beam.geometry.length, formulation, torsionalInertia);
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
    // f_grid = Tᵀ·f_end (T without the offsets under ELMOFF).
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
