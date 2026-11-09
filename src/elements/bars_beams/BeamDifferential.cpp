#include "oos/elements/bars_beams/BeamDifferential.hpp"

#include "oos/elements/bars_beams/BeamFlexibility.hpp"
#include "oos/elements/bars_beams/BeamThermal.hpp"
#include "oos/elements/bars_beams/PinRelease.hpp"

#include <optional>
#include <stdexcept>

namespace oos::elements
{
namespace
{

// The end DOFs of the two bending planes: (deflection, rotation) at A then at B; the rotation of plane 2 is −w'.
constexpr std::array<std::size_t, 4> PLANE1_DOFS{1, 5, 7, 11}; // v_A, θz_A, v_B, θz_B
constexpr std::array<std::size_t, 4> PLANE2_DOFS{2, 4, 8, 10}; // w_A, θy_A, w_B, θy_B
constexpr std::array<double, 4> PLANE2_SIGNS{1.0, -1.0, 1.0, -1.0};

} // namespace

base::MatN<12> beamEndDifferentialStiffness(double axialForce, double length)
{
    const double l = length;
    // N/(30 L)·[36 3L −36 3L; 3L 4L² −3L −L²; −36 −3L 36 −3L; 3L −L² −3L 4L²] (Przemieniecki §15.3).
    const std::array<std::array<double, 4>, 4> shape{{{36.0, 3.0 * l, -36.0, 3.0 * l},
                                                      {3.0 * l, 4.0 * l * l, -3.0 * l, -l * l},
                                                      {-36.0, -3.0 * l, 36.0, -3.0 * l},
                                                      {3.0 * l, -l * l, -3.0 * l, 4.0 * l * l}}};
    const double scale = axialForce / (30.0 * l);
    base::MatN<12> result;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
        {
            result(PLANE1_DOFS[i], PLANE1_DOFS[j]) = scale * shape[i][j];
            result(PLANE2_DOFS[i], PLANE2_DOFS[j]) = scale * PLANE2_SIGNS[i] * PLANE2_SIGNS[j] * shape[i][j];
        }
    return result;
}

base::MatN<12> beamDifferentialStiffness(const BeamData& beam, double axialForce)
{
    base::MatN<12> endDifferential = beamEndDifferentialStiffness(axialForce, beam.geometry.length);
    if (beam.geometry.hasPinFlags())
    {
        const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
        const std::optional<PinRelease> release = PinRelease::make(flexibility.stiffness(), beam.geometry.isReleased);
        if (!release)
            throw std::invalid_argument(
                "bar or beam: pin flags on DOFs without stiffness (the caller skipped check(): a bug of O_o_Solver, "
                "report it)");
        // P, column by column: a kept DOF at 1 and the released ones as the condensation makes them.
        base::MatN<12> map;
        const std::array<double, 12> noLoads{};
        for (std::size_t j = 0; j < 12; ++j)
        {
            if (beam.geometry.isReleased[j])
                continue;
            std::array<double, 12> kept{};
            kept[j] = 1.0;
            const std::array<double, 12> completed = release->completeDisplacements(kept, noLoads);
            for (std::size_t i = 0; i < 12; ++i)
                map(i, j) = completed[i];
        }
        endDifferential = base::congruence(endDifferential, map);
    }
    return base::congruence(endDifferential, beam.geometry.endTransform());
}

double beamPreloadAxialForce(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                             const NodalDisplacements& preload)
{
    // The end force at B along the element x axis: the force the grid point exerts on the element, positive in
    // tension.
    constexpr std::size_t AXIAL_AT_B = 6;
    return BeamResponse(beam, preload, {}, beamThermalStrain(view, row, beam, preload)).endForces()[AXIAL_AT_B];
}

} // namespace oos::elements
