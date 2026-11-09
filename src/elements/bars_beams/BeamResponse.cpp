#include "oos/elements/bars_beams/BeamResponse.hpp"

#include "oos/elements/bars_beams/BeamFlexibility.hpp"
#include "oos/elements/bars_beams/BeamThermal.hpp"
#include "oos/elements/bars_beams/PinRelease.hpp"
#include "oos/elements/core/MarginOfSafety.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace oos::elements
{

BeamEndForces::BeamEndForces(const BeamData& beam, std::span<const BeamLineLoad> loads, double thermalStrain)
    : m_beam(&beam)
{
    const BeamFlexibility flexibility(beam.section, beam.material, beam.geometry.length);
    m_stiffness = flexibility.stiffness();
    m_equivalent = flexibility.equivalentLoads(loads);
    if (thermalStrain != 0.0)
    {
        const std::array<double, 12> thermal = beamThermalEndLoads(beam, thermalStrain);
        for (std::size_t i = 0; i < 12; ++i)
            m_equivalent[i] += thermal[i];
    }
    if (beam.geometry.hasPinFlags())
    {
        m_release = PinRelease::make(m_stiffness, beam.geometry.isReleased);
        if (!m_release)
            throw std::invalid_argument(
                "bar or beam: pin flags on DOFs without stiffness (the caller skipped check(): a bug of O_o_Solver, "
                "report it)");
    }
}

std::array<double, 12> BeamEndForces::of(const std::array<double, 12>& grid) const
{
    std::array<double, 12> ends = m_beam->geometry.endTransform() * grid;
    if (m_release)
        ends = m_release->completeDisplacements(ends, m_equivalent);
    const std::array<double, 12> elastic = m_stiffness * ends;
    std::array<double, 12> forces{};
    for (std::size_t i = 0; i < 12; ++i)
        forces[i] = elastic[i] - m_equivalent[i];
    return forces;
}

BeamResponse::BeamResponse(const BeamData& beam, const NodalDisplacements& displacements,
                           std::span<const BeamLineLoad> loads, double thermalStrain)
    : m_beam(&beam), m_loads(loads)
{
    const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
    std::array<double, 12> grid{};
    for (std::size_t i = 0; i < 12; ++i)
        grid[i] = displacements.valueAt(dofs[i]);
    m_endForces = BeamEndForces(beam, loads, thermalStrain).of(grid);
}

BeamResponse::BeamResponse(const BeamData& beam, const std::array<double, 12>& endForces,
                           std::span<const BeamLineLoad> loads)
    : m_beam(&beam), m_loads(loads), m_endForces(endForces)
{
}

BeamForces BeamResponse::forcesAt(double x) const
{
    const double length = m_beam->geometry.length;
    const base::Vec3 forceA{m_endForces[0], m_endForces[1], m_endForces[2]};
    const base::Vec3 momentA{m_endForces[3], m_endForces[4], m_endForces[5]};
    const LoadResultant left = loadResultant(m_loads, 0.0, x, x, x >= length);
    const base::Vec3 force = -forceA - left.force;
    // x·e_x × F_A = x·(0, −F_z, F_y).
    const base::Vec3 moment = -momentA + base::Vec3{0.0, -x * forceA[2], x * forceA[1]} - left.moment;
    BeamForces forces;
    forces.axial = force[0];
    forces.shear1 = force[1];
    forces.shear2 = force[2];
    forces.torque = moment[0];
    forces.moment1 = moment[2];
    forces.moment2 = -moment[1];
    // The bending moments about the neutral axis (M1, M2 at the neutral axis, V1, V2 and the
    // torque at the shear centre, the element axis): Mz_n = Mz + y_n·N, My_n = My − z_n·N (BeamFlexibility).
    const BeamSection& section = m_beam->section;
    if (section.hasNeutralAxisOffset())
    {
        const BeamStation station = section.at(x / length);
        forces.moment1 += station.neutralAxis[0] * forces.axial;
        forces.moment2 += station.neutralAxis[1] * forces.axial;
    }
    return forces;
}

std::array<double, 4> bendingStresses(const BeamStation& station, const BeamForces& forces)
{
    const double momentY = -forces.moment2;
    const double momentZ = forces.moment1;
    std::array<double, 4> stresses{};
    for (std::size_t point = 0; point < 4; ++point)
    {
        const double y = station.recoveryPoints[point][0];
        const double z = station.recoveryPoints[point][1];
        double stress = 0.0;
        if (station.productOfInertia != 0.0)
        {
            const double determinant =
                station.inertia1 * station.inertia2 - station.productOfInertia * station.productOfInertia;
            stress = (z * (station.inertia1 * momentY + station.productOfInertia * momentZ) -
                      y * (station.productOfInertia * momentY + station.inertia2 * momentZ)) /
                     determinant;
        }
        else
        {
            if (station.inertia2 > 0.0)
                stress += z * momentY / station.inertia2;
            if (station.inertia1 > 0.0)
                stress -= y * momentZ / station.inertia1;
        }
        stresses[point] = stress;
    }
    return stresses;
}

double axialStress(const BeamStation& station, const BeamForces& forces)
{
    return station.area > 0.0 ? forces.axial / station.area : 0.0;
}

StationStresses stationStresses(const BeamStation& station, const BeamForces& forces)
{
    StationStresses result;
    result.bending = bendingStresses(station, forces);
    result.axial = axialStress(station, forces);
    result.maximum = result.axial + *std::max_element(result.bending.begin(), result.bending.end());
    result.minimum = result.axial + *std::min_element(result.bending.begin(), result.bending.end());
    return result;
}

double tensionMargin(double largestStress, const IsotropicMaterial& material)
{
    return largestStress > 0.0 ? normalStressMargin(largestStress, material.tensionLimit, material.compressionLimit)
                               : UNDEFINED_MARGIN;
}

double compressionMargin(double smallestStress, const IsotropicMaterial& material)
{
    return smallestStress < 0.0 ? normalStressMargin(smallestStress, material.tensionLimit, material.compressionLimit)
                                : UNDEFINED_MARGIN;
}

} // namespace oos::elements
