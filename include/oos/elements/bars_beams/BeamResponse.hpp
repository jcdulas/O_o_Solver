#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/bars_beams/BeamLineLoad.hpp"
#include "oos/elements/bars_beams/PinRelease.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace oos::elements
{

// The element forces of a bar or beam at one section, in the sign convention of the Bar2 and Beam2 force output:
// moment1 and moment2 the bending moments in planes 1 (x-y) and 2 (x-z), positive with a positive curvature of the
// deflection along y, resp. z (moment1 = E·I1·v'' = Mz, moment2 = E·I2·w'' = −My for I12 = 0, My and Mz the moments
// acting on the face of normal +x); shear1 = Vy and shear2 = Vz, the forces on that face (so that shear = −dM/dx in
// each plane); axial positive in tension; torque the moment about +x on that face. The moments are about the neutral
// axis, the shears and the torque at the element axis (the shear centre of a Beam2 whose neutral axis is offset).
struct BeamForces
{
    double moment1 = 0.0;
    double moment2 = 0.0;
    double shear1 = 0.0;
    double shear2 = 0.0;
    double axial = 0.0;
    double torque = 0.0;
};

// The end forces of one bar or beam as a function of the displacements of its grid points (over
// beam.geometry.dofs(), in their order), its end transform, stiffness, equivalent loads and pin release made once:
// BeamResponse uses it for one set of displacements, the recovery slopes of a Bar2 for thirteen (the rows at rest
// and one per DOF). The steps and their order are BeamResponse's (class comment below), so the same bits.
//
// Thread-safety: immutable. Ownership: borrows the beam data and the loads, which must outlive it.
class BeamEndForces
{
public:
    // Throws std::invalid_argument for pin flags that check() refuses (a bug of the caller).
    BeamEndForces(const BeamData& beam, std::span<const BeamLineLoad> loads, double thermalStrain = 0.0);

    // f_e = K·d − f with d = T·g (the released DOFs completed), g the displacements of the grid points.
    [[nodiscard]] std::array<double, 12> of(const std::array<double, 12>& grid) const;

private:
    const BeamData* m_beam;
    base::MatN<12> m_stiffness;
    std::array<double, 12> m_equivalent{};
    std::optional<PinRelease> m_release;
};

// The response of one bar or beam to the displacements of its grid points (and the line loads on it, for the effect
// of loads between its ends):
//
// - end DOFs d = T·g (pin flags: the released DOFs from PinRelease::completeDisplacements);
// - end forces, the forces the grid points exert on the element, f_e = K·d − f (f the equivalent loads of
//   BeamFlexibility, so that a loaded element gives its exact end moments, plus K·d_T, the load of the
//   free thermal expansion of a subcase with a thermal load: BeamThermal);
// - stress resultants at a distance x from end A from the equilibrium of the part [0, x]: force −f_A − F[0,x),
//   moment −M_A + x·e_x × F_A − M[0,x) about x (the loads of [0, x] included at x = L, so that the value at L is f_B).
//
// Thread-safety: immutable. Precondition: the element passed check() and the loads checkLineLoads().
class BeamResponse
{
public:
    // `thermalStrain` the free axial strain α·ΔT of the element (beamThermalStrain), 0 without a thermal load.
    BeamResponse(const BeamData& beam, const NodalDisplacements& displacements, std::span<const BeamLineLoad> loads,
                 double thermalStrain = 0.0);
    // The response whose end forces `endForces` made (BeamEndForces::of), under the same loads.
    BeamResponse(const BeamData& beam, const std::array<double, 12>& endForces, std::span<const BeamLineLoad> loads);

    // The forces at a distance x from end A, in the output convention (BeamForces).
    [[nodiscard]] BeamForces forcesAt(double x) const;
    // The forces the grid points exert on the element, over the end DOFs (element system).
    [[nodiscard]] const std::array<double, 12>& endForces() const noexcept { return m_endForces; }

private:
    const BeamData* m_beam;
    std::span<const BeamLineLoad> m_loads;
    std::array<double, 12> m_endForces{};
};

// The bending stresses at the recovery points C, D, E, F of a station: σ = E·(−y·χz + z·χy) with the curvatures of
// the moments, i.e. σ = (z·(I1·My + I12·Mz) − y·(I12·My + I2·Mz)) / (I1·I2 − I12²) (σ = z·My/I2 − y·Mz/I1 for
// I12 = 0; a plane without bending stiffness contributes nothing).
[[nodiscard]] std::array<double, 4> bendingStresses(const BeamStation& station, const BeamForces& forces);

// The axial stress N/A (0 for A = 0: not defined).
[[nodiscard]] double axialStress(const BeamStation& station, const BeamForces& forces);

// The stresses of one station (the Bar2 stress output: SA1-SA4 the bending stresses at C, D, E, F, AXIAL, SA-MAX and
// SA-MIN the axial stress plus the largest and the smallest bending stress).
struct StationStresses
{
    std::array<double, 4> bending{};
    double axial = 0.0;
    double maximum = 0.0;
    double minimum = 0.0;
};
[[nodiscard]] StationStresses stationStresses(const BeamStation& station, const BeamForces& forces);

// The margin of safety in tension of the largest stress against ST, undefined (UNDEFINED_MARGIN) unless that stress
// is a tension and ST is given.
[[nodiscard]] double tensionMargin(double largestStress, const IsotropicMaterial& material);
// The margin of safety in compression of the smallest stress against SC, undefined unless that stress is a
// compression and SC is given.
[[nodiscard]] double compressionMargin(double smallestStress, const IsotropicMaterial& material);

} // namespace oos::elements
