#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/bars_beams/BeamResponse.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <array>
#include <cstdint>

namespace oos::elements
{

// The differential stiffness of a bar or beam (CBAR, Beam2; SOL 105) under the axial force N of
// the preload (positive in tension), over the end DOFs d = [u θ]_A [u θ]_B in the element system, from the second
// variation of ½ ∫ N (v'² + w'²) dx with the cubic (Hermite) deflections of the Euler-Bernoulli bar (Przemieniecki,
// Theory of Matrix Structural Analysis, §15.3; Cook, Malkus, Plesha & Witt, 4th ed., §18.2): in plane 1 on
// (v_A, θz_A, v_B, θz_B)
//
//   K_d1 = N/(30 L)·[36 3L −36 3L; 3L 4L² −3L −L²; −36 −3L 36 −3L; 3L −L² −3L 4L²],
//
// and the same in plane 2 on (w_A, θy_A, w_B, θy_B) with the signs of the rotation terms reversed (θy = −w'). The
// chord rotation is exact (K_d turns the axial force with a rigid rotation, K_d·d_rot = ω × f), and the Euler loads of
// a discretised column converge as h² from above for the clamped and pinned cases.
//
// Pin flags: the released end DOFs follow the kept ones as the elastic condensation makes them (PinRelease,
// d = P·d_kept with d_r = −K_rr⁻¹·K_ra·d_a), and K_d is condensed with the same map, Pᵀ·K_d·P: the deflected shape of
// the released element is that of its stiffness. Then the offsets and displacement systems: Tᵀ·K_d·T (BeamGeometry).
//
// Modelling choices of this formulation: the axial force is that
// of the element's ends without the line loads between them (constant along the element), the shear flexibility, the
// taper of a Beam2, the bending moments and the torque of the preload give no differential stiffness, and the axial
// force gives no torsional (Wagner) term.
[[nodiscard]] base::MatN<12> beamEndDifferentialStiffness(double axialForce, double length);
// The same over the grid DOFs [GA 1-6, GB 1-6] (pin flags, offsets, CD systems).
[[nodiscard]] base::MatN<12> beamDifferentialStiffness(const BeamData& beam, double axialForce);

// The axial force of the preload: the end force at B along x (BeamResponse, without line loads), the free axial strain
// of the preload's initial strains subtracted as the recovery subtracts it (beamThermalStrain: α·ΔT + D/L;
// without it the thermal strain would be counted as a mechanical one).
[[nodiscard]] double beamPreloadAxialForce(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                           const NodalDisplacements& preload);

// The differential stiffness facet of Bar2 and Beam2: `resolve(row)` returns the BeamData of a row.
template <typename Resolve>
void appendBeamDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                     const NodalDisplacements& preload, ElementMatrixBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const BeamData beam = resolve(row);
        const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
        const base::MatN<12> differential =
            beamDifferentialStiffness(beam, beamPreloadAxialForce(view, row, beam, preload));
        block.append(row, dofs, differential.values);
    }
}

} // namespace oos::elements
