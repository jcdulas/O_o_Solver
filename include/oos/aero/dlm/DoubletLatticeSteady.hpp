#pragma once

#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseLu.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <cstdint>
#include <span>

namespace oos::aero
{

// The doublet-lattice method at zero reduced frequency (k = 0), the steady subsonic aerodynamics of SOL 144 (Albano &
// Rodden, AIAA J. 7 (1969) 279). The doublet-lattice kernel splits into its steady part and an oscillatory increment
// that vanishes at k = 0; the steady part is the horseshoe vortex of the vortex-lattice method (Hedman, FFA Report 105,
// 1966): on each box a bound vortex on the quarter-chord line and two trailing vortices from its ends to downstream
// infinity, parallel to the flow.
//
// The normalwash factors D relate the pressure coefficients of the boxes to the normalwash at the control points,
//     w_i / V = Σ_j D_ij Δcp_j,    D_ij = (c_j / 2) n_i · v(P_i; horseshoe j),
// with v the velocity per unit circulation (Biot-Savart: the 1/4π included), c_j the chord of box j and n_i the normal
// of box i: the circulation of a box carrying Δcp_j over its chord is Γ_j = Δcp_j V c_j / 2 (Kutta-Joukowski, a lift
// Δcp q c per unit span). Compressibility by the Prandtl-Glauert transformation: in the subsonic steady flow the
// perturbation potential satisfies β² φ_xx + φ_yy + φ_zz = 0, β = √(1 − M²), which x' = x / β turns into Laplace's
// equation; the steady kernel of the lifting-surface integral then has the same form in x' with the area element
// dx = β dx', so that D_ij keeps its incompressible form with the geometry stretched in x and the box chords as they
// are (the normals are perpendicular to the flow, so the normalwash takes only the v and w components, which the
// stretch leaves unchanged). A 2-D check: one box of chord c gives Δcp = 2π α / β, thin-airfoil theory with the
// Prandtl-Glauert factor.
//
// Symmetry (AEROS SYMXZ, SYMXY): the boxes model half of the configuration; each horseshoe has an image in the plane,
// mirrored, whose pressure is s·Δcp on the mirrored normal (s = +1 for a symmetric motion, −1 for an antisymmetric
// one): it adds −s·v(mirrored start → mirrored end) (the mirror reverses the sense n × x of the quarter-chord line).
// With both keys the image in both planes adds sXZ·sXY·v. The forces are those of the modelled boxes.
//
// A point closer to a vortex line than VORTEX_CORE_RATIO of the box span takes no velocity from it: on the line the
// Biot-Savart velocity is singular, and the principal value the lifting-surface theory takes is zero for a straight
// line (the control points of a lattice are never on a line of their own surface; a surface in the wake plane of
// another may put one there).
//
// Thread-safety: stateless.
class DoubletLatticeSteady
{
public:
    // The ratio to the box span below which a point is taken on a vortex line (see the class comment).
    static constexpr double VORTEX_CORE_RATIO = 1.0e-9;

    // D (n × n) for the boxes at Mach number `mach` (0 ≤ M < 1) with the symmetry keys (+1, −1 or 0). Throws
    // std::invalid_argument for a Mach number outside [0, 1) or a key outside {−1, 0, 1}.
    [[nodiscard]] static linalg::DenseMatrix normalwashFactors(const AeroBoxes& boxes, double mach,
                                                               std::int32_t symmetryXz, std::int32_t symmetryXy,
                                                               base::MemoryBudget& budget);

    // The velocity at `point` induced by a horseshoe vortex of unit circulation: trailing vortex from +∞ (along x) to
    // `start`, bound vortex from `start` to `end`, trailing vortex from `end` to +∞; `coreLength` the length below
    // which a distance to a line counts as zero.
    [[nodiscard]] static base::Vec3 horseshoeVelocity(base::Vec3 point, base::Vec3 start, base::Vec3 end,
                                                      double coreLength);
};

// The pressure coefficients of the boxes for normalwashes: D Δcp = w, solved column by column on the LU factor of D
// (`normalwash`: n × m, overwritten by Δcp). Throws SolverError when D is singular (two boxes at the same place).
void solvePressures(const linalg::DenseLu& factor, linalg::DenseMatrixView normalwash);

} // namespace oos::aero
