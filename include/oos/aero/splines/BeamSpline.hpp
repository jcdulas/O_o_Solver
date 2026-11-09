#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <array>
#include <span>

namespace oos::aero
{

// The beam spline of AeroBeamSpline2 and AeroBeamSpline5 (the beam spline of Rodden, Harder & Bellinger, NASA CR-3094,
// 1979): a free beam along the y axis of the spline, loaded at the structural stations y_i by forces F_i, by moments
// M_i about x when the slopes are attached (DTHX ≥ 0) and by torques T_i when the twists are (DTHY ≥ 0), with EI = 1
// and GJ = 1 / DTOR:
//     w(y) = a0 + a1 y + Σ F_i |y − y_i|³ / 12 − Σ M_i (y − y_i)|y − y_i| / 4,
//     θ(y) = b0 − DTOR Σ T_i |y − y_i| / 2,
// the Green functions of w'''' = F δ, of a couple (Maxwell's reciprocal: the slope at j of a force at i is the
// deflection at i of a couple at j) and of θ'' = −T δ / GJ, the attachment flexibilities added to their diagonal; the
// loads in equilibrium (Σ F = 0, Σ F y + Σ M = 0, Σ T = 0) and the beam attached by springs: w(y_j) + DZ F_j = u_j,
// w'(y_j) + DTHX M_j = θx_j, θ(y_j) + DTHY T_j = θy_j. A box at (x, y) moves by h = w(y) − x θ(y) (a twist θ about y
// turns z towards x), with the slope ∂h/∂x = −θ(y). The rigid motions (a plunge, a rotation about x, and about y when
// the twists are attached) are reproduced exactly; without attached twists the beam does not twist (θ = 0), without
// attached slopes the slopes come from the deflections.
//
// Thread-safety: stateless.
class BeamSpline
{
public:
    // The flexibilities of the attachments: DZ, DTHX and DTHY (negative: not attached), and DTOR = EI / GJ.
    struct Flexibility
    {
        double dz = 0.0;
        double dthx = -1.0;
        double dthy = -1.0;
        double dtor = 1.0;
    };

    // The deflection and its slopes ∂h/∂x and ∂h/∂y at each target (x, y) for a unit value of each structural unknown,
    // the three of each station in this order: the deflection u_z, the rotation about x θx, the rotation about y θy (a
    // column is zero for an unknown that is not attached). Throws SolverError for fewer than two distinct stations (the
    // beam is undetermined).
    struct Interpolation
    {
        linalg::DenseMatrix plunge;    // targets × 3 stations
        linalg::DenseMatrix slope;     // ∂h/∂x = −θ(y), targets × 3 stations
        linalg::DenseMatrix spanSlope; // ∂h/∂y = w'(y) − x θ'(y), targets × 3 stations
    };
    [[nodiscard]] static Interpolation interpolate(std::span<const double> stations, const Flexibility& flexibility,
                                                   std::span<const std::array<double, 2>> targets,
                                                   base::MemoryBudget& budget);
};

} // namespace oos::aero
