#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <array>
#include <span>

namespace oos::aero
{

// The infinite plate spline (IPS) of AeroSurfaceSpline1 and AeroSurfaceSpline4 (Harder & Desmarais, "Interpolation
// using surface splines", J. Aircraft 9 (1972) 189; the surface spline of Rodden, Harder & Bellinger, NASA CR-3094,
// 1979): the deflection normal to the plane of a panel is that of an infinite plate loaded by point forces P_i at the
// structural points,
//     w(x, y) = a0 + a1 x + a2 y + Σ_i P_i K(r_i),    K(r) = r² ln r² / (16π D),    r_i the distance to point i,
// with the forces in equilibrium (Σ P_i = Σ x_i P_i = Σ y_i P_i = 0) and the plate attached to the structure by springs
// of flexibility DZ (SPLINE1 DZ, 0: the plate passes through every point): a0 + a1 x_j + a2 y_j + Σ_i K_ji P_i +
// DZ P_j = u_j. D = 1, as in NASA CR-3094: the interpolation does not depend on it when DZ = 0, and DZ is a
// flexibility relative to it. The polynomial part holds the rigid motions (a plunge and the rotations about x and y),
// which the spline reproduces exactly, whatever DZ. TPS (the thin plate spline of METH) has the same kernel up to a
// factor, so the same interpolation.
//
// Thread-safety: stateless.
class SurfaceSpline
{
public:
    // The deflection and its slope ∂w/∂x at each target point (x, y) for a unit displacement of each structural point:
    // plunge (targets × points) and slope (targets × points), w_target = Σ_i plunge(t, i) u_i. Throws SolverError when
    // the points are fewer than three or all on a line (the plane of the spline is then undetermined).
    struct Interpolation
    {
        linalg::DenseMatrix plunge;
        linalg::DenseMatrix slope;
    };
    [[nodiscard]] static Interpolation interpolate(std::span<const std::array<double, 2>> points, double dz,
                                                   std::span<const std::array<double, 2>> targets,
                                                   base::MemoryBudget& budget);

    // K(r) for r² (0 at r = 0, its limit) and ∂K/∂x for the offset (dx, dy).
    [[nodiscard]] static double kernel(double rSquared);
    [[nodiscard]] static double kernelSlope(double dx, double dy);
};

} // namespace oos::aero
