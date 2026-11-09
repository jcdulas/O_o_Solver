#pragma once

#include "oos/base/math/MatN.hpp"

#include <array>
#include <cstddef>

namespace oos::elements
{

// The differential stiffness of the shells (SOL 105, linear buckling): the initial-stress stiffness of the
// membrane stress resultants N = (Nx, Ny, Nxy) of the preload (forces per unit length in the element frame, the
// thermal resultant subtracted) acting on the in-plane gradients of the three translations,
//
//   K_d(a i, b i) = ∫ [∂N_a/∂x ∂N_a/∂y]·[Nx Nxy; Nxy Ny]·[∂N_b/∂x; ∂N_b/∂y] dA,   i = u, v, w (element frame),
//
// the second variation of ∫ N_αβ·½ (∂u_k/∂x_α)(∂u_k/∂x_β) dA, the membrane part of the Green-Lagrange strain under a
// fixed prestress (Zienkiewicz & Taylor, The Finite Element Method, vol. 2, §"initial stress" of plates and shells;
// Cook, Malkus, Plesha & Witt, Concepts and Applications of FEA, 4th ed., §18.4). The w terms are the plate buckling
// stiffness; the u, v terms make K_d turn the prestress correctly under a rigid rotation about the normal
// (K_d·u_rot = ω × f, the element's internal forces f rotated). Simplifications of this project: the resultants are
// those of the element centre (constant over the element), the bending moments and the transverse shear forces of the
// preload give no differential stiffness, and the rotations take none (no t³/12 term): a differential stiffness formed
// from the membrane forces only, the usual one of thin plates.
//
// Adds the contribution of one integration point to `k`, the matrix over `STRIDE` DOFs per node whose first three are
// the translations (u, v, w) of the element frame: `dX`, `dY` the physical derivatives of the shape functions there,
// `weight` the quadrature weight times det J.
template <std::size_t N, std::size_t STRIDE>
void addMembraneDifferentialStiffness(base::MatN<STRIDE * N>& k, const std::array<double, N>& dX,
                                      const std::array<double, N>& dY, double weight,
                                      const std::array<double, 3>& membraneForce)
{
    const double nx = membraneForce[0];
    const double ny = membraneForce[1];
    const double nxy = membraneForce[2];
    for (std::size_t a = 0; a < N; ++a)
    {
        const double flowX = nx * dX[a] + nxy * dY[a];
        const double flowY = nxy * dX[a] + ny * dY[a];
        for (std::size_t b = 0; b < N; ++b)
        {
            const double term = weight * (flowX * dX[b] + flowY * dY[b]);
            for (std::size_t i = 0; i < 3; ++i)
                k(STRIDE * a + i, STRIDE * b + i) += term;
        }
    }
}

} // namespace oos::elements
