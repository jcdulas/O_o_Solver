#pragma once

#include "oos/base/math/MatN.hpp"

#include <array>

namespace oos::elements
{

// The rigid offsets that join the corner grid points of a warped quadrilateral to their projections on its mean plane,
// for a plate formed flat in that plane (ShellQuad4DrillingPlate: the ShellQuad4Drilling). The mean plane is the one of
// ShellQuad4DrillingGeometry: through the mean of the four grid points, normal to the cross product of the diagonals (so also
// through the four mid-side points); grid point i lies at X_i = P_i + h_i n, h_i = z_i^d, P_i its projection (the plane
// coordinates x_i, y_i). A rigid link from X_i to P_i, the vector −h_i n, moves P_i by
//
//   u_P = u + θ × (−h_i n),   θ_P = θ,
//
// that is, in the element frame (n the z axis), on the 6 DOFs of corner i:
//
//   u_P = u − h_i θy,   v_P = v + h_i θx,   w_P = w,   θx_P = θx,   θy_P = θy,   θn_P = θn.
//
// The drilling rotation θn is a rotation like the others: it passes the link unchanged and no translation along n
// couples to it. With q_P = T q on the 24 local DOFs, the element's matrices on the grid points are Tᵀ K T and its
// loads Tᵀ f, the plate (K, f) being formed on the flat projected element. A rigid motion of the grid points is a rigid
// motion of the projections (u_P = u_0 + θ × P_i when u_i = u_0 + θ × X_i), so a warped element keeps the rigid modes
// of the flat one, free of strain.
//
// Sources: the rigid-link projection of R. L. Taylor, "Finite element analysis of linear shell problems", in J. R.
// Whiteman (ed.), The Mathematics of Finite Elements and Applications VI (MAFELAP 1987), Academic Press, 1988; written
// out (the matrix W_i = Tᵀ_i, K = W K_flat Wᵀ, the mean plane through the mid-side points) in H. Nguyen-Van,
// N. Mai-Duy, T. Tran-Cong, "An improved quadrilateral flat element with drilling degrees of freedom for shell
// structural analysis", CMES 49 (2009) 81-110, eqs. (25)-(26); R. D. Cook, "Four-node 'flat' shell element: drilling
// degrees of freedom, membrane-bending coupling, warped geometry, and behavior", Computers & Structures 50 (1994)
// 549-555.
struct ShellQuad4DrillingWarpOffsets
{
    std::array<double, 4> height{}; // h_i = z_i^d: the grid point i lies h_i above its projection along n

    // q_P = T q: the displacements of the projections from those of the grid points (local DOFs).
    [[nodiscard]] std::array<double, 24> projected(const std::array<double, 24>& q) const;
    // Tᵀ f: the loads on the grid points of the loads f on the projections.
    [[nodiscard]] std::array<double, 24> loadsOnGridPoints(const std::array<double, 24>& f) const;
    // Tᵀ K T: a matrix on the grid points of a matrix K on the projections.
    [[nodiscard]] base::MatN<24> matrixOnGridPoints(const base::MatN<24>& k) const;
};

} // namespace oos::elements
