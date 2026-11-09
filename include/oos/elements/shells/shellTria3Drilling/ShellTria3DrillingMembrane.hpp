#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"

#include <array>

namespace oos::elements
{

// The membrane of a ShellTria3Drilling (a triangle with a normal rotational (drilling) degree of freedom): the triangle
// of Allman (A compatible triangular element including vertex rotations for plane elasticity analysis, Computers &
// Structures 19, 1984), on (u, v, θn) of the three corners, in the element frame:
//
//   u = Σ L_i u_i + Σ_e 4 L_i L_j (y_j − y_i)/8 (θn_j − θn_i),
//   v = Σ L_i v_i + Σ_e 4 L_i L_j (x_i − x_j)/8 (θn_j − θn_i),
//
// e = (i, j) the edges 1-2, 2-3, 3-1; with the rotation penalty γ ∫ (θn − ω)² t dA (θn linear, γ = G,
// DRILLING_MEMBRANE_PENALTY_RATIO) that removes Allman's mode of equal rotations (Hughes, Brezzi 1989). Three-point
// rule (exact for the quadratic field's strain energy), t̄ the centroid thickness:
//
//   K = Σ_g (A_e/3) t̄ (Bᵀ C B + γ hᵀ h),  h·q = θn − ω.
//
// Status: the published formulation.
//
// Thread-safety: immutable once built. A plain value.
class ShellTria3DrillingMembrane
{
public:
    ShellTria3DrillingMembrane(const ShellTria3DrillingGeometry& geometry, const base::Mat3& planeStress, double thickness,
                             double shearModulus);

    // K (9 × 9) on (u, v, θn) of the three corners, node by node.
    [[nodiscard]] const base::MatN<9>& stiffness() const noexcept { return m_stiffness; }
    // B (3 × 9) of the strains at the area coordinates (L1, L2, L3).
    [[nodiscard]] base::MatN<3, 9> strainMatrix(const std::array<double, 3>& l) const;

private:
    ShellTria3DrillingGeometry m_geometry;
    base::MatN<9> m_stiffness;
};

} // namespace oos::elements
