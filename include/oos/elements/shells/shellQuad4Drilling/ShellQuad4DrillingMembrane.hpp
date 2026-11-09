#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"

#include <array>

namespace oos::elements
{

// The membrane of a ShellQuad4Drilling (a quadrilateral membrane with a normal rotational (drilling) degree of freedom):
// the quadrilateral of Ibrahimbegović, Taylor and Wilson (A robust quadrilateral membrane finite element with drilling
// degrees of freedom, IJNME 30, 1990), on (u, v, θn) of the four corners, flat, in the element frame:
//
//   u = Σ N_i u_i + Σ_k M_k (y_j − y_i)/8 (θn_j − θn_i),   v = Σ N_i v_i + Σ_k M_k (x_i − x_j)/8 (θn_j − θn_i),
//
// N_i the bilinear functions, M_k the edge functions of the eight-node serendipity element on edge k = (i, j)
// (½(1 − ξ²)(1 ∓ η), ½(1 ± ξ)(1 − η²)), the Allman-type field of the edge rotations; and the rotation penalty
// γ ∫ (θn − ω)² t dA, θn = Σ N_i θn_i, ω = ½ (∂v/∂x − ∂u/∂y), γ = G (DRILLING_MEMBRANE_PENALTY_RATIO). Both
// integrated by the 2 × 2 Gauss rule, t(ξ, η) = Σ N_i t_i:
//
//   K = Σ_g w det J t (Bᵀ C B + γ hᵀ h),  h·q = θn − ω.
//
// Status: the published formulation.
//
// Thread-safety: immutable once built. A plain value.
class ShellQuad4DrillingMembrane
{
public:
    ShellQuad4DrillingMembrane(const ShellQuad4DrillingGeometry& geometry, const base::Mat3& planeStress,
                             const std::array<double, 4>& thickness, double shearModulus);

    // K (12 × 12) on (u, v, θn) of the four corners, node by node.
    [[nodiscard]] const base::MatN<12>& stiffness() const noexcept { return m_stiffness; }
    // B (3 × 12) of the strains (εx, εy, γxy) at (ξ, η).
    [[nodiscard]] base::MatN<3, 12> strainMatrix(double xi, double eta) const;

private:
    ShellQuad4DrillingGeometry m_geometry;
    base::MatN<12> m_stiffness;
};

} // namespace oos::elements
