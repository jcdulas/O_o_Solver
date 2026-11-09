#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/solids/SolidResponse.hpp"

#include <span>

namespace oos::elements
{

// The eight-node SolidHexa of the default integration (PSOLID IN and ISOP blank): the trilinear hexahedron enriched by
// incompatible modes, the element of Wilson, Taylor, Doherty & Ghaboussi (Incompatible displacement models, in
// Numerical and Computer Methods in Structural Mechanics, Academic Press, 1973) with the correction of Taylor,
// Beresford & Wilson (A non-conforming element for stress analysis, IJNME 10, 1976), as the open-source solver
// CalculiX gives it (its element C3D8I); Bathe, Finite Element Procedures, §4.4.1 (incompatible displacement models).
// Positions, the elasticity D and the expansion a are in the material system; the response (SolidResponse) is over the
// 24 translations of G1-G8 in that system.
//
// Formulation:
// - the displacement is the trilinear field plus three quadratic modes per component, u_inc = Σ_k (1 − ξ_k²)·α_k
//   (ξ_k = ξ, η, ζ; α_k a vector: 9 internal parameters), which the nodes do not see (incompatible between elements);
// - the strain of the modes, G·α, takes the parent derivative −2ξ_k of each mode through the inverse Jacobian at the
//   centre J0⁻¹ and the factor det J0 / det J (Taylor's correction):
//   ∇u_inc = Σ_k α_k ⊗ (det J0 / det J)·J0⁻¹·(−2ξ_k e_k). Then ∫ G dV = det J0·J0⁻¹·∫ ∂(1 − ξ_k²)/∂ξ_k dξ = 0 on
//   any shape, so a constant stress does no work on the modes: the element passes the patch test on distorted meshes
//   (the uncorrected modes pass it on parallelepipeds only);
// - K_uu, K_uα, K_αα integrated by 2 × 2 × 2 Gauss points, the modes condensed at element level:
//   K = K_uu − K_uα·K_αα⁻¹·K_αu, α = −K_αα⁻¹·K_αu·u (+ the thermal part below).
// On a parallelepiped the modes give the exact linear bending strain of each direction: no shear locking, and the
// Poisson coupling of bending is free (a rectangular element under pure bending is exact).
//
// Recovery: the total strain ε = B·u + G·α. At the centre G vanishes (ξ_k = 0) and ε = B(0)·u; at the corners the
// strains of the eight Gauss points are extrapolated by their trilinear functions (Hinton & Campbell, IJNME 8, 1974;
// CalculiX also extrapolates the Gauss point values of its elements to the nodes), which avoids evaluating
// det J0 / det J at a corner of a distorted element. σ = D·ε.
//
// Thermal strain ε_T = a·ΔT, ΔT interpolated from the nodes: f_α = ∫ Gᵀ·D·a·ΔT dV, condensed into the nodal loads
// (SolidResponse::thermalLoad, −K_uα·K_αα⁻¹·f_α) and into the strains (thermalStrain, G·K_αα⁻¹·f_α), so that a free
// linear temperature gradient bends a rectangular element without stress.
void buildSolidHexaIncompatibleModesResponse(std::span<const base::Vec3> positions, const Mat6& elasticity,
                                            const Voigt6& expansion, SolidResponseParts parts, SolidResponse& response);

} // namespace oos::elements
