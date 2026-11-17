#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The linear shell element of the mixed hybrid formulation (document 34 of the reference, sections 2 to 4; spec
// section 3.1): the TDNNS shell of M. Neunteufel and J. Schöberl (The Hellan-Herrmann-Johnson and TDNNS methods for
// linear and nonlinear shells, Computers & Structures 305 (2024) 107543) with membrane strains interpolated in the
// Regge space (M. Neunteufel, J. Schöberl, Avoiding membrane locking with Regge interpolation, Computer Methods in
// Applied Mechanics and Engineering 373 (2021) 113524).
//
// Unknowns of an element, in this order (the local ordering every caller follows):
// - the three Cartesian components of the displacement for each displacement function i: index 3 i + c;
// - the shear coefficients of the shear functions i: index 3 h1Count + i;
// - the rotations about the edges, edge k and Legendre degree l: index 3 h1Count + hcurlCount + k (p + 1) + l.
// (shellUnknownCount.) Edge rotation: the rotation about edge k is a_T = s_k â(τ_e) / (|X_τ| W²), â = Σ_l a_kl
// P_l(2 τ_e − 1), τ_e the canonical parameter of the edge (τ where s_k = +1, 1 − τ otherwise); so the value of â is the
// rotation about the edge in the canonical direction times |X_τ| W², and a rigid motion is exact (document 34 of the
// reference, section 4.2). The shear: β = γ − ∇uᵀ ν is the turn of the director, θ the rotation with β = θ × ν.
//
// The moments m (broken Hellan-Herrmann-Johnson space, the double contravariant Piola map) are eliminated in the
// element:
//   b_T(m; u, γ, a) = ∫_T m : κ(u, γ) dA − ∫_∂T m_μμ (β · μ − a_T) ds,
//   κ_αβ = −ν · (∂_αβ u − Γ^λ_αβ ∂_λ u) + ½ (γ_α|β + γ_β|α),
// μ = t × ν the outward conormal of the edge (t the unit tangent in the element's direction). With
// M_σ = ∫ S : D⁻¹ : S, G = b_T(S; ·) and the membrane-bending coupling C = ∫ S : D⁻¹ Bᵀ e(·) (partially mixed: the
// membrane strain stays primal, document 34 of the reference, section 3.2),
//   K = K_membrane + K_shear + (G + C)ᵀ M_σ⁻¹ (G + C) = K_membrane + K_shear + Yᵀ Y,   Y = L⁻¹ S (G + C),
// S L Lᵀ S the Cholesky factorisation of M_σ after its scaling to a unit diagonal (EquilibratedCholesky: the Piola map
// spreads the diagonal of M_σ over many decades on elongated elements). K_membrane = Iᵀ (∫ Rᵀ Ā R) I with Ā = A − B
// D⁻¹ Bᵀ and I the Regge interpolation of the membrane strain of the displacement; K_shear = ∫ γᵀ H γ.
//
// A membrane-only section on the surface of the grid points (sectionCarriesMoments false; spec section 4.1, not in the
// reference) has no moments and no transverse shear: K = K_membrane with Ā = A, the pairing and M_σ⁻¹ are zero (the
// moments of the recovery are zero), and its shear and edge rotation unknowns carry no stiffness (the model holds
// those no other element stiffens). A bending-only section has A = B = 0: K_membrane and C vanish.
//
// The three reference constants of the membrane and moment spaces are replaced by the constant states of the element's
// frame (PointGeometry::phi), and the three constant interior functionals of the Regge interpolation by the work of the
// three constant membrane forces, unless `mapReferenceConstants` (document 34 of the reference, section 4.3). On an
// affine element this changes nothing; on the others it restores the constant states (the patch tests).
//
// Reference: tdnns/element.rs (ElemMat, nx, kinematics, regge_local, hhj_local, regge_functionals, element).
// The loads are in ShellElementLoads.hpp, the stresses in ShellElementStresses.hpp.
//
// Thread-safety: pure functions; the matrices are plain values.

// The element: stiffness, mass, and what the recovery of the stresses needs.
// Reference: tdnns/element.rs::ElemMat (nx, k, mass, gc, msig_inv, iop, rg_dinv)
struct ShellElementMatrices
{
    std::size_t unknownCount = 0;
    Matrix stiffness;
    std::optional<Matrix> mass;
    // G + C (momentCount × unknownCount).
    Matrix pairing;
    // M_σ⁻¹ (momentCount × momentCount), symmetric.
    Matrix complianceInverse;
    // The Regge interpolation of the membrane strain (reggeCount × unknownCount).
    Matrix reggeInterpolation;
    // The inverse of the membrane functionals on the element's membrane basis (reggeCount × reggeCount).
    Matrix reggeFunctionalInverse;
};

// The number of local unknowns: 3 h1Count + hcurlCount + cornerCount (p + 1).
// Reference: tdnns/element.rs::nx
[[nodiscard]] std::size_t shellUnknownCount(const ShellReferenceElement& re);

// The displacement u (3 × unknownCount) and the director change β (3 × unknownCount) of every unknown at surface
// point q: u = Σ N_i u_i, β = −(∇u)ᵀ ν + γ, ∇N_i = a^α ∂_α N_i, γ = a^α g_α (covariant map of the shear functions).
// Reference: tdnns/element.rs::kinematics
[[nodiscard]] std::pair<Matrix, Matrix> pointKinematics(const ShellReferenceElement& re,
                                                        const ShellElementDefinition& def, std::size_t q,
                                                        const PointGeometry& g, std::size_t unknownCount);

// The values (11, 22, 12) of reference function m at the point.
using ReferenceTensorValues = std::function<std::array<double, 3>(std::size_t)>;

// The element's membrane basis at a point in local Voigt strains (3 × reggeCount): the reference functions mapped
// covariantly (covariantToVoigt), the last three the constant strains of the frame of the constant states,
// T(−φ) e_c (unless `mapReferenceConstants`).
// Reference: tdnns/element.rs::regge_local
[[nodiscard]] Matrix reggeLocal(const ShellReferenceElement& re, const ReferenceTensorValues& values,
                                const PointGeometry& g, bool mapReferenceConstants);

// The element's moment basis at a point in local moments (m11, m22, m12) (3 × momentCount): the reference functions
// mapped by the double contravariant Piola map, the last three the constant moments of the frame of the constant
// states, T(φ)ᵀ e_c (unless `mapReferenceConstants`).
// Reference: tdnns/element.rs::hhj_local
[[nodiscard]] Matrix momentsLocal(const ShellReferenceElement& re, const ReferenceTensorValues& values,
                                  const PointGeometry& g, bool mapReferenceConstants);

// The covariant strain (11, 22, 12) of a field at surface point q.
using CovariantStrainAtPoint = std::function<std::array<double, 3>(std::size_t)>;
// t̂ᵀ ε t̂ of the field at point g of edge k (t̂ the reference edge vector).
using TangentialStrainAtEdge = std::function<double(std::size_t, std::size_t)>;

// The membrane functionals applied to a strain field: the tangential moments of each edge (k, l) against
// P_l(2τ − 1), the reference interior moments of reggeInteriorMoments against their test polynomials of zero mean,
// then the work of the three constant membrane forces of the frame of the constant states (or the plain moments of
// the three covariant components with `mapReferenceConstants`).
// Reference: tdnns/element.rs::regge_functionals
[[nodiscard]] Vector reggeFunctionals(const ShellReferenceElement& re, std::span<const PointGeometry> geometries,
                                      const CovariantStrainAtPoint& volume, const TangentialStrainAtEdge& edge,
                                      bool mapReferenceConstants);

// A point of the map of a shell whose Jacobian is at most this fraction of the largest makes the map not invertible
// (buildShellElement refuses the element with SHELL_MAP_NOT_INVERTIBLE).
inline constexpr double SHELL_JACOBIAN_RATIO = 1e-10;
inline constexpr const char* SHELL_MAP_NOT_INVERTIBLE =
    "the map of the element is not invertible (its Jacobian vanishes inside)";

// Builds the element: stiffness (symmetric), the mass when asked
//   M = ∫ (I0 uᵀu + I1 (uᵀβ + βᵀu) + I2 βᵀβ) dA,
// and the operators of the stress recovery. The refusal (a message) when the map is not invertible (its Jacobian
// below 1e-10 of its largest at a point), the membrane functionals are not unisolvent on the element, or M_σ is not
// positive definite (a degenerate element).
// Reference: tdnns/element.rs::element
[[nodiscard]] std::variant<ShellElementMatrices, std::string> buildShellElement(const ShellReferenceElement& re,
                                                                                const ShellElementDefinition& def,
                                                                                bool withMass);

} // namespace oos::elements::mixedHybrid
