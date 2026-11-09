#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/core/SolidIntegrationRules.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"
#include "oos/elements/solids/SolidResponse.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// How the stresses of an isoparametric solid are found at its recovery places (the centre and the corners):
// - Direct: the strains of the shape functions evaluated at the centre and at each corner (the fully integrated
//   elements);
// - FromHexaGauss: the centre evaluated there, the corners extrapolated from the values at the 2 × 2 × 2 Gauss points
//   of the parent hexahedron by the trilinear functions of these points (Hinton & Campbell, IJNME 8, 1974). The
//   pyramids, whose collapsed apex has no Jacobian: the apex takes the mean of the four collapsed corners;
// - AllFromHexaGauss: the corners as FromHexaGauss and the centre the mean of the eight Gauss points (the 20-node
//   hexahedron with the reduced 2 × 2 × 2 rule, whose strains are sampled at its Gauss points only: away from them the
//   zero-energy modes of the reduced rule would show).
enum class CornerRecovery : std::uint8_t
{
    Direct,
    FromHexaGauss,
    AllFromHexaGauss
};

// The response of a displacement-based isoparametric solid (Zienkiewicz & Taylor vol. 1 §6.2, §9.3):
// K = Σ_g Bᵀ·D·B·|det J|·w_g over the rule; σ = D·B(p)·u and ε = B(p)·u at the centre and the corners (CornerRecovery).
// Positions are in the material system. Precondition: the Jacobian is non-singular at the rule's points and, for
// Direct recovery, at the corners (checkSolidJacobian).
void buildIsoparametricResponse(SolidShape shape, std::span<const base::Vec3> positions, const Mat6& elasticity,
                                const SolidRule& rule, CornerRecovery corners, SolidResponseParts parts,
                                SolidResponse& response);

// The weight of the 2 × 2 × 2 Gauss point `point` in the value at `place` of the parent hexahedron: the trilinear
// function of the point, ⅛ Π_k (1 + s_k·√3·p_k), s the signs of the point's coordinates (the corners lie at √3 times
// the points in the parent coordinates of the cell the points span).
[[nodiscard]] double hexaGaussExtrapolationWeight(const ParentPoint& point, const ParentPoint& place);

// The initial-stress (differential) stiffness of a solid under a constant Cauchy stress σ (Voigt: σx, σy, σz, τxy,
// τyz, τzx), over the translations of its nodes (3·count × 3·count, the same block for each translation):
//
//   K_d(a i, b i) = ∫ ∇N_aᵀ·σ·∇N_b dV,   i = x, y, z,
//
// the second variation of ∫ σ_jk·½ (∂u_i/∂x_j)(∂u_i/∂x_k) dV, the non-linear part of the Green-Lagrange strain under a
// fixed prestress (Bathe, Finite Element Procedures, §6.3.2, the matrix K_NL; Zienkiewicz & Taylor vol. 2, initial
// stress), integrated by the rule on the isoparametric shape functions whatever the formulation of the linear
// stiffness (the incompatible modes of the default SolidHexa do not enter it). A design choice of this project: the
// stress is the element's stress at its centre.
void buildInitialStressStiffness(SolidShape shape, std::span<const base::Vec3> positions, const Voigt6& stress,
                                 const SolidRule& rule, SolidMatrix& differential);

// The scalar consistent mass M_ab = ∫ ρ N_a N_b dV of an element (count × count, the same for each translation),
// integrated by the rule.
void buildScalarMass(SolidShape shape, std::span<const base::Vec3> positions, double density, const SolidRule& rule,
                     SolidMatrix& mass);

// The volume of an element, ∫ |det J| dV by the rule.
[[nodiscard]] double solidVolume(SolidShape shape, std::span<const base::Vec3> positions, const SolidRule& rule);

// The sign of det J of an element at the points of a rule and, when `withCorners`, at its corners: +1 or −1 when it
// keeps one sign everywhere and its magnitude stays above `relativeTolerance` times the mean |det J|, 0 otherwise
// (a degenerate, inverted or folded element).
[[nodiscard]] int solidJacobianSign(SolidShape shape, std::span<const base::Vec3> positions, const SolidRule& rule,
                                    bool withCorners, double relativeTolerance);

} // namespace oos::elements
