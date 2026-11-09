#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingMembrane.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace oos::elements
{

// One triangular plate formulated in its element frame, the formulation of the ShellTria3Drilling family
// (ShellTria3DrillingElement, ShellTria3DrillingElement), on its 18 local DOFs (u, v, w, θx, θy, θn per corner, node by node,
// element frame, on the reference plane moved by ZOFFS):
//
// - membrane: ShellTria3DrillingMembrane on (u, v, θn) of the corners (the membrane with a drilling DOF);
// - bending: the DOFs (w, θx, θy) on linear rotations, constant curvatures with the twist row,
//   K_curvature = A_e B_bᵀ D B_b;
//   edge shear of the NASTRAN-95 TRIA3 (routines t3gemd, tria3d): the tangential shears at the three mid-edges
//   γ_e = (w_b − w_a)/L_e − ½ u_e·(β_a + β_b) give three centre parameters g = (γyz, γxz, φ) = M⁻ᵀ γ, M the columns
//   (sin_e, cos_e, 1) of the edge directions in the frame of the median from the corner opposite the shortest edge
//   (LX its length, LY = 2 A_e / LX); K_edges = A_e Bᵀ Z⁻¹ B, B = M⁻ᵀ C, with the compliance
//   Z = diag(LY²/(24 D'22), LX²/(24 D'11), A_e/(2 √(D'11 D'22))) + the transverse shear compliance of MID3 (the 2 × 2
//   block of 1/(t κ G) on γyz, γxz and its mean on φ), D' the bending stiffness in that frame.
//
// There is no membrane-bending coupling: the family refuses a section with one (MID4, a laminate with B ≠ 0).
//
// Thermal loads (as in ShellQuad4DrillingPlate): the thermal resultants N_T, M_T of the section at t̄ give f_m = Σ (A_e/3) B_mᵀ
// N_T over the three points of the membrane's rule (B_m the strain matrix of ShellTria3DrillingMembrane) and f_b = −A_e
// B_bᵀ M_T (the initial curvature κ_T = −D⁻¹ M_T in the curvature sign of ShellQuad4DrillingBending).
//
// t̄ is the thickness at the centroid (the mean of the corner thicknesses): the one-point rules take the thickness
// there. The triangle is never warped.
//
// Thread-safety: immutable once built. A plain value, built per element.
class ShellTria3DrillingPlate
{
public:
    ShellTria3DrillingPlate(const ShellTria3DrillingGeometry& geometry, const ShellSection& section,
                   const std::array<double, 3>& thickness);

    [[nodiscard]] const ShellTria3DrillingGeometry& geometry() const noexcept { return m_geometry; }
    // K18 (18 × 18, local DOFs).
    [[nodiscard]] base::MatN<18> stiffness() const;
    // The differential stiffness on the local DOFs of the membrane resultants (Nx, Ny, Nxy) of the preload
    // (ShellDifferential: the gradients of the linear shape functions are constant, K_d = A_e·∇Nᵀ·N·∇N).
    [[nodiscard]] base::MatN<18> differentialStiffness(const std::array<double, 3>& membraneForce) const;
    // dK18/dT with respect to the ShellProp thickness T when every corner thickness is proportional to it (t_i =
    // r_i·T), in closed form: with t̄ ∝ T, the membrane ∝ T, the constant curvatures ∝ T³, and the edge shear K_edges =
    // A_e·Cᵀ·Z⁻¹·C whose compliance Z = Z_s + Z_b holds the transverse shear part Z_s ∝ 1/T and the residual bending
    // part Z_b ∝ 1/T³, so that
    //   d(Z⁻¹)/dT = Z⁻¹·(Z_s + 3·Z_b)·Z⁻¹/T.
    [[nodiscard]] base::MatN<18> thicknessRate(double thickness) const;
    // The rates with respect to T of the resultants of the centre state `state` of the local displacements q
    // (centreState) at fixed q, under the same condition (shellResultantRates; the shear forces Z⁻¹·g of the edge
    // shear, their rate d(Z⁻¹)/dT·g).
    [[nodiscard]] ShellPlaceState centreStateRates(const ShellPlaceState& state, const std::array<double, 18>& q,
                                                   double thickness) const;
    // dM18/dT under the same condition, `areaMass` as for mass(): the translational mass ∝ T plus the NSM, lumped by
    // HRZ (the scale total/Σ diagonal, a ratio of two affine functions of T, differentiated as such), the rotary
    // inertia ∝ T³.
    [[nodiscard]] base::MatN<18> massThicknessRate(MassFormulation formulation, double areaMass,
                                                   double thickness) const;
    // dM18 along a rate `densityRate` of the section's density at fixed thicknesses (a laminate's ρ = Σρt/T along a ply
    // thickness): the translational mass and the total linear in ρ, the HRZ scale total/Σ diagonal differentiated as a
    // ratio, the rotary inertia ρ t³/12.
    [[nodiscard]] base::MatN<18> massDensityRate(MassFormulation formulation, double areaMass,
                                                 double densityRate) const;
    // d²K18/dT² under the same condition: the membrane has none, the curvatures 6·K_curvature/T², and with W = Z⁻¹,
    // Y = Z_s + 3·Z_b,
    //   d²W/dT² = W·(2·Y·W·Y − 2·Z_s − 12·Z_b)·W/T²
    // (W ∝ T where Z_b = 0, ∝ T³ where Z_s = 0) the edge shear A_e·Cᵀ·(d²W/dT²)·C.
    [[nodiscard]] base::MatN<18> thicknessSecondRate(double thickness) const;
    // The second rates of the resultants of the centre state at fixed q (shellResultantSecondRates; the shear forces
    // d²W/dT²·g).
    [[nodiscard]] ShellPlaceState centreStateSecondRates(const ShellPlaceState& state, const std::array<double, 18>& q,
                                                         double thickness) const;
    // d²M18/dT²: lumped (HRZ, m_ii = M_ii·s, s = total/Σ, every one affine in T) 2·M'_ii·s' + M_ii·s'' with
    // s'' = −2·s'·Σ'/Σ; coupled, the translations none and the rotary inertia ρ t̄³/(2 T²).
    [[nodiscard]] base::MatN<18> massThicknessSecondRate(MassFormulation formulation, double areaMass,
                                                         double thickness) const;

    // The rates of the plate along a rate of its section at fixed geometry and thickness (the AnisoMat2D terms of
    // ShellSectionRate): the membrane is its stiffness at the rates (ShellTria3DrillingMembrane is linear in C and G
    // together), the constant curvatures are linear in D (their part at the rate); the edge shear K_edges = A_e·Cᵀ·W·C,
    // W = Z⁻¹, through the inverse, dW = −W·dZ·W, with dZ of the residual bending compliances L²/(24 D') and A_e/(2
    // √(D'11 D'22)) (dD' = vᵀ dD v) and of the transverse shear compliance of MID3 (ShellSectionRate's, over t̄, in the
    // median frame).
    struct SectionRate
    {
        base::MatN<18> stiffness;
        base::Mat3 rigidity; // dW of the edge shear
    };
    [[nodiscard]] SectionRate sectionRate(const ShellSectionRate& rate) const;
    // The rates of the centre state of the local displacements q (centreState without a temperature change: `state`)
    // at fixed q along `rate` (`plateRate` its sectionRate): the strains and the curvatures do not move,
    // dN = t̄ dG1 ε, dM = −dD κ, the shear forces dW·C·u; no thickness rate.
    [[nodiscard]] ShellPlaceState centreStateSectionRates(const ShellPlaceState& state, const std::array<double, 18>& q,
                                                          const ShellSectionRate& rate,
                                                          const SectionRate& plateRate) const;

    // The state at the centre (the membrane strains at the centroid, constant curvatures), node 0, under the
    // temperature change of the element (zero: none). Without a membrane (MID1 blank) the strains are those of the
    // linear (u, v) field, for the strains at the fibres. The transverse shear forces are the first two components of
    // Z⁻¹ g, the forces conjugate to (γyz, γxz) in K_edges, carried from the median frame to the element frame.
    [[nodiscard]] ShellPlaceState centreState(const std::array<double, 18>& q,
                                              const ShellTemperatureChange& temperature = {}) const;
    // centreState in parts: what does not depend on the displacements nor on the temperature (CentreOperators: the
    // strain matrices, the bending stiffness at the centre), formed once for many displacements (the
    // unit displacements of the recovery slopes), and the thermal resultants of the temperature;
    // centreState(operators, resultants, q, temperature) gives the bits of centreState(q, temperature).
    struct CentreOperators
    {
        base::MatN<3, 9> membrane;     // ShellTria3DrillingMembrane at the centroid (with a membrane)
        base::MatN<3, 6> linearStrain; // the strains of the linear (u, v) field (without a membrane)
        base::MatN<3, 9> curvature;
        base::Mat3 bending; // D (with bending)
    };
    [[nodiscard]] CentreOperators centreOperators() const;
    [[nodiscard]] ThermalResultants centreThermal(const ShellTemperatureChange& temperature) const;
    [[nodiscard]] ShellPlaceState centreState(const CentreOperators& operators, const ThermalResultants& thermal,
                                              const std::array<double, 18>& q,
                                              const ShellTemperatureChange& temperature) const;

    // The mass on the local DOFs: lumped HRZ translations, or the consistent translations and the rotary inertia
    // ρ t̄³/12 on θx, θy; `areaMass` the non-structural mass per unit area added to the section's.
    [[nodiscard]] base::MatN<18> mass(MassFormulation formulation, double areaMass) const;
    // The equivalent nodal loads of a pressure linear between the corner pressures p_i: F_i = ∫ N_i p dA =
    // A_e (2 p_i + p_j + p_k) / 12 (exact), along +n or along the unit vector `direction` of the element frame.
    [[nodiscard]] std::array<double, 18> pressureLoads(const std::array<double, 3>& cornerPressures,
                                                       const std::optional<base::Vec3>& direction = {}) const;
    // The equivalent nodal loads of a temperature change (see the class comment).
    [[nodiscard]] std::array<double, 18> thermalLoads(const ShellTemperatureChange& temperature) const;

private:
    // The strains (εx, εy, γxy) of the linear (u, v) field (3 × 6 over u, v per corner).
    [[nodiscard]] base::MatN<3, 6> linearStrainMatrix() const;
    [[nodiscard]] base::MatN<3, 9> curvatureMatrix() const;
    // Builds m_edgeShear and m_shearForce from the bending stiffness D at the centroid.
    void setEdgeShear(const base::Mat3& d);
    // d²W/dT² of the edge shear's rigidity W = Z⁻¹ (thicknessSecondRate).
    [[nodiscard]] base::Mat3 rigiditySecondRate(double thickness) const;

    ShellTria3DrillingGeometry m_geometry;
    ShellSection m_section;
    std::array<double, 3> m_thickness{};
    double m_centreThickness = 0.0;
    base::MatN<9> m_edgeShear;     // K_edges on the bending DOFs (w, θx, θy per corner)
    base::MatN<2, 9> m_shearForce; // (Qx, Qy) in the element frame from the bending DOFs
    // The parts of K_edges for its thickness rate: C of the median frame (3 × 9), Z⁻¹, and Z = Z_s + Z_b.
    base::MatN<3, 9> m_edgeCentre;
    base::Mat3 m_edgeRigidity;
    base::Mat3 m_shearCompliance;
    base::Mat3 m_bendingCompliance;
    double m_medianCos = 1.0; // the median frame of the edge shear (t3setd.f)
    double m_medianSin = 0.0;
    double m_medianLengthX = 0.0; // LX, LY of the edge shear (setEdgeShear)
    double m_medianLengthY = 0.0;
    double m_bendingAlongX = 0.0; // D'11, D'22 of the edge shear
    double m_bendingAlongY = 0.0;
    std::optional<ShellTria3DrillingMembrane> m_membrane;
};

} // namespace oos::elements
