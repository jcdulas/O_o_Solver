#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"

#include <array>

namespace oos::elements
{

// The transverse shear of the bending of a shell: κ·G per unit thickness of the x and y directions (TS/T times G3,
// G1Z or G2Z; H/T of a laminate), 0 for a rigid direction (MID3 blank, a zero modulus).
struct TransverseShear
{
    std::array<double, 2> stiffness{};
};

// The coefficients c̄x, c̄y, cgx, cgy of the edge shear, with the saturation in a flexible direction:
// c → c / (1 + 12 c / (t κ G)), t the thickness at the centre (t κ G = H for a laminate). `widthA`, `widthB` are the
// half-widths A and B of ShellQuad4DrillingGeometry, d11, d22 the bending stiffnesses D11, D22 of the element frame.
struct EdgeShearCoefficients
{
    double meanX = 0.0;     // c̄x = D11 / A²
    double meanY = 0.0;     // c̄y = D22 / B²
    double gradientX = 0.0; // cgx = D11 (1/A² + 39/B²)
    double gradientY = 0.0; // cgy = D22 (1/B² + 39/A²)
};
[[nodiscard]] EdgeShearCoefficients edgeShearCoefficients(double d11, double d22, double widthA, double widthB,
                                                          const TransverseShear& shear, double centreThickness);

// The edge shear of a quadrilateral is defined when its Cartesian edge components are: the bottom and top edges not
// parallel to y, the right and left edges not parallel to x, and the 2 × 2 system of the means not singular. A
// component of a unit edge vector at or below this bound marks a shape the formula cannot treat.
inline constexpr double EDGE_SHEAR_MIN_COMPONENT = 1.0e-6;

// The bending of a quadrilateral plate (ShellQuad4DrillingPlate) on the local DOFs (w, θx, θy) of its four corners, node by
// node, in the element frame: K_bending,12 = K_curvature + Cᵀ S C.
//
// - Curvatures, Hellinger-Reissner with the Pian-Sumihara moment field and the Taylor-Beresford-Wilson factor:
//   rotations bilinear, B_b the curvatures (κx, κy, 2κxy), κx = −∂θy/∂x, κy = ∂θx/∂y, 2κxy = ∂θx/∂x − ∂θy/∂y at the
//   current Jacobian; moments M(ξ, η) = m0 + m4 η r v_ξ + m5 ξ r v_η, r = det J0 / det J(ξ, η), v_ξ = (e_x², e_y²,
//   e_x e_y) with e the first row of J0 normalised, v_η the same with the second row; with C_b = D⁻¹ and P (3 × 5) the
//   moment interpolation, Gauss 2 × 2: H = Σ det J Pᵀ C_b P, Gm = Σ det J Pᵀ B_b, K_curvature = Gmᵀ H⁻¹ Gm.
// - Edge shear: γ_e = (w_b − w_a)/L_e − ½ u_e·(β_a + β_b), β = (−θy, θx), on the four edges G1→G2 (bottom),
//   G2→G3 (right), G3→G4 (top), G4→G1 (left); Cartesian components per edge, corrected by the self-consistent means
//   γ̄x, γ̄y, and the gradients gx, gy; γᵀ S γ = 12 𝒜 (c̄x γ̄x² + c̄y γ̄y²) + 4 𝒜 (cgx gx² + cgy gy²), 𝒜 = 4 det J0.
//
// D = (12I/T³)·t³/12·G2 follows the thickness t(ξ, η) = Σ N_i t_i at each Gauss point; the edge-shear coefficients
// take D at the centre thickness.
//
// Thread-safety: immutable once built. A plain value.
class ShellQuad4DrillingBending
{
public:
    // `bendingMatrix` is G2 (per unit D / ((12I/T³)·t³/12)), in the element frame.
    // Precondition: hasEdgeShearFrame(geometry).
    ShellQuad4DrillingBending(const ShellQuad4DrillingGeometry& geometry, const base::Mat3& bendingMatrix, double bendingRatio,
                     const std::array<double, 4>& thickness, const TransverseShear& shear);

    [[nodiscard]] static bool hasEdgeShearFrame(const ShellQuad4DrillingGeometry& geometry);

    // K_bending,12 (12 × 12).
    [[nodiscard]] const base::MatN<12>& stiffness() const noexcept { return m_stiffness; }
    // The moments of the hybrid field, conjugate to the curvatures (M = D·κ), at (ξ, η) for the bending DOFs u12;
    // `initial` the moment parameters an initial curvature takes off (initialCurvatureParameters), zero without one.
    [[nodiscard]] std::array<double, 3> momentAt(double xi, double eta, const std::array<double, 12>& u12,
                                                 const std::array<double, 5>& initial = {}) const;
    // momentAt in two parts, so that a caller evaluating many u12 forms the interpolation of a point once: the moment
    // parameters H⁻¹·Gm·u12 − initial, then the moment P(ξ, η)·parameters of the interpolation of the point
    // (momentInterpolationAt); the bits of momentAt.
    [[nodiscard]] std::array<double, 5> momentParameters(const std::array<double, 12>& u12,
                                                         const std::array<double, 5>& initial) const;
    [[nodiscard]] base::MatN<3, 5> momentInterpolationAt(double xi, double eta) const
    {
        return momentInterpolation(xi, eta);
    }
    // The curvatures of the hybrid field at (ξ, η): κ = D(t(ξ, η))⁻¹·M.
    [[nodiscard]] std::array<double, 3> curvatureAt(double xi, double eta, const std::array<double, 12>& u12) const;

    // An initial curvature κ_T(ξ, η) (the sign of the curvatures above; a thermal one) given at the 2 × 2 Gauss
    // points enters the Hellinger-Reissner functional as ∫ Mᵀ (κ(u) − κ_T): the moment parameters become
    // m = H⁻¹ (Gm u − g_T), g_T = Σ_g w det J Pᵀ κ_T, and the nodes are loaded by Gmᵀ H⁻¹ g_T.
    struct InitialCurvatureLoads
    {
        std::array<double, 12> nodal{};
        std::array<double, 5> parameters{}; // H⁻¹ g_T
    };
    [[nodiscard]] InitialCurvatureLoads initialCurvatureLoads(
        const std::array<std::array<double, 3>, 4>& atGaussPoints) const;
    // The transverse shear forces (Qx, Qy) = (12 c̄x γ̄x, 12 c̄y γ̄y), the forces conjugate to the mean edge shears in
    // γᵀ S γ / 2 (constant over the element).
    [[nodiscard]] std::array<double, 2> shearForces(const std::array<double, 12>& u12) const;
    // D at (ξ, η).
    [[nodiscard]] base::Mat3 bendingStiffnessAt(double xi, double eta) const;

    // dK_bending,12/dT with respect to a thickness T every corner thickness is proportional to: D ∝ T³ at every point,
    // so K_curvature ∝ T³ (H ∝ T⁻³); each edge-shear coefficient is c = c_r/(1 + 12·c_r/(t κ G)) of an unsaturated
    // c_r ∝ T³ and a centre thickness t ∝ T, whose rate is dc/dT = (c/T)·(1 + 2·c/c_r) (3c/T in a rigid direction,
    // where c = c_r; c/T when the shear saturates):
    //   dK/dT = 3·K_curvature/T + Cᵀ·(dS/dT)·C.
    [[nodiscard]] base::MatN<12> thicknessRate(double thickness) const;
    // d(Qx, Qy)/dT of shearForces(u12) at fixed u12 under the same condition: 12·(dc̄/dT)·γ̄.
    [[nodiscard]] std::array<double, 2> shearForceRates(const std::array<double, 12>& u12, double thickness) const;
    // d²K_bending,12/dT² under the same condition: K_curvature ∝ T³ and, for each coefficient with
    // 1/c = 1/c_r + 12/(t κ G) of degrees −3 and −1 in T and q = c/c_r,
    //   d²c/dT² = 2·c·q·(4q − 1)/T²
    // (6c/T² in a rigid direction, 0 once the shear saturates fully):
    //   d²K/dT² = 6·K_curvature/T² + Cᵀ·(d²S/dT²)·C.
    [[nodiscard]] base::MatN<12> thicknessSecondRate(double thickness) const;
    // d²(Qx, Qy)/dT² at fixed u12: 12·(d²c̄/dT²)·γ̄.
    [[nodiscard]] std::array<double, 2> shearForceSecondRates(const std::array<double, 12>& u12,
                                                              double thickness) const;

    // The rates along a rate of the section: `bendingMatrixRate` dG2 in the element frame, `shearRate` the rates of the
    // κ·G per unit thickness along the element axes (TransverseShear). The curvature part through the inverses,
    //   dH = −Σ_g w det J Pᵀ D⁻¹ dD D⁻¹ P,  d(H⁻¹ Gm) = −H⁻¹ dH (H⁻¹ Gm),  dK_curvature = −(H⁻¹ Gm)ᵀ dH (H⁻¹ Gm),
    // dD = (12I/T³) t³/12 dG2 at each point; the edge shear Cᵀ S(dc) C with each coefficient's rate through its
    // saturation c = c_r/(1 + 12 c_r/g), g = t κG:
    //   dc = q² dc_r + 12 (c/g)² t d(κG),  q = g/(g + 12 c_r)   (dc = dc_r in a rigid direction),
    // dc_r of the unsaturated coefficients from dD11, dD22 at the centre.
    struct SectionRate
    {
        base::MatN<12> stiffness;
        base::MatN<5, 12> momentParameters; // d(H⁻¹ Gm)
        EdgeShearCoefficients coefficients; // dc̄x, dc̄y, dcgx, dcgy
    };
    // `withStiffness` false leaves `stiffness` zero (the rates of the states need the other parts only).
    [[nodiscard]] SectionRate sectionRate(const base::Mat3& bendingMatrixRate, const TransverseShear& shearRate,
                                          bool withStiffness = true) const;
    // The rates of shearForces(u12) at fixed u12 for the coefficients' rates: 12·(dc̄x γ̄x, dc̄y γ̄y).
    [[nodiscard]] std::array<double, 2> shearForceRatesOf(const EdgeShearCoefficients& coefficientRates,
                                                          const std::array<double, 12>& u12) const;

private:
    [[nodiscard]] base::MatN<3, 5> momentInterpolation(double xi, double eta) const;
    // The edge-shear form S for the coefficients `c` (4 × 4 over the edge shears γ0 ... γ3).
    [[nodiscard]] base::MatN<4> edgeShearForm(const EdgeShearCoefficients& c) const;

    ShellQuad4DrillingGeometry m_geometry;
    base::Mat3 m_bendingMatrix;
    double m_bendingRatio = 0.0;
    std::array<double, 4> m_thickness{};
    base::MatN<12> m_stiffness;
    base::MatN<5, 12> m_momentParameters; // H⁻¹ Gm: the five moment parameters of u12
    base::MatN<5> m_momentFlexibility;    // H⁻¹
    base::MatN<5, 12> m_gm;               // Gm
    base::MatN<2, 12> m_meanShears;       // (γ̄x, γ̄y) of u12
    EdgeShearCoefficients m_coefficients;
    // The parts of the thickness rate: K_curvature, the coefficients before the saturation (c_r), C and the rows of
    // the mean shears and gradients over (γ0 ... γ3).
    base::MatN<12> m_curvatureStiffness;
    EdgeShearCoefficients m_unsaturatedCoefficients;
    base::MatN<4, 12> m_edgeShears;
    std::array<std::array<double, 4>, 4> m_shearRows{}; // γ̄x, γ̄y, gx, gy
    // The transverse shear and the centre thickness of the saturation (the section rates).
    TransverseShear m_shear;
    double m_centreThickness = 0.0;
};

} // namespace oos::elements
