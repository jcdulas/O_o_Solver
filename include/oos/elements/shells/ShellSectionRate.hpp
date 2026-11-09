#pragma once

// The rate of a shell section with respect to one stiffness term of a AnisoMat2D of its ShellProp: what the
// closed-form derivatives of the shells in their section terms (ShellQuad4DrillingPlate::sectionRate,
// ShellTria3DrillingPlate::sectionRate) differentiate along. A AnisoMat2D term enters the section as written (ShellSection.cpp:
// G1 = the MID1 matrix, G2 = the MID2 matrix, B/t² = −G4 of the MID4, the G = G33 of MID1 of the drilling membranes'
// rotation penalty, κG = TS/T·G11 and TS/T·G22 of the MID3 along the material axes), so its rate is a unit matrix (or
// unit modulus) in each part the material acts on, turned into the element frame as the section is
// (ShellSection::rotatedToElementFrame, linear in the matrices).
//
//   | AnisoMat2D | MID | G11 | G12 | G13 | G22 | G23 | G33 | RHO | ... |    stiffness items 0 to 5
//
// The elements are not linear in these terms (the hybrid curvatures through D⁻¹, the saturated edge shear): the plates
// chain the rate through each of them exactly.

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/DesignFieldUse.hpp"
#include "oos/elements/shells/LaminateRate.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <array>
#include <optional>

namespace oos::elements
{

// The derivatives of the parts of a ShellSection (in the element frame) along one designed field; zero where the
// field does not act.
struct ShellSectionRate
{
    base::Mat3 membraneMatrix;         // dG1
    base::Mat3 bendingMatrix;          // dG2
    base::Mat3 couplingMatrix;         // d(B/t²), the lamination theory's sign: −dG4
    double membraneShearModulus = 0.0; // dG of the drilling membranes' rotation penalty (γ = G)
    // d(κ G) per unit thickness along the material axes (ShellSection::transverseShearStiffness).
    std::array<double, 2> transverseShearStiffness{};
    bool movesMembrane = false;
    bool movesBending = false;
    bool movesCoupling = false;
    bool movesShear = false;
};

// The fields of a section of which the shells give these rates: a stiffness term G11 ... G33 of a AnisoMat2D (its
// "stiffness" items 0 to 5) named by its MID.
[[nodiscard]] bool isAnisoMat2DStiffnessTerm(const DesignedField& field) noexcept;

// The rate of `section` (in the element frame: turned by `materialAngle` from the material frame) with respect to the
// AnisoMat2D stiffness term `field`. Nothing where the closed forms do not hold, the caller then differencing the
// facets: a field that is not a AnisoMat2D stiffness term, a laminate, a section none of whose MIDs is the field's
// AnisoMat2D, a MID3 term of a direction that is rigid at the point (G11 or G22 not positive: the shear saturation is
// not differentiable there), a MID4 term of a coupling that is zero at the point on a family that takes no coupling
// (`takesCoupling` false: the drilling membranes of the ShellQuad4Drilling and ShellTria3Drilling refuse a MID4 with a
// coupling, so a move off zero has no element to differentiate).
//
// A MID4 term of a coupling zero at the point (ShellSection::hasCoupling false: the coupling part is not formed there,
// a balanced symmetric start ξB = 0 of the lamination mode) has its rate all the same: the coupling stiffness of a
// plate that takes one is linear in [B], so at [B] = 0 the derivative is the coupling of the rate alone,
// d[B] = t²·dB/t², which such a plate forms whatever hasCoupling.
[[nodiscard]] std::optional<ShellSectionRate> shellSectionRate(const ShellSection& section, const DesignedField& field,
                                                               double materialAngle, bool takesCoupling = true);

// The rate of a laminate's section (ShellSection.cpp's laminateSection: G1 = A/T, G2 = 12 D/T³ with 12I/T³ = 1, B/T²,
// the drilling G = A33/T, κG = H/T along the material axes) along the thickness of one of its card's plies,
// split as the plates differentiate: `atFixedThickness` the rate of the per-unit-thickness section at fixed T (in the
// element frame, as `section`), which their sectionRate takes, and `thickness` the rate dT, which multiplies their
// thicknessRate (corners following T: a laminate's corners are its thickness), so that
//   dK = sectionRate(atFixedThickness) + dT · thicknessRate(T),
//   dG1 = (dA − G1 dT)/T,  dG2 = 12 dD/T³ − 3 G2 dT/T,  d(B/T²) = dB/T² − 2 (B/T²) dT/T,
//   d(κG) = (dH − κG dT)/T,  dG_drill = (dA33 − G_drill dT)/T  (A33 of the material frame, as laminateSection).
// `ofProducts`: the rates of the products a formulation linear in them reads at corner
// thickness T, t·G1 = A, t³/12·G2 = D, t²·B/t² = B, t·κG = H, t·G = A33, written per unit of T, T³/12, T², T and T
// (dA/T, 12 dD/T³, dB/T², dH/T, dA33/T: MindlinShell::stiffnessSectionRate takes the whole rate from them).
// `density`: dρ at fixed T, (d(Σ ρ t) − ρ dT)/T, the mass's rate being that of ρ plus dT · the mass's thickness rate.
// Nothing when `field` is not a ply thickness of the section's laminate (laminateThicknessRate), or it moves a
// coupling zero at the point on a family that takes none (`takesCoupling`, as shellSectionRate).
struct LaminateSectionRate
{
    ShellSectionRate atFixedThickness;
    ShellSectionRate ofProducts;
    double thickness = 0.0;
    double density = 0.0;
    LaminateRate laminate;
};

[[nodiscard]] std::optional<LaminateSectionRate> laminateSectionRate(const ShellSection& section,
                                                                     const DesignedField& field, double materialAngle,
                                                                     bool takesCoupling = true);

// The rates along `rate` of ShellSection::transverseShearCompliance (xx, yy, xy) and of
// ShellSection::transverseShearStiffnessAlongAxes: dC = R diag(−ds₁/s₁², −ds₂/s₂²) Rᵀ (a rigid direction counts 0,
// as its compliance), and d(1/C_xx) = −dC_xx/C_xx² (0 where C_xx is 0).
[[nodiscard]] std::array<double, 3> transverseShearComplianceRate(const ShellSection& section,
                                                                  const ShellSectionRate& rate);
[[nodiscard]] std::array<double, 2> transverseShearStiffnessAlongAxesRate(const ShellSection& section,
                                                                          const ShellSectionRate& rate);

} // namespace oos::elements
