#pragma once

// The constitutive data of the ShellQuad4 and ShellTria3 shells of the NASTRAN-95 formulation (ShellQuad4, ShellTria3):
// the four plate matrices of a section and what the transverse shear of these elements reads, templated on the scalar
// so that the design derivatives can be taken by the complex step (ShellScalar.hpp).
//
// Reference: NASTRAN-95 (the program of NASA, open-sourced in 2015), routines premat (the material lookup),
// q4gmgs/shgmgd (the material matrices: "spec §1.2"), quad4d (the moduli of the QUAD4 shear-correction factors:
// "spec §2.4") and the material-axis code of quad4d and shcsgd ("spec §1.3"); the thermal strains of tlqd4d and tltr3d
// ("spec §6").
//
// The four matrices, per unit thickness, act on the engineering strains (εx, εy, γxy):
//   G1  membrane (MID1),  G2 bending (MID2, scaled by 12I/T³ in the elements),  G4 membrane-bending coupling (MID4),
//   G3  the transverse shear on (γxz, γyz) (MID3).
// A CompProp or CompPropGlobal drives the same matrices with its equivalent section (ShellSection: G1 = A/T, G2 = 12
// D/T³, G4 = −B/T², the transverse shear H/T): this project's choice, since how NASTRAN-95 forms the equivalent
// materials of a laminate was not specified.
//
// Deviation from NASTRAN-95 (deliberate): NASTRAN-95 multiplies γyz by G1Z and γxz by G2Z of a OrthoMat2D (its two
// words are swapped against the card's order). This implementation takes the physical reading of the card, G1Z on γxz
// and G2Z on γyz, as ShellSection already does.

#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellScalar.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace oos::elements
{

// The card that gives the bending material: the QUAD4 shear-correction factors read its moduli the way the card
// defines them (spec §2.4: E of a IsoMat, the raw G11 and G22 of a AnisoMat2D, E1 and E2 of a OrthoMat2D). A laminate
// counts as a AnisoMat2D (its equivalent G2 is a general matrix).
enum class BendingCard : std::uint8_t
{
    Isotropic,  // IsoMat
    General,    // AnisoMat2D, or a laminate
    Orthotropic // OrthoMat2D
};

// A shell section in its material frame, as the kernels read it.
template <typename T>
struct PlateSectionValues
{
    bool hasMembrane = false;
    bool hasBending = false;
    bool hasCoupling = false;
    bool hasShearFlexibility = false; // MID3 given with a positive modulus (or a laminate's equilibrium shear)
    bool isLaminate = false;
    BendingCard bendingCard = BendingCard::Isotropic;
    Matrix3<T> membrane; // G1
    Matrix3<T> bending;  // G2
    Matrix3<T> coupling; // G4, in the sign of a ShellProp MID4 (NASTRAN-95's moments M = −∫σ z dz)
    // κ·G per unit thickness of the transverse shears along the material axes x, y (MID3, κ = TS/T); 0 for a direction
    // left rigid.
    std::array<T, 2> transverseShear{};
    // The transverse shear moduli of a OrthoMat2D bending card (its G1Z, G2Z, G12 where blank): the shear without MID3.
    std::array<double, 2> bendingCardShear{};
    T bendingRatio = T(SHELL_PROP_DEFAULT_BENDING_RATIO); // 12I/T³
    T shearRatio = T(SHELL_PROP_DEFAULT_SHEAR_RATIO);     // TS/T
    T density{};
    T nonStructuralMass{};
};

// The section values of a resolved section (in its material frame: not rotated), `bendingMaterial` the MID2 card of a
// ShellProp with bending (null for a laminate or without bending).
[[nodiscard]] PlateSectionValues<double> plateSectionValuesOf(const ShellSection& section,
                                                              const ShellMaterial* bendingMaterial);
// The rates of the section values along a section rate given in the material frame (shellSectionRate or
// laminateSectionRate with a zero material angle), `densityRate` dρ.
[[nodiscard]] PlateSectionValues<double> plateSectionRatesOf(const PlateSectionValues<double>& values,
                                                             const ShellSectionRate& rate, double densityRate);
// The section values moved along their rates by the complex step.
[[nodiscard]] PlateSectionValues<Complex> complexStepped(const PlateSectionValues<double>& values,
                                                         const PlateSectionValues<double>& rates);

// The constitutive data in the element frame.
template <typename T>
struct PlateConstitutive
{
    bool hasMembrane = false;
    bool hasBending = false;
    bool hasCoupling = false;
    bool hasShearFlexibility = false;
    Matrix3<T> membrane; // G1 in the element frame
    Matrix3<T> bending;  // G2
    Matrix3<T> coupling; // G4 (PSHELL MID4 sign)
    // κ·G of the transverse shear on (γxz, γyz) in the element frame: with MID3 its moduli (a rigid direction
    // taking the in-plane shear modulus of the bending card, see the .cpp); without MID3 TS/T times the bending card's
    // G3, which NASTRAN-95's QUAD4 uses then (spec §1.2).
    Matrix2<T> transverseShear;
    // (κ·G)⁻¹ in the element frame, a rigid direction counting 0; zero without MID3 (the TRIA3 shear compliance).
    Matrix2<T> transverseShearCompliance;
    T bendingRatio{};
    T shearRatio{};
    // Spec §2.4: the moduli of the QUAD4 shear-correction factors, card values along the material axes: Ex, Ey of
    // the bending card, Gx, Gy of MID3 (of the bending card without MID3).
    T youngX{};
    T youngY{};
    T shearModulusX{};
    T shearModulusY{};
};

// The constitutive data in the element frame of a section whose material x axis is at `materialAngle` (radians,
// counter-clockwise about the element normal) from the element x axis: G_e = U3ᵀ G U3 for G1, G2, G4 and
// G3_e = U2ᵀ G3 U2 for the shear (spec §1.3, the in-plane case: U3 the engineering-strain rotation, U2 the rotation of
// the shear-strain vector).
template <typename T>
[[nodiscard]] PlateConstitutive<T> plateConstitutiveOf(const PlateSectionValues<T>& section, double materialAngle);

// U3: the engineering strains (εx, εy, γxy) of a frame turned by `angle` from the strains of the current one.
[[nodiscard]] Matrix3<double> engineeringStrainRotation(double angle);
// The stresses (σx, σy, τxy) of a frame turned by `angle` from those of the current one.
[[nodiscard]] Matrix3<double> stressRotation(double angle);

// The thermal strain of a shell in the element frame: ε(z) = membrane + z·gradient (z along the element normal), the
// strain the element would take freely under its temperature change.
struct PlateThermalStrain
{
    std::array<double, 3> membrane{};
    std::array<double, 3> gradient{};
    bool isZero = true;
};

// The thermal strain of a section (in its material frame, as resolved) whose material axis is at `materialAngle` from
// the element x axis, under `temperature` at a point of thickness t (spec §6: ε_m = α_m ΔT and the gradient α_b T′,
// the expansion coefficients rotated to the element frame). A laminate gives its thermal resultants N_T, M_T, turned
// into the equivalent free strains by the inverse of its stiffness [A B; B D] at that thickness (spec §6.4).
[[nodiscard]] PlateThermalStrain plateThermalStrainOf(const ShellSection& section, double materialAngle,
                                                      const PlateConstitutive<double>& constitutive,
                                                      const ShellTemperatureChange& temperature, double thickness);

} // namespace oos::elements
