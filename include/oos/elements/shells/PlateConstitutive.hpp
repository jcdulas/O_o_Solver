#pragma once

// The constitutive data of the ShellQuad4 and ShellTria3 (docs/shell-quad4-formulation.md, section 1;
// docs/shell-tria3-formulation.md): the plate matrices of a section, templated on the scalar so that the design
// derivatives can be taken by the complex step (ShellScalar.hpp).
//
// The four matrices, per unit thickness, act on the engineering strains (εx, εy, γxy):
//   G1  membrane (MID1),  G2 bending (MID2, scaled by 12I/T³ in the elements),  G4 membrane-bending coupling (MID4),
//   G3  the transverse shear on (γxz, γyz) (MID3).
// A CompProp or CompPropGlobal drives the same matrices with its equivalent section (ShellSection: G1 = A/T, G2 = 12
// D/T³, G4 = −B/T², the transverse shear H/T).
//
// An OrthoMat2D gives G1Z on γxz and G2Z on γyz, as ShellSection does.

#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellScalar.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace oos::elements
{

// A shell section in its material frame, as the kernels read it.
template <typename T>
struct PlateSectionValues
{
    bool hasMembrane = false;
    bool hasBending = false;
    bool hasCoupling = false;
    bool hasShearFlexibility = false; // MID3 given with a positive modulus (or a laminate's equilibrium shear)
    bool isLaminate = false;
    Matrix3<T> membrane; // G1
    Matrix3<T> bending;  // G2
    Matrix3<T> coupling; // G4, in the sign of a ShellProp MID4 (moments M = −∫σ z dz)
    // κ·G per unit thickness of the transverse shears along the material axes x, y (MID3, κ = TS/T); 0 for a direction
    // left rigid.
    std::array<T, 2> transverseShear{};
    T bendingRatio = T(SHELL_PROP_DEFAULT_BENDING_RATIO); // 12I/T³
    T shearRatio = T(SHELL_PROP_DEFAULT_SHEAR_RATIO);     // TS/T
    T density{};
    T nonStructuralMass{};
};

// The section values of a resolved section (in its material frame: not rotated).
[[nodiscard]] PlateSectionValues<double> plateSectionValuesOf(const ShellSection& section);
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
    // The angle of the material x axis from the element x axis (radians): the kernels whose points have a frame of
    // their own (ShellQuad4 on a warped element) turn the data from the element frame to it.
    double materialAngle = 0.0;
    Matrix3<T> membrane; // G1 in the element frame
    Matrix3<T> bending;  // G2
    Matrix3<T> coupling; // G4 (ShellProp MID4 sign)
    // κ·G of the transverse shear on (γxz, γyz) in the element frame, per unit thickness, when the section has a
    // transverse shear flexibility (one direction left rigid: see the .cpp); zero otherwise, the kernels taking
    // their own stiffness then (EdgeShearShellSection.hpp).
    Matrix2<T> transverseShear;
    T bendingRatio{};
    T shearRatio{};
};

// The constitutive data in the element frame of a section whose material x axis is at `materialAngle` (radians,
// counter-clockwise about the element normal) from the element x axis: G_e = U3ᵀ G U3 for G1, G2, G4 and
// G3_e = U2ᵀ G3 U2 for the shear (U3 the engineering-strain rotation, U2 the rotation of the shear-strain vector).
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
// the element x axis, under `temperature` at a point of thickness t (ε_m = α_m ΔT and the gradient α_b T′, the
// expansion coefficients rotated to the element frame). A laminate gives its thermal resultants N_T, M_T, turned
// into the equivalent free strains by the inverse of its stiffness [A B; B D] at that thickness.
[[nodiscard]] PlateThermalStrain plateThermalStrainOf(const ShellSection& section, double materialAngle,
                                                      const PlateConstitutive<double>& constitutive,
                                                      const ShellTemperatureChange& temperature, double thickness);

} // namespace oos::elements
