#pragma once

#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The section law of a shell of the mixed hybrid formulation: membrane, coupling and bending stiffness (A, B, D), the
// transverse shear stiffness H, the inertia and the thermal resultants, given in the material axes of the section and
// turned into the local tangent frame of the surface point by point.
//
// Conventions (every quantity of this header):
// - strains [e11, e22, γ12, κ11, κ22, 2 κ12] with engineering shear, transverse shear strains [γ1, γ2];
// - forces N = A (e − e_0) + B (κ − κ_0), M = Bᵀ (e − e_0) + D (κ − κ_0), Q = H γ;
// - thermal resultants per unit temperature rise T0 and per unit through-thickness gradient T1:
//   N_T = thermalA T0 + thermalB T1, M_T = thermalB T0 + thermalD T1;
// - inertia (I0, I1, I2) = ∫ ρ (1, z, z²) dz about the reference surface of the section.
// The laminate is classical lamination theory (R. M. Jones, Mechanics of Composite Materials, 2nd ed., Taylor &
// Francis, 1999, chapter 4). Its transverse shear stiffness is that of the equilibrium distribution of the shear
// stress through the thickness in two cylindrical bendings (J. M. Whitney, Shear correction factors for orthotropic
// laminates under static load, Journal of Applied Mechanics 40 (1973) 302-304): 5/6 G t on a single layer, the core's
// on a sandwich.
//
// The inputs are plain numbers (moduli, plies, thicknesses): the adapters from the solver's shell properties are
// written by the integrator.
//
// Reference: section.rs (the whole file).
//
// Thread-safety: plain value types; every function is pure. ShellSection is immutable once built and may be read from
// several threads.

// The parts of the section law a shell property gives (docs/mixed-hybrid-formulation-spec.md section 4.1; not in the
// reference): both, the membrane alone (no bending material: B = D = 0, no transverse shear) or the bending alone (no
// membrane material: A = B = 0).
enum class SectionParts : std::uint8_t
{
    MembraneAndBending,
    MembraneOnly,
    BendingOnly
};

// The section at one point, in the local tangent frame (e1, e2) of the surface.
// Reference: section.rs::SectionPoint
struct SectionPoint
{
    SectionParts parts = SectionParts::MembraneAndBending;
    Matrix33 a{};
    Matrix33 b{};
    Matrix33 d{};
    Matrix22 h{};
    // I0, I1, I2 about the reference surface.
    std::array<double, 3> mass{};
    // Thermal resultants: N_T = thermalA T0 + thermalB T1, M_T = thermalB T0 + thermalD T1 (Reference: ta, tb, td).
    Voigt3 thermalA{};
    Voigt3 thermalB{};
    Voigt3 thermalD{};
    double thickness = 0.0;
};

// One layer of a section, for the recovery of the stresses of the plies.
// Reference: section.rs::Ply
struct Ply
{
    // Reduced stiffness in the material axes of the section (already turned by the ply angle).
    Matrix33 q{};
    // Thermal expansion in the material axes of the section [a11, a22, 2 a12].
    Voigt3 alpha{};
    // Bottom and top of the layer, from the middle of the section.
    double z0 = 0.0;
    double z1 = 0.0;
    // Angle of the ply from the material x axis of the section, radians.
    double angle = 0.0;
    // Transverse shear moduli in the material axes of the section, on (γ_xz, γ_yz); zero for a layer rigid in
    // transverse shear.
    Matrix22 shearModuli{};
};

// The transverse shear stress through one layer per unit shear force, in the material axes of the section:
//   τ(z) = (g0 + f0 (z − z0) + f1 (z² − z0²) / 2) Q   on [z0, z1] of the layer,
// (τ_xz, τ_yz) for (Q_x, Q_y). It equilibrates the gradient of the in-plane stresses of two cylindrical bendings, one
// along each axis (M_x,x = Q_x, M_y,y = Q_y, the twisting moment uniform, no membrane force).
// Reference: section.rs::PlyShear
struct PlyShear
{
    Matrix22 g0{};
    Matrix22 f0{};
    Matrix22 f1{};

    // The 2 × 2 matrix of τ(z) per unit shear force, z0 the bottom of the layer.
    [[nodiscard]] Matrix22 at(double z0, double z) const;
};

// One layer of a laminate, from the bottom of the section to its top.
// Reference: the [f64; 11] rows of section.rs::Section::laminate
struct LaminateLayer
{
    double e1 = 0.0;
    double e2 = 0.0;
    double nu12 = 0.0;
    double g12 = 0.0;
    // Transverse shear moduli; a layer with either zero is rigid in transverse shear.
    double g13 = 0.0;
    double g23 = 0.0;
    double density = 0.0;
    double alpha1 = 0.0;
    double alpha2 = 0.0;
    double thickness = 0.0;
    // Radians, from the material x axis of the section.
    double angle = 0.0;
};

// The equilibrium distribution of the transverse shear stress (one PlyShear per layer) and the transverse shear
// stiffness it implies (empty when every layer is rigid in transverse shear).
struct ShearDistribution
{
    std::vector<PlyShear> plies;
    std::optional<Matrix22> stiffness;
};

// A section in its material axes at the reference thickness.
// Reference: section.rs::Section (the parts are not in the reference: it takes both)
struct Section
{
    SectionParts parts = SectionParts::MembraneAndBending;
    Matrix33 a{};
    Matrix33 b{};
    Matrix33 d{};
    // Empty: no transverse shear flexibility (the Kirchhoff limit).
    std::optional<Matrix22> h;
    double referenceThickness = 0.0;
    double i0 = 0.0;
    double i1 = 0.0;
    double i2 = 0.0;
    // The non-structural mass per unit area (it does not scale with the thickness).
    double nonStructuralMass = 0.0;
    Voigt3 thermalA{};
    Voigt3 thermalB{};
    Voigt3 thermalD{};
    // Distance from the surface of the grid points to the reference surface of the section, along the normal.
    double offset = 0.0;
    std::vector<Ply> plies;
    // The distribution of the transverse shear stress through each layer, empty when the section has no transverse
    // shear flexibility.
    std::vector<PlyShear> shear;
    // (shear stiffness used in the Kirchhoff limit) / (mean membrane stiffness); a section without membrane stiffness
    // takes the membrane stiffness 12 D / t² of its bending (that of a homogeneous plate).
    double kirchhoffRatio = 1e4;

    // A homogeneous isotropic section of one material: A = Q t, D = Q t³ / 12, H = k G t (k the shear factor),
    // N_T = E α t / (1 − ν) T0, M_T = E α t³ / (12 (1 − ν)) T1.
    [[nodiscard]] static Section isotropic(double youngModulus, double poissonRatio, double thickness,
                                           double shearFactor, double density, double expansion);

    // A laminate by classical lamination theory about its middle plane, the layers from the bottom (z = −t/2) to the
    // top, with the equilibrium transverse shear stiffness (shearDistribution). If every layer is rigid in transverse
    // shear, the section has none.
    [[nodiscard]] static Section laminate(std::span<const LaminateLayer> layers);

    // The equilibrium distribution of the transverse shear stress: for a unit moment without membrane force the
    // in-plane stress of a layer is Q_ply (β + z δ) M, (β, δ) the coupling and bending blocks of [A B; Bᵀ D]⁻¹; in
    // two cylindrical bendings the equilibrium τ_α,z = −σ_αβ,β gives τ(z) = G(z) Q, which vanishes on both faces and
    // carries exactly Q. The transverse shear stiffness is the inverse of the complementary energy
    // ∫ Gᵀ G_ply⁻¹ G dz per unit shear force (3-point Gauss rule per layer). Empty if the section has no layer or
    // [A B; Bᵀ D] is singular.
    [[nodiscard]] std::optional<ShearDistribution> shearDistribution() const;

    // The section in the local frame for a thickness t and a material x axis at `angle` (radians) from e1: with
    // r = t / t_ref, A r, B r², D r³, H r (or the Kirchhoff value), every stiffness turned by the angle.
    [[nodiscard]] SectionPoint at(double thickness, double angle) const;
};

// The reduced stiffness of an isotropic layer: E / (1 − ν²) [[1, ν, 0], [ν, 1, 0], [0, 0, (1 − ν) / 2]].
// Reference: section.rs::q_isotropic
[[nodiscard]] Matrix33 isotropicReducedStiffness(double youngModulus, double poissonRatio);

// The reduced stiffness of an orthotropic layer turned by `angle` (radians) from the section x axis
// (ν21 = ν12 E2 / E1, the denominator 1 − ν12 ν21).
// Reference: section.rs::q_orthotropic
[[nodiscard]] Matrix33 orthotropicReducedStiffness(double e1, double e2, double nu12, double g12, double angle);

// The strain transformation of axes turned by `angle`: the strains in the turned axes are T ε (engineering shear),
//   T = [[c², s², c s], [s², c², −c s], [−2 c s, 2 c s, c² − s²]].
// Reference: section.rs::strain_t
[[nodiscard]] Matrix33 strainTransformation(double angle);

// Tᵀ M T: a stiffness given in axes turned by `angle`, in the current axes.
// Reference: section.rs::rotate_stiffness
[[nodiscard]] Matrix33 rotateStiffness(const Matrix33& m, double angle);

// Tᵀ v: a stress-like vector given in axes turned by `angle`, in the current axes.
// Reference: section.rs::rotate_resultant
[[nodiscard]] Voigt3 rotateResultant(const Voigt3& v, double angle);

// The section as a field over the element: what the kernel asks at each quadrature point.
// Reference: section.rs::SectionField
class SectionField
{
public:
    virtual ~SectionField() = default;
    // The section at the reference point (u, v) whose local tangent frame is `frame`.
    [[nodiscard]] virtual SectionPoint at(double u, double v, const SurfaceFrame& frame) const = 0;

protected:
    SectionField() = default;
    SectionField(const SectionField&) = default;
    SectionField& operator=(const SectionField&) = default;
};

// A section with a bilinear thickness (the values at the four corners of the reference square; a triangle of corner
// thicknesses t0, t1, t2 passes [t0, t1, t1 + t2 − t0, t2], whose bilinear interpolation is the linear one on the
// reference triangle) and a material direction given as a vector of the space, projected on the tangent plane point
// by point.
// Reference: section.rs::ShellSection
class ShellSection final : public SectionField
{
public:
    ShellSection() = default;
    ShellSection(Section section, const std::array<double, 4>& thickness, const SpaceVector& direction);
    // The thickness of the section at every corner.
    [[nodiscard]] static ShellSection uniform(Section section, const SpaceVector& direction);

    [[nodiscard]] const Section& section() const { return m_section; }
    [[nodiscard]] const std::array<double, 4>& thickness() const { return m_thickness; }
    [[nodiscard]] const SpaceVector& direction() const { return m_direction; }

    // t(u, v) = ¼ Σ (1 ± u)(1 ± v) t_k, the bilinear interpolation of the corner thicknesses on [−1, 1]².
    [[nodiscard]] double thicknessAt(double u, double v) const;
    // The angle from e1 to the material x axis: atan2(e2 · d, e1 · d).
    [[nodiscard]] double angle(const SurfaceFrame& frame) const;

    [[nodiscard]] SectionPoint at(double u, double v, const SurfaceFrame& frame) const override;

private:
    Section m_section;
    std::array<double, 4> m_thickness{};
    SpaceVector m_direction{};
};

} // namespace oos::elements::mixedHybrid
