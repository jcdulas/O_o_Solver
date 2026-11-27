#pragma once

// A laminate of the shells (CompProp, CompPropGlobal): the fields of the property (bottomOffset, nonStructMass,
// bondAllowable, failureTheory, referenceTemperature, laminateOption) and, for each ply from the bottom, plyMatId,
// plyThickness, plyAngle and plyStressOutput.
//
// The stack comes from laminatePliesTable (the plies from the bottom with their z from the reference plane,
// bottomOffset applied, the mirrored half of laminateOption = Symmetry included), the option from laminateSummary, the
// other fields from the CompProp or CompPropGlobal table. The stiffness is the classical lamination theory (Jones,
// Mechanics of Composite Materials, §4.2), sums in increasing ply order:
//
//   A = Σ Q̄_k t_k,  B = ½ Σ Q̄_k (z_k² − z_{k−1}²),  D = ⅓ Σ Q̄_k (z_k³ − z_{k−1}³),
//   Q̄_k = T(θ_k)ᵀ Q_k T(θ_k)   (Q_k the plane-stress matrix of the ply material, θ_k its angle to the material axis),
//
// and the transverse shear H_x, H_y from the shear stresses of cylindrical bending, 1/H = ∫ τ²/G dz (Laminate.cpp,
// setTransverseShear; κ G t with κ = 5/6 for one material). A blank shearModulus1Z or shearModulus2Z is an infinite
// modulus: such a ply is rigid in transverse shear along that axis and adds nothing to the integral, and the laminate
// is rigid in a direction (H = 0) only when every one of its plies is. The model parameter LaminateTransverseShear
// (LaminateShearConvention) takes H = 5/6 Σ G_k t_k instead. A ply material without transverse shear moduli
// (AnisoMat2D) follows AnisotropicPlyShear. The thermal resultants of a temperature change ΔT(z) =
// ΔT + z·T' (Jones §4.5): N_T = Σ Q̄_k ᾱ_k (ΔT t_k + T' ½ (z_k² − z_{k−1}²)), M_T = Σ Q̄_k ᾱ_k (ΔT ½ (z_k² − z_{k−1}²)
// + T' ⅓ (z_k³ − z_{k−1}³)), ᾱ_k = T(θ_k)⁻¹ α_k. Every matrix and vector is in the material frame of the element
// (θ = 0 along the material axis of the element's materialAngle or matFrameId); the strains carry the engineering
// shear.
//
// laminateOption: blank and Symmetry (the stack as above), MembraneOnly (B, D left out), BendingOnly (A, B left out),
// SmearedLaminate (the stacking ignored: D = A t²/12, B = 0, the transverse shear rigid); SandwichCore is refused.
// failureTheory: Hill, Hoffman, TsaiWu, StrainTheory; the Hashin criteria are refused.

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/shells/ShellMaterial.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements
{

// The shear correction factor of a homogeneous section, κ = 5/6: the TS/T of a laminate's shell section. The laminate's
// transverse shear stiffness H comes from its ply stack and equals κ G t for one material.
inline constexpr double LAMINATE_SHEAR_FACTOR = 5.0 / 6.0;

enum class FailureTheory : std::uint8_t
{
    None,
    Hill,
    Hoffman,
    TsaiWu,
    MaxStrain
};

enum class LaminateOption : std::uint8_t
{
    Full,     // blank or Symmetry: every stiffness term
    Membrane, // MembraneOnly
    Bending,  // BendingOnly
    Smeared   // SmearedLaminate
};

// The transverse shear of a ply whose material gives no transverse shear modulus of its own (an AnisoMat2D, whose
// matrix holds the in-plane terms only):
// - Rigid, the classic formulation: the ply has no transverse shear flexibility, as a ply of OrthoMat2D whose
//   shearModulus1Z and shearModulus2Z are blank (one rule: no modulus given, no flexibility);
// - InPlaneShearModulus, the mixed hybrid formulation: the in-plane shear modulus G33 of the matrix in both
//   directions, the closest estimate the card allows (exact for an isotropic material written as an AnisoMat2D).
enum class AnisotropicPlyShear : std::uint8_t
{
    Rigid,
    InPlaneShearModulus
};

// How the transverse shear stiffness H of a stack is formed along each material axis: the model parameter
// LaminateTransverseShear.
// - Equilibrium (the default): from the shear stresses that the equilibrium gives in cylindrical bending,
//   1/H = ∫ τ²/G dz (Whitney, "Structural Analysis of Laminated Anisotropic Plates", 1987, §4.6). The stiffness
//   depends on the stacking sequence: 5/6 G t for one material, another share of Σ G_k t_k for plies of different
//   moduli (0.78 and 0.66 along the two axes of a carbon/epoxy [0/90]s).
// - FiveSixths: the first-order shear deformation theory with the correction factor of a homogeneous section,
//   H = 5/6 Σ G_k t_k (Reddy, "Mechanics of Laminated Composite Plates and Shells", 2nd ed., 2004, §3.4, with the
//   shear correction coefficient K = 5/6). The stiffness does not depend on the stacking sequence; a ply rigid along
//   the axis (a blank modulus) makes the sum infinite and the laminate rigid along it. For a comparison with a
//   program or a formula that uses this theory.
// The shear stresses of the plies are those of the equilibrium in both (LaminatePly::shearAtTopPerForce).
enum class LaminateShearConvention : std::uint8_t
{
    Equilibrium,
    FiveSixths
};

inline constexpr std::string_view LAMINATE_SHEAR_PARAMETER = "LaminateTransverseShear";
inline constexpr std::string_view LAMINATE_SHEAR_EQUILIBRIUM = "Equilibrium";
inline constexpr std::string_view LAMINATE_SHEAR_FIVE_SIXTHS = "FiveSixths";

// One ply, from the bottom.
struct LaminatePly
{
    std::int32_t number = 0; // 1 at the bottom (the mirrored plies of Symmetry numbered on)
    std::int32_t plyId = 0;  // the global ply ID of a CompPropGlobal, 0 for a CompProp
    // The ply of the card it comes from (0 the first): number − 1, or 2n − number in the mirrored half of Symmetry
    // (n the card's plies); the item of a designed ply thickness (plyThickness).
    std::int32_t cardPly = 0;
    ShellMaterial material;
    double thickness = 0.0;
    double angle = 0.0;   // θ, radians, from the material axis of the element
    double zBottom = 0.0; // from the reference plane
    double zTop = 0.0;
    bool hasOutput = false;            // plyStressOutput = Yes
    base::Mat3 stiffness;              // Q̄ = T(θ)ᵀ Q T(θ), material frame of the element
    std::array<double, 3> expansion{}; // ᾱ = T(θ)⁻¹ α, material frame of the element
    // The transverse shear stress at the top of the ply per unit shear force (τxz/Qx, τyz/Qy), material frame of the
    // element: the equilibrium distribution (setTransverseShear), which does not depend on the shear moduli; 0 at the
    // top of the stack.
    std::array<double, 2> shearAtTopPerForce{};
};

// A share cos²θ or sin²θ of a ply axis in a direction at or below this is round-off (cos² of a right angle is 4e-33,
// not 0): the direction is across that axis.
inline constexpr double PLY_AXIS_SHARE_ROUND_OFF = 1.0e-12;

// The transverse shear modulus of a ply in direction d (0: x, 1: y) of the material frame of the element: the moduli
// (G1Z, G2Z: shearModulus1Z, shearModulus2Z) of the ply axes turned by the ply angle θ, G_xz = G1Z cos²θ + G2Z sin²θ,
// G_yz = G1Z sin²θ + G2Z cos²θ.
// 0 when the ply is rigid in that direction: a blank G1Z or G2Z is an infinite modulus (ShellMaterial gives it as 0),
// and so is its sum with the other one as soon as the direction has a share of that axis (above
// PLY_AXIS_SHARE_ROUND_OFF). The modulus of the laminate's shear integral, of its rate and of the plies' shear strains.
[[nodiscard]] double plyShearModulus(const LaminatePly& ply, std::size_t d);

// The resolved laminate of one CompProp or CompPropGlobal.
struct Laminate
{
    std::int32_t propId = 0;
    bool isCompPropGlobal = false; // the card: CompPropGlobal (global ply IDs) or CompProp
    // bottomOffset blank (−T/2, following the plies' thicknesses: LaminateRate) or given (fixed): the card's
    // bottomOffsetGiven of data contract 54, or before it an offset equal to −T/2 to the bit (as
    // design::DerivedProperties reads it).
    bool isZ0Blank = true;
    LaminateOption option = LaminateOption::Full;
    std::vector<LaminatePly> plies;          // bounded by the plies of one property
    double thickness = 0.0;                  // Σ t_k
    double massPerArea = 0.0;                // Σ ρ_k t_k
    double nonStructuralMass = 0.0;          // nonStructMass
    base::Mat3 a;                            // membrane
    base::Mat3 b;                            // membrane-bending coupling
    base::Mat3 d;                            // bending
    std::array<double, 2> transverseShear{}; // H_x, H_y; 0: rigid in that direction (every ply rigid in it)
    // How H was formed (the model's LaminateTransverseShear): its rate along a ply thickness follows it.
    LaminateShearConvention shearConvention = LaminateShearConvention::Equilibrium;
    // The plies' interlaminar shear follows the equilibrium distribution (shearAtTopPerForce), whatever the stack: one
    // material or several, flexible or rigid in transverse shear. Only SmearedLaminate, which ignores the stacking,
    // gives every ply the mean shear Q / t (ShellRecovery).
    std::array<bool, 2> hasShearDistribution{};
    // The thermal resultants per unit ΔT and per unit through-thickness gradient T'.
    std::array<double, 3> forcePerDegree{};
    std::array<double, 3> momentPerDegree{};
    std::array<double, 3> forcePerGradient{};
    std::array<double, 3> momentPerGradient{};
    double referenceTemperature = 0.0; // of the CompProp (the reference temperatures of the plies are not used)
    FailureTheory failureTheory = FailureTheory::None;
    double bondAllowable = 0.0; // the allowable interlaminar shear stress of the bonds

    [[nodiscard]] bool hasMembrane() const noexcept { return option != LaminateOption::Bending; }
    [[nodiscard]] bool hasBending() const noexcept { return option != LaminateOption::Membrane; }
};

// The columns of the laminate tables a shell batch reads, once per facet call.
struct LaminateColumns
{
    const base::ContractReader* plies = nullptr;   // laminatePliesTable
    const base::ContractReader* summary = nullptr; // laminateSummary
    ShellMaterialColumns materials;
    // By the formulation of the model: Rigid in the classic one, InPlaneShearModulus in the mixed hybrid one.
    AnisotropicPlyShear anisotropicPlyShear = AnisotropicPlyShear::Rigid;
    // The model parameter LaminateTransverseShear; Equilibrium without it.
    LaminateShearConvention shearConvention = LaminateShearConvention::Equilibrium;
    // Why the parameter cannot be read (laminateShearParameterProblem), empty when it can: every laminate is then
    // refused with it.
    std::string shearConventionProblem;

    explicit LaminateColumns(const ElementBatchView& view);
};

// The sum Σ G_k t_k of the transverse shear moduli of the plies along direction d (0: x, 1: y) of the material frame
// of the element, each times its thickness (plyShearModulus): the stiffness of the plies sheared together, which the
// FiveSixths convention multiplies by 5/6 and the shear factor of a stack is measured against. 0 when a ply is rigid
// along d (its modulus is infinite, and the sum with it).
[[nodiscard]] double plyShearSum(const Laminate& laminate, std::size_t d);

// What is wrong with the model parameter LaminateTransverseShear, or nothing: a value that is not Equilibrium or
// FiveSixths. The message names the parameter, what was expected and found, and what to do.
[[nodiscard]] std::optional<std::string> laminateShearParameterProblem(const input::ModelSnapshot& snapshot);

// The laminate of property `propId` (a CompProp or CompPropGlobal, `property` its table and `propertyRow` its row) for
// the shell of batch row `row`. Problems: no ply stack for the property (or a snapshot taken without
// laminatePliesTable), a ply of thickness not positive, a ply material the shells cannot take, the SandwichCore
// option, a Hashin failure theory.
[[nodiscard]] Resolution<Laminate> resolveLaminate(const ElementBatchView& view, const LaminateColumns& columns,
                                                   std::int64_t row, const input::CardTable& property,
                                                   std::int64_t propertyRow);

} // namespace oos::elements
