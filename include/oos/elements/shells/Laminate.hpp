#pragma once

// A laminate of the shells (PCOMP, CompPropGlobal):
//
//   | CompProp | PID  | Z0 | NSM    | SB    | FT   | TREF | GE     | LAM   |
//   |       | MID1 | T1 | THETA1 | SOUT1 | MID2 | T2   | THETA2 | SOUT2 | ...
//
// The stack comes from laminatePliesTable (the plies from the bottom with their z from the reference plane, Z0
// applied, the mirrored half of LAM = SYM included), the option LAM from laminateSummary, the other fields from the
// CompProp or CompPropGlobal table. The stiffness is the classical lamination theory (Jones, Mechanics of Composite
// Materials, §4.2), sums in increasing ply order:
//
//   A = Σ Q̄_k t_k,  B = ½ Σ Q̄_k (z_k² − z_{k−1}²),  D = ⅓ Σ Q̄_k (z_k³ − z_{k−1}³),
//   Q̄_k = T(θ_k)ᵀ Q_k T(θ_k)   (Q_k the plane-stress matrix of the ply material, θ_k its angle to the material axis),
//
// and the transverse shear: H_x = κ Σ G_k,1z t_k, H_y = κ Σ G_k,2z t_k, κ = 5/6, a ply with a blank
// G1Z (G2Z) making the laminate rigid in that direction. The thermal resultants of a temperature change ΔT(z) =
// ΔT + z·T' (Jones §4.5): N_T = Σ Q̄_k ᾱ_k (ΔT t_k + T' ½ (z_k² − z_{k−1}²)), M_T = Σ Q̄_k ᾱ_k (ΔT ½ (z_k² − z_{k−1}²)
// + T' ⅓ (z_k³ − z_{k−1}³)), ᾱ_k = T(θ_k)⁻¹ α_k. Every matrix and vector is in the material frame of the element
// (θ = 0 along the material axis of THETA or MCID); the strains carry the engineering shear.
//
// LAM (PCOMP): blank and SYM (the stack as above), MEM (membrane only: B, D left out), BEND (bending
// only: A, B left out), SMEAR (the stacking ignored: D = A t²/12, B = 0, the transverse shear rigid); SMCORE is
// refused. Failure theories (FT): HILL, HOFF, TSAI, STRN; the Hashin criteria are refused.

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/shells/ShellMaterial.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
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
    Full,     // blank or SYM: every stiffness term
    Membrane, // MEM
    Bending,  // BEND
    Smeared   // SMEAR
};

// One ply, from the bottom.
struct LaminatePly
{
    std::int32_t number = 0; // 1 at the bottom (the mirrored plies of SYM numbered on)
    std::int32_t plyId = 0;  // the global ply ID of a CompPropGlobal, 0 for a CompProp
    // The ply of the card it comes from (0 the first): number − 1, or 2n − number in the mirrored half of LAM = SYM
    // (n the card's plies); the item of a designed ply thickness ("ply_thickness").
    std::int32_t cardPly = 0;
    ShellMaterial material;
    double thickness = 0.0;
    double angle = 0.0;   // θ, radians, from the material axis of the element
    double zBottom = 0.0; // from the reference plane
    double zTop = 0.0;
    bool hasOutput = false;            // SOUT = YES
    base::Mat3 stiffness;              // Q̄ = T(θ)ᵀ Q T(θ), material frame of the element
    std::array<double, 3> expansion{}; // ᾱ = T(θ)⁻¹ α, material frame of the element
    // The transverse shear stress at the top of the ply per unit shear force (τxz/Qx, τyz/Qy), material frame of the
    // element: the equilibrium distribution (setTransverseShear); 0 in a direction rigid in transverse shear.
    std::array<double, 2> shearAtTopPerForce{};
};

// The resolved laminate of one CompProp or CompPropGlobal.
struct Laminate
{
    std::int32_t propId = 0;
    bool isCompPropGlobal = false; // the card: CompPropGlobal (global ply IDs) or CompProp
    // Z0 blank (−T/2, following the plies' thicknesses: LaminateRate) or given (fixed): the card's z0_given of data
    // contract 54, or before it a Z0 equal to −T/2 to the bit (as design::DerivedProperties reads it).
    bool isZ0Blank = true;
    LaminateOption option = LaminateOption::Full;
    std::vector<LaminatePly> plies;          // bounded by the plies of one property
    double thickness = 0.0;                  // Σ t_k
    double massPerArea = 0.0;                // Σ ρ_k t_k
    double nonStructuralMass = 0.0;          // NSM
    base::Mat3 a;                            // membrane
    base::Mat3 b;                            // membrane-bending coupling
    base::Mat3 d;                            // bending
    std::array<double, 2> transverseShear{}; // H_x, H_y; 0: rigid in that direction
    // The plies' interlaminar shear follows the equilibrium distribution (shearAtTopPerForce) in a flexible direction
    // of a stack of several materials or angles; otherwise every ply carries the mean shear Q / t (ShellRecovery).
    std::array<bool, 2> hasShearDistribution{};
    // The thermal resultants per unit ΔT and per unit through-thickness gradient T'.
    std::array<double, 3> forcePerDegree{};
    std::array<double, 3> momentPerDegree{};
    std::array<double, 3> forcePerGradient{};
    std::array<double, 3> momentPerGradient{};
    double referenceTemperature = 0.0; // TREF of the CompProp (the plies' own TREF are not used)
    FailureTheory failureTheory = FailureTheory::None;
    double bondAllowable = 0.0; // SB

    [[nodiscard]] bool hasMembrane() const noexcept { return option != LaminateOption::Bending; }
    [[nodiscard]] bool hasBending() const noexcept { return option != LaminateOption::Membrane; }
};

// The columns of the laminate tables a shell batch reads, once per facet call.
struct LaminateColumns
{
    const base::ContractReader* plies = nullptr;   // laminatePliesTable
    const base::ContractReader* summary = nullptr; // laminateSummary
    ShellMaterialColumns materials;

    explicit LaminateColumns(const ElementBatchView& view);
};

// The laminate of property `propId` (a CompProp or CompPropGlobal, `property` its table and `propertyRow` its row) for
// the shell of batch row `row`. Problems: no ply stack for the PID (or a snapshot taken without laminatePliesTable), a
// ply of thickness not positive, a ply material the shells cannot take, LAM = SMCORE, a Hashin failure theory.
[[nodiscard]] Resolution<Laminate> resolveLaminate(const ElementBatchView& view, const LaminateColumns& columns,
                                                   std::int64_t row, const input::CardTable& property,
                                                   std::int64_t propertyRow);

} // namespace oos::elements
