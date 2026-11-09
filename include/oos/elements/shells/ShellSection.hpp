#pragma once

// The section of a shell, shared by every shell element (PSHELL, CompProp, CompPropGlobal):
//
//   | ShellProp | PID | MID1 | T | MID2 | 12I/T**3 | MID3 | TS/T | NSM |
//   |        | Z1  | Z2   | MID4 |
//
// MID1 gives the membrane, MID2 the bending, MID3 the transverse shear flexibility, MID4 the membrane-bending coupling;
// a blank MID (0) leaves its part out. Each MIDi is a IsoMat, AnisoMat2D or OrthoMat2D (ShellMaterial). MID2 = −1 is
// the plane-strain option of ShellProp: the membrane of a IsoMat MID1 in plane strain, no bending.
//
// A CompProp or CompPropGlobal (Laminate) is written as its equivalent ShellProp: T = the laminate thickness, G1 = A/T,
// 12I/T³ = 1 with G2 = 12 D/T³, G4 = B/T², and the transverse shear of the laminate (κ G per unit thickness = H/T).
// The corner thicknesses T1-T4 of the element do not apply to a laminate.
//
// The coupling is kept in the sign of the lamination theory (B = ½ Σ Q̄ (z_k² − z_{k−1}²), z along the element normal),
// which SHELL_COUPLING_SIGN carries to the shell curvature of ShellRecovery.hpp. The input format gives a MID4 in the
// sign of the moments of the result tables (M = −∫σ·z dz), so a ShellProp MID4 enters as −G4 (ShellSection.cpp gives
// the evidence): a G4 > 0 is a plate stiffer on the −n side.
//
// The matrices are kept in the material frame (the x axis of THETA or MCID); rotatedToElementFrame
// writes them in the element frame by M_e = T(φ)ᵀ M T(φ), the strain-like vectors by v_e = T(φ)⁻¹ v.

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/DesignFieldUse.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/shells/Laminate.hpp"
#include "oos/elements/shells/ShellMaterial.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace oos::elements
{

// ShellProp defaults, applied when O_o_Reader gives a field null.
inline constexpr double SHELL_PROP_DEFAULT_BENDING_RATIO = 1.0;    // 12I/T³
inline constexpr double SHELL_PROP_DEFAULT_SHEAR_RATIO = 0.833333; // TS/T, 5/6 rounded to six digits
// The ratio γ / G of the rotation penalty of the drilling membranes of ShellQuad4Drilling and ShellTria3Drilling
// (Ibrahimbegović, Taylor, Wilson 1990: γ = G gives accurate results with no locking; Hughes, Brezzi 1989 for the
// variational basis).
inline constexpr double DRILLING_MEMBRANE_PENALTY_RATIO = 1.0;

// The thermal stress resultants of a shell (CLT signs: N = ∫σ dz, M = ∫σ z dz of the stresses α (T − T0) would carry
// if the strains were held at zero), in the frame of the section.
struct ThermalResultants
{
    std::array<double, 3> force{};
    std::array<double, 3> moment{};
};

// The section of one shell property with its materials resolved. The matrices are per unit thickness: the membrane
// stiffness of a thickness t is t·membraneMatrix, the bending stiffness D = bendingRatio·t³/12·bendingMatrix, the
// coupling [B] = t²·couplingMatrix (each follows the thickness at each integration point).
struct ShellSection
{
    std::int32_t propId = 0;
    // MID1 to MID4 of a ShellProp as given (0 blank, MID2 −1 the plane-strain option); zero for a laminate. The
    // closed-form rates of a material field (ShellSectionRate) read which parts a material acts on.
    std::array<std::int32_t, 4> materialIds{};
    bool hasMembrane = false;          // MID1 given (a laminate but LAM = BEND)
    bool hasBending = false;           // MID2 given (a laminate but LAM = MEM)
    bool hasCoupling = false;          // MID4 given, or a laminate with B ≠ 0
    bool hasShearFlexibility = false;  // a transverse shear direction flexible (MID3 given with its modulus > 0)
    base::Mat3 membraneMatrix;         // G1
    base::Mat3 bendingMatrix;          // G2
    base::Mat3 couplingMatrix;         // B/t² in the lamination theory's sign: −G4 of a MID4, B/T² of a laminate
    double membraneShearModulus = 0.0; // γ = G of the drilling membranes' rotation penalty: of MID1, A33/T (laminate)
    // κ·G per unit thickness of the transverse shears along the two material axes (κ = TS/T, G = G3, G1Z or G2Z;
    // H/T for a laminate along its axes); 0 for a rigid direction. The material axes are at frameAngle from
    // the x axis of the section's frame: the elements take the transverse shear through transverseShearCompliance and
    // transverseShearStiffnessAlongAxes, which rotate it into their frame.
    std::array<double, 2> transverseShearStiffness{};
    double bendingRatio = SHELL_PROP_DEFAULT_BENDING_RATIO; // 12I/T³
    double shearRatio = SHELL_PROP_DEFAULT_SHEAR_RATIO;     // TS/T (the shear correction factor κ)
    double density = 0.0;           // RHO of MID1, of MID2 when MID1 is blank (PSHELL); Σρt/T for a laminate
    double nonStructuralMass = 0.0; // NSM, mass per unit area
    bool hasThickness = false;      // T given (always for a laminate)
    double thickness = 0.0;         // T
    // Z1 and Z2 given (PSHELL: a blank one is −t/2 or +t/2 of the thickness at each point).
    bool hasFiberDistance1 = false;
    bool hasFiberDistance2 = false;
    double fiberDistance1 = 0.0;
    double fiberDistance2 = 0.0;
    // The thermal expansion (α1, α2, α12) of MID1 and MID2 and the reference temperature (TREF of MID1, of MID2 when
    // MID1 is blank; of the CompProp for a laminate).
    std::array<double, 3> membraneExpansion{};
    std::array<double, 3> bendingExpansion{};
    double referenceTemperature = 0.0;
    // The laminate of a CompProp or CompPropGlobal (in the material frame whatever the frame of the section), null for
    // a ShellProp.
    std::shared_ptr<const Laminate> laminate;
    double frameAngle =
        0.0; // the angle from the x axis of the section's frame to the material axis (0: material frame)

    [[nodiscard]] bool isLaminate() const noexcept { return laminate != nullptr; }
    // D = (12I/T³)·t³/12·G2 at a thickness t.
    [[nodiscard]] base::Mat3 bendingStiffness(double t) const
    {
        return bendingRatio * t * t * t / 12.0 * bendingMatrix;
    }
    // [B] = t²·couplingMatrix at a thickness t, in the lamination theory's sign (−t²·G4 of a MID4).
    [[nodiscard]] base::Mat3 couplingStiffness(double t) const { return t * t * couplingMatrix; }
    // The fibre distances (Z1, Z2) at a point of thickness t.
    [[nodiscard]] std::array<double, 2> fibres(double t) const
    {
        return {hasFiberDistance1 ? fiberDistance1 : -t / 2.0, hasFiberDistance2 ? fiberDistance2 : t / 2.0};
    }
    // The section written in the frame whose x axis is at −materialAngle from the material axis (the element frame of
    // an element whose material axis is at materialAngle from its x axis).
    [[nodiscard]] ShellSection rotatedToElementFrame(double materialAngle) const;
    // The transverse shear compliance 1/(κ G) per unit thickness in the section's frame, as (xx, yy, xy): the shear
    // strain γ = (γxz, γyz) and force per thickness τ transform as vectors, so C = R diag(1/s₁, 1/s₂) Rᵀ with s the
    // two values of transverseShearStiffness and R the rotation by frameAngle; a rigid direction counts 0 (the
    // material values are along the material axes, never along the element axes whatever THETA or MCID).
    [[nodiscard]] std::array<double, 3> transverseShearCompliance() const;
    // κ G per unit thickness along the x and y axes of the section's frame: the inverse of the diagonal of
    // transverseShearCompliance (0 for a direction with no compliance, a rigid one). For a formulation that treats the
    // two shears apart (the ShellQuad4 edge shear), which then leaves out the coupling term C_xy of a
    // material axis that is neither along x nor along y.
    [[nodiscard]] std::array<double, 2> transverseShearStiffnessAlongAxes() const;
    // The thermal resultants of a temperature change ΔT at the reference plane and a gradient T' through the
    // thickness, at a thickness t, in the frame of the section: N = t G1 α1 ΔT, M = (12I/T³) t³/12 G2 α2 T' for a
    // ShellProp (the bending of a MID2 strains only under a gradient, PlateTemperatureLoad field TPRIME), the
    // laminate's for a CompProp.
    [[nodiscard]] ThermalResultants thermalResultants(double deltaT, double gradient, double t) const;
};

// The property and material columns of a shell batch, read once per facet call.
//
// Thread-safety: used by one facet call, on one thread (it keeps the last laminate it resolved).
struct ShellSectionColumns
{
    PropertyRows shellProps;
    PropertyRows compProps;
    PropertyRows compPropGlobals;
    ShellMaterialColumns materials;
    LaminateColumns laminates;
    std::span<const std::int32_t> membraneMatId, bendingMatId, shearMatId, couplingMatId;
    std::array<base::ValidityView, 4> isMatIdGiven;
    std::span<const double> t, bendingRatio, shearRatio, nonStructMass, fiberDistance1, fiberDistance2;
    base::ValidityView isTGiven, isBendingRatioGiven, isShearRatioGiven, isNonStructMassGiven, isFiberDistance1Given,
        isFiberDistance2Given;
    // The section of the last laminate resolveShellSection resolved (its property table and row): the next element of
    // the same CompProp takes it again.
    struct LastLaminate
    {
        const input::CardTable* table = nullptr;
        std::int64_t row = -1;
        ShellSection section;
    };
    mutable std::optional<LastLaminate> lastLaminate;

    explicit ShellSectionColumns(const ElementBatchView& view);
};

// The section of the shell of batch row `row`. Problems: a PID that is not a ShellProp, CompProp or CompPropGlobal, an
// unknown material or one the shells cannot take, no MID1 and no MID2, MID3 or MID4 without the parts they act on, MID2
// = −1 with a MID1 that is not a IsoMat, the laminate problems of resolveLaminate.
[[nodiscard]] Resolution<ShellSection> resolveShellSection(const ElementBatchView& view,
                                                           const ShellSectionColumns& columns, std::int64_t row);

// The thickness at each corner of a shell (CQUAD4, ShellTria3 fields T1-T4 and TFLAG): a corner thickness as given
// (TFLAG 0) or as a fraction of T (TFLAG 1); a blank one (O_o_Reader: −1) takes T (TFLAG 0) or 1·T (TFLAG 1). A
// laminate takes its own thickness at every corner. A blank corner without a T, a TFLAG other than 0 or 1, or a
// thickness that is not positive is a problem.
template <std::size_t N>
[[nodiscard]] Resolution<std::array<double, N>> resolveCornerThicknesses(const ElementBatchView& view, std::int64_t row,
                                                                         const ShellSection& section,
                                                                         std::int32_t thicknessFlag,
                                                                         std::span<const double> cornerThickness);

// Whether a designed field is the thickness T of a ShellProp, the field whose shell matrices follow in closed form.
[[nodiscard]] bool isShellThickness(const DesignedField& field) noexcept;

// Whether every corner thickness of a shell is proportional to the T of its ShellProp, t_i = r_i·T (each corner blank,
// or TFLAG = 1: resolveCornerThicknesses), so that its matrices follow T in closed form (ShellQuad4DrillingPlate::thicknessRate,
// ShellTria3DrillingPlate::thicknessRate); never for a laminate, nor without a positive T.
[[nodiscard]] bool cornersFollowThickness(const ShellSection& section, std::int32_t thicknessFlag,
                                          std::span<const double> cornerThickness) noexcept;

// The problem of a shell offset by ZOFFS from its grid points, or nothing. A shell (CQUAD4, ShellTria3, ShellQuad8,
// ShellTria6, ShellQuad4Drilling, ShellTria3Drilling) with a ZOFFS needs both MID1 and MID2 on its property. The offset ties
// the rotations of the grid points to the membrane (u_plane = u + θ × ZOFFS·n): without the bending of MID2 the
// rotations act only through the membrane, in the combination u + ZOFFS·θ, so the assembled stiffness is singular in
// every free rotation (and the grid point singularity check, which looks at the rotation block of each grid point
// alone, does not see it); without the membrane of MID1 the offset has nothing to act on. Such an element is refused,
// naming ZOFFS.
[[nodiscard]] std::optional<std::string> offsetSectionProblem(const ShellSection& section, double offset);

// The problem of a shell offset by ZOFFS under the model's ModelParams OFFDEF, or nothing: LROFF takes a shell's offset
// direction from the shell normals at its grid points (MDLPRM OFFDEF LROFF), which O_o_Solver does not
// compute (its offsets are along the element normal, as ELMOFF and ELMZ); refused, naming ELMZ (LROFF with the element
// z direction: input::OffsetDefinition).
[[nodiscard]] std::optional<std::string> offsetDefinitionProblem(const input::ModelSnapshot& snapshot, double offset);

} // namespace oos::elements
