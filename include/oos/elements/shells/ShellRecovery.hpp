#pragma once

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/RecoverySlopes.hpp"
#include "oos/elements/shells/LaminateRate.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::elements
{

// The output rows of the shells (CQUAD4, ShellTria3, ShellQuad8, ShellTria6, ShellQuad4Drilling, ShellTria3Drilling), shared:
// from the generalised strains and forces at one place of an element (its centre or a corner) to the rows of the
// O_o_Reader result tables (stress.shellQuad4, strain.shellQuad4, forces.shellQuad4, stress.shellQuad4_composite ...), in
// the element frame (the plies in their own frame).
//
// Sign conventions: the fibre at distance z from the reference plane strains as ε(z) = ε_m − z·κ, κ the shell
// curvature (κx = −∂θy/∂x, κy = ∂θx/∂y, 2κxy = ∂θx/∂x − ∂θy/∂y, which follows from u(z) = u + z·θy,
// v(z) = v − z·θx); the moments are the resultants M = ∫σ·z dz, so that the fibre stress is σ(z) = N/t + z·M/I with
// I = (12I/T³)·t³/12 (the plate stress at a fibre distance) — ShellPlaceState holds these resultants, the force table
// writes −M by the convention of the result tables; shear strains are engineering strains (γxy). The curvature of
// the classical lamination theory (Jones §4.2, ε(z) = ε_m + z κ') is κ' = −κ.

// The sign of the membrane-bending coupling of a shell in the curvature κ above: the energy density of the section is
// ½ εᵀ A ε + s εᵀ [B] κ + ½ κᵀ D κ, [B] in the sign of the lamination theory (ShellSection: B = ½ Σ Q̄ (z_k² −
// z_{k−1}²) of a laminate, and −t² G4 of a ShellProp MID4, whose input sign follows the moments of the result tables),
// whose curvature is κ' = −κ: s = −1. The enriched plates write the coupling block as Σ ε_enrᵀ [B] κ_h with the
// input's [B] (+t² G4); s·[B] is that. Checked against the rigid offset ZOFFS of a homogeneous plate, which the
// coupling of the offset section must reproduce; the opposite sign puts an offset laminate on the other side of the
// grid plane.
inline constexpr double SHELL_COUPLING_SIGN = -1.0;

// The temperature change of a shell element from its initial state: ΔT at the reference plane and the gradient T'
// through the thickness (TEMPP1 field TPRIME). ΔT is uniform (a PlateTemperatureLoad, or a deck without grid point
// temperatures) or given at the corners from the grid point temperatures (TEMP, DefaultTemperatureLoad), interpolated
// over the element by its shape functions; `deltaT` is then the value at the centre.
struct ShellTemperatureChange
{
    double deltaT = 0.0;
    double gradient = 0.0;
    bool hasCornerValues = false;
    std::array<double, 8> atCorners{}; // ΔT at the grid points G1, G2 ... (hasCornerValues)

    [[nodiscard]] bool isZero() const noexcept
    {
        if (gradient != 0.0 || deltaT != 0.0)
            return false;
        for (const double value : atCorners)
            if (value != 0.0)
                return false;
        return true;
    }
};

// The state of a shell at one place, in the element frame.
struct ShellPlaceState
{
    std::int32_t nodeId = 0;                // the grid point of a corner, 0 for the centre
    double thickness = 0.0;                 // t at the place
    std::array<double, 3> membraneStrain{}; // εx, εy, γxy of the reference plane (total: the thermal strain included)
    std::array<double, 3> curvature{};      // κx, κy, 2κxy (the shell curvature; total)
    std::array<double, 3> membraneForce{};  // Nx, Ny, Nxy per unit length (the thermal resultant subtracted)
    std::array<double, 3> moment{};         // Mx, My, Mxy per unit length (∫σ·z dz; the thermal resultant subtracted)
    std::array<double, 2> shearForce{};     // Qx, Qy per unit length
    ShellTemperatureChange temperature;     // of the element, for the ply stresses
};

// How a laminated shell writes its stresses and strains (PARAM NOCOMPS, default +1): the plies only (+1), the
// plies and the equivalent homogeneous element (0), the homogeneous element only (−1).
enum class CompositeOutput : std::uint8_t
{
    PliesOnly,
    PliesAndHomogeneous,
    HomogeneousOnly
};

// PARAM NOCOMPS of the snapshot (default +1), read once per facet call.
[[nodiscard]] CompositeOutput compositeOutputOf(const input::ModelSnapshot& snapshot);

// The columns of each output kind in the data contract: stresses fiber_distance, oxx, oyy, txy, angle, omax, omin,
// von_mises; strains the same with exx... (Strains: at the fibres; StrainCurvatures: first column fiber_curvature);
// forces nx, ny, nxy, mx, my, mxy, qx, qy.
[[nodiscard]] std::vector<std::string> shellOutputColumns(ElementOutput output);

// The form of each column of shellOutputColumns(output) in the displacements (ColumnForm): the forces, the
// fibre distance or curvature flag and the three components of a stress or strain row Linear (the fibre value does not
// depend on u), the angle and the principal values Other, the von Mises a PlaneVonMises of the components (k = ½ and
// scale = ⅔ for the strains: shellStrainRow).
[[nodiscard]] std::vector<ColumnForm> shellOutputColumnForms(ElementOutput output);

// Appends the rows of one place: Forces one row (layer 1); Stresses and Strains one row per fibre Z1, Z2 (layers 1, 2);
// StrainCurvatures the membrane strains (layer 1, fiber_curvature 0) and the curvatures (layer 2, fiber_curvature -1,
// by the convention of the result tables), the curvature κ above (ε(z) = ε_m − z·κ); Forces the moments with the sign
// of the result tables, −∫σ·z dz = D·κ, the membrane and shear forces as they are. A laminate writes its stresses and
// strains here only when `composite` asks for the homogeneous element; its forces always.
// `values`: ShellRowValues::LinearOnly leaves the columns that are not Linear at zero (the angle, the principal values
// and the von Mises of a stress or strain row: the rows of the recovery slopes, whose other columns get no slopes).
enum class ShellRowValues : std::uint8_t
{
    All,
    LinearOnly
};
void appendShellRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId, const ShellPlaceState& state,
                     const ShellSection& section, ElementOutput output,
                     CompositeOutput composite = CompositeOutput::PliesOnly,
                     ShellRowValues values = ShellRowValues::All);

// The vectorised recovery of the ShellQuad4, ShellQuad4Drilling, ShellTria3 and ShellTria3Drilling
// (ElementFamily::vectorisedRecoveryMinCases): the fewest cases of a block from which their rows are made from their
// recovery slopes. Calibrated on plate models with every element's outputs, six threads: at 1 003 686 DOFs the per-case
// recovery takes 1.07 s for one case and 0.74 s per further case, the vectorised one 2.85 s and 0.14 s (the slopes and
// the rows at rest made once); at 251 226 DOFs 0.28 s + 0.27 s per case against 0.69 s + 0.04 s. They break even at 4
// and 3 cases; 8 leaves a margin for a block that asks for one kind only (the slopes then shared by fewer rows: a
// ShellQuad4's stresses alone cost about 7 of its recoveries, 34 µs against 5 µs per element on one thread).
inline constexpr std::size_t SHELL_VECTORISED_RECOVERY_MIN_CASES = 8;

// The columns of the rows of appendShellRows that are not Linear (shellOutputColumnForms), set from their Linear ones
// in every row of every block (ElementFamily::completeRecoveredRows): a stress row by shellStressRow from its fibre
// distance and (σx, σy, τxy), a strain row (at a fibre, or the membrane strains and the curvatures of STRCUR) by
// shellStrainRow from its first column and (εx, εy, γxy); the forces are all Linear. The same functions as
// appendShellRows, so the bits of its rows for the same Linear values.
void completeShellRows(ElementOutput output, std::span<ElementResultBlock> blocks);

// The columns of the ply tables (stress.shellQuad4_composite: o11, o22, t12, t1z, t2z, angle, major, minor, von_mises;
// strain: e11 ...; failure_indices: failure_index, bond_failure_index, element_failure_index), and of the ply states
// (LayerOutput::PlyStates, no table: the stresses σ1, σ2, τ12, the mechanical strains ε − α·ΔT and the transverse
// shears τ1z, τ2z of the plies whose failure indices are written, PLY_STATE_COLUMNS values).
inline constexpr std::size_t PLY_STATE_COLUMNS = 8;
[[nodiscard]] std::vector<std::string> laminateOutputColumns(LayerOutput output);

// The rates with respect to the ShellProp thickness T of the resultants of a place state at fixed displacements, for a
// shell whose thickness at every point is proportional to T (ShellQuad4DrillingPlate::thicknessRate): the membrane
// strains and the curvatures do not move with T (the hybrid curvatures are ratios of matrices of one degree in T),
// N = t·G1·ε + s·[B]·κ − N_T holds terms of degrees 1 and 2, M = −M_h − s·[B]·ε of degrees 3 and 2, so that
//   dN/dT = (N + s·[B]·κ)/T,  dM/dT = (3·M + s·[B]·ε)/T,
// [B] at the place's thickness, present where the element couples them (`isCoupled`); the rates of the transverse
// shear forces are the element's (its saturated edge shear: `shearForceRates`).
[[nodiscard]] ShellPlaceState shellResultantRates(const ShellPlaceState& state, const ShellSection& section,
                                                  bool isCoupled, const std::array<double, 2>& shearForceRates,
                                                  double thickness);

// Appends the rates of the rows of one place (appendShellRows) with respect to a designed field at fixed displacements,
// from its state and the rates of its resultants: those of T (shellResultantRates, rates.thickness = t/T: the fibre
// distance ∓t/2 of a blank Z1, Z2 follows T, a given one does not; I = (12I/T³)·t³/12 ∝ T³), or those of a section
// term (ShellQuad4DrillingPlate::placeStateSectionRates, ShellTria3DrillingPlate::centreStateSectionRates: rates.thickness 0, the
// membrane strains and the curvatures with rates of their own); σ(z) = N/t + z·M/I, ε(z) = ε_m − z·κ; the von Mises
// chained from its components, the angle and the principal values NaN.
void appendShellRowRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                         const ShellPlaceState& state, const ShellPlaceState& rates, const ShellSection& section,
                         ElementOutput output, CompositeOutput composite = CompositeOutput::PliesOnly);

// The second rates with respect to T at fixed displacements of the resultants of a place state, under the condition
// of shellResultantRates: with N = N₁ + N_c, N₁ ∝ T, N_c = s·[B]·κ
// ∝ T², and M = M₃ + M_c, M₃ ∝ T³, M_c = −s·[B]·ε ∝ T² (the thermal resultants apart: no closed form under a
// temperature change),
//   d²N/dT² = 2·s·[B]·κ/T²,  d²M/dT² = (6·M + 4·s·[B]·ε)/T²,
// the strains, the curvatures and the place's thickness (t ∝ T) having none; the transverse shear forces' are the
// element's (`shearForceSecondRates`).
[[nodiscard]] ShellPlaceState shellResultantSecondRates(const ShellPlaceState& state, const ShellSection& section,
                                                        bool isCoupled,
                                                        const std::array<double, 2>& shearForceSecondRates,
                                                        double thickness);

// Appends the second rates of the rows of one place with respect to T at fixed displacements, from its
// state, the rates of its resultants and their second rates: σ(z) = N/t + z·M/I differentiated twice with t and z of
// a blank fibre linear in T (t'' = z'' = 0) and I ∝ T³ (I'' = 6·I·t'²/t²); the strains ε(z) = ε_m − z·κ have none
// (z'' = 0), nor the curvature rows; the von Mises, the angle and the principal values NaN (not Linear: the caller
// chains the von Mises from its components).
void appendShellRowSecondRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                               const ShellPlaceState& state, const ShellPlaceState& rates,
                               const ShellPlaceState& secondRates, const ShellSection& section, ElementOutput output,
                               CompositeOutput composite = CompositeOutput::PliesOnly);

// The form of each column of laminateOutputColumns(output) (ColumnForm): the in-plane and transverse shear
// components of a ply's stresses or strains Linear, its angle and principal values Other, its von Mises a PlaneVonMises
// of the in-plane components; the failure indices Other.
[[nodiscard]] std::vector<ColumnForm> laminateOutputColumnForms(LayerOutput output);

// Appends the ply rows of a laminated shell at its centre (one row per ply with SOUT = YES, layer = the ply number;
// nothing for a ShellProp, nor when `composite` asks for the homogeneous element only):
// - the strains of the ply at its mid-thickness z_k, ε(z_k) = ε_m − z_k κ, in the ply frame (θ_k from the material
// axis,
//   itself at section.frameAngle from the element x axis), and the ply stresses σ = Q_k (ε − α_k ΔT(z_k)) (the plies
//   are evaluated at their mid-thickness);
// - the transverse shear of the plies from the element's shear forces (see laminatePlyShear);
// - the failure indices of the plies by the theory FT (PCOMP field FT), the bond indices max(|τ1z|, |τ2z|)/SB at the
//   interface above each ply (NaN above the last), and the element index, the largest of them, on the last ply (NaN on
//   the others), by the convention of the failure index table. No rows
//   without FT, or when a ply's material lacks its allowables.
// LayerOutput::PlyStates: one row per ply with an output, as the failure indices (none when the laminate writes no
// failure index), the stresses, mechanical strains and transverse shears of the ply (PLY_STATE_COLUMNS).
void appendLaminateRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                        const ShellPlaceState& centre, const ShellSection& section, LayerOutput output,
                        CompositeOutput composite = CompositeOutput::PliesOnly);

// The rates of a place state along a ply thickness from its two parts: `section` the rates of the section at
// fixed T (the plates' placeStateSectionRates, no thickness rate) and `thickness` those of T (placeStateRates,
// shellResultantRates) scaled by dT; the sum of each component.
[[nodiscard]] ShellPlaceState combinedStateRates(const ShellPlaceState& section, const ShellPlaceState& thickness,
                                                 double thicknessRate);

// Appends the rates of the ply rows appendLaminateRows gives (the same rows, places and columns) along a ply thickness
// at fixed displacements, from the centre's state and its rates (combinedStateRates) and the laminate's rate
// (LaminateRate: the stack's z and the plies' shear at their top per force): with z_k the ply's mid-thickness,
//   dε_k = T(θ)(dε − z_k dκ − dz_k κ),  dσ_k = Q dε_k,
//   dτ_k = dQ s_k + Q ds_k (a shear distribution), d(Q/t) = dQ/T − Q dT/T² (the mean shear),
// the ply's von Mises chained from its components, its angle and principal values NaN; the failure index along the
// gradient of its theory (plyFailureIndexGradient), the bond index along its larger shear's, the element index (the
// largest) NaN. No temperature: the caller refuses the closed form under a temperature change.
void appendLaminateRowRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                            const ShellPlaceState& centre, const ShellPlaceState& centreRates,
                            const ShellSection& section, const LaminateRate& rate, LayerOutput output,
                            CompositeOutput composite = CompositeOutput::PliesOnly);

// The vectorised recovery of the ply tables: the laminated shells' ply rows of many cases at once from their slopes
// (ElementFamily::completeRecoveredLayerRows).
//
// completeLaminateRows sets the angle, the principal values and the von Mises value of every ply stress or strain row
// of the blocks from its components, with appendLaminateRows' functions (the same bits for the same components).
// appendLaminateFailureRows appends the failure index rows of one element from its ply state rows
// plyStateRows[first, last) (appendLaminateRows with LayerOutput::PlyStates), as appendLaminateRows makes them from the
// same states. Both throw std::invalid_argument on an output or rows they do not take.
void completeLaminateRows(LayerOutput output, std::span<ElementResultBlock> blocks);
void appendLaminateFailureRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                               const ShellSection& section, const ElementResultBlock& plyStateRows, std::size_t first,
                               std::size_t last);

// The failure index rows of every element of `blocks` from its ply state rows in `sources` (the same case in each;
// ElementFamily::completeRecoveredLayerRows): the rows of an element are consecutive and the same in every source, and
// `withSection(row, visit)` calls visit(section) with the section of the element of batch row `row`, resolved once for
// every case.
template <typename WithSection>
void completeLaminateFailureRows(std::span<const ElementResultBlock> sources, std::span<ElementResultBlock> blocks,
                                 WithSection&& withSection)
{
    if (sources.empty())
        return;
    const ElementResultBlock& structure = sources.front();
    std::size_t first = 0;
    while (first < structure.size())
    {
        std::size_t last = first + 1;
        while (last < structure.size() && structure.elementRow(last) == structure.elementRow(first))
            ++last;
        const std::int64_t row = structure.elementRow(first);
        const std::int32_t elementId = structure.elementId(first);
        withSection(row,
                    [&](const ShellSection& section)
                    {
                        for (std::size_t c = 0; c < sources.size(); ++c)
                            appendLaminateFailureRows(blocks[c], row, elementId, section, sources[c], first, last);
                    });
        first = last;
    }
}

// The forms and the columns of an element or layer output of the shells (shellOutputColumnForms and
// shellOutputColumns, laminateOutputColumnForms and laminateOutputColumns).
[[nodiscard]] std::vector<ColumnForm> shellRecoveryOutputForms(const RecoveryOutput& output);
[[nodiscard]] std::vector<std::string> shellRecoveryOutputColumns(const RecoveryOutput& output);
// Appends the rows of `output` at the place states of one element (its centre first): an element output at every
// place (appendShellRows; with ShellRowValues::LinearOnly, the default, the columns that are not Linear left at zero:
// the rows whose slopes the shells' recoverySlopesOfOutputs takes), a layer output at the centre (appendLaminateRows).
void appendShellOutputRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                           std::span<const ShellPlaceState> states, const ShellSection& section,
                           const RecoveryOutput& output, CompositeOutput composite,
                           ShellRowValues values = ShellRowValues::LinearOnly);
// Resets blocks[k] to the columns of outputs[k] (shellRecoveryOutputColumns) for the shells' recoverOutputs;
// returns whether an element output is among them (a layer output has rows for the laminates only).
bool resetShellRecoveryBlocks(std::span<const RecoveryOutput> outputs, std::span<ElementResultBlock> blocks);

// The fewest cases of a block from which the ply tables of the laminated shells are made from their slopes
// (ElementFamily::vectorisedLayerRecoveryMinCases): their stresses, strains and failure indices (the latter from the
// ply states), the slopes of every output of an element from one plate. Calibrated on a CompProp plate (22 500
// ShellQuad4, 8 plies, six threads): the per-case recovery takes about 0.50 s per case, the recovery by blocks 1.8 s
// once and 0.044 s per case: even from 2.4 cases; 4 leaves a margin for a block that asks for fewer ply tables (the
// slopes then shared by fewer rows).
inline constexpr std::size_t LAMINATE_VECTORISED_RECOVERY_MIN_CASES = 4;
// LAMINATE_VECTORISED_RECOVERY_MIN_CASES for the tables, 0 for the ply states (no table).
[[nodiscard]] std::size_t laminateVectorisedRecoveryMinCases(LayerOutput output);
// The source of a ply table in the vectorised recovery (ElementFamily::vectorisedLayerSource): the ply states for the
// failure indices, the table itself for the stresses and strains.
[[nodiscard]] LayerOutput laminateVectorisedSource(LayerOutput output);

// The eight values of a stress row (plate stresses): the fibre distance, σx, σy, τxy, the angle of the major
// principal stress in degrees, the major and minor principal stresses and the von Mises stress √(σ1² − σ1σ2 + σ2²).
[[nodiscard]] std::array<double, 8> shellStressRow(double fibre, const std::array<double, 3>& stress);
// The eight values of a strain row, the shear strain γxy engineering: principal strains from (εx, εy, γxy/2), the
// angle ½·atan2(γxy, εx − εy) in degrees, and the von Mises strain (2/3)·√(ε1² − ε1ε2 + ε2²) (the equivalent strain
// of the plane state with εz left out).
[[nodiscard]] std::array<double, 8> shellStrainRow(double first, const std::array<double, 3>& strain);

// The failure index of a ply by a failure theory (the classical criteria of composite plies), from its stresses
// (σ1, σ2, τ12) and mechanical strains (ε1, ε2, γ12) in the ply frame:
// - HILL: σ1²/X² − σ1σ2/X² + σ2²/Y² + τ12²/S², X = Xt or Xc by the sign of σ1 (of σ1σ2 in the product term), Y = Yt or
//   Yc by the sign of σ2;
// - HOFF: (1/Xt − 1/Xc)σ1 + (1/Yt − 1/Yc)σ2 + σ1²/(Xt Xc) + σ2²/(Yt Yc) − σ1σ2/(Xt Xc) + τ12²/S²;
// - TSAI: F1σ1 + F2σ2 + F11σ1² + F22σ2² + F66τ12² + 2F12σ1σ2, F1 = 1/Xt − 1/Xc, F11 = 1/(Xt Xc), F2, F22 alike,
//   F66 = 1/S², F12 of the OrthoMat2D;
// - STRN: the largest of ε1/Xt (ε1 ≥ 0) or −ε1/Xc, ε2/Yt or −ε2/Yc, |γ12|/S, the allowables strains (STRN = 1.0) or
//   the stress allowables divided by E1, E2, G12.
[[nodiscard]] double plyFailureIndex(FailureTheory theory, const PlyAllowables& allowables,
                                     const ShellMaterial& material, const std::array<double, 3>& stress,
                                     const std::array<double, 3>& strain);

// The gradient of plyFailureIndex in the stresses and the mechanical strains of the ply: the quadratic
// criteria's ∂/∂σ (HILL with the allowables its signs select at the point), STRN's the largest ratio's ∂/∂ε;
// `isDefined` false where the index has no derivative (two ratios of STRN equal, the largest).
struct FailureIndexGradient
{
    std::array<double, 3> stress{};
    std::array<double, 3> strain{};
    bool isDefined = true;
};
[[nodiscard]] FailureIndexGradient plyFailureIndexGradient(FailureTheory theory, const PlyAllowables& allowables,
                                                           const ShellMaterial& material,
                                                           const std::array<double, 3>& stress,
                                                           const std::array<double, 3>& strain);

} // namespace oos::elements
