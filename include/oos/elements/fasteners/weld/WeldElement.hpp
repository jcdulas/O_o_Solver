#pragma once

// Weld — weld or fastener between two shells (CWELD, WeldProp):
//
//   | Weld | EWID | PWID | GS   | TYPE  | GA    | GB    |      | MCID |
//   |       | (PARTPAT, ELPAT: PIDA/SHIDA, PIDB/SHIDB; ELEMID: SHIDA, SHIDB; GRIDID: SPTYP, GA1-GA8, GB1-GB8)    |
//   | WeldProp | PID  | MID  | D    |       |       | MSET  |      | TYPE |
//   |       | LDMIN| LDMAX|
//
// Connectivity (fasteners/FastenerConnection, the fastener piercing table the reader computes): each piercing point
// moves with the reference point of an RigidInterpolation interpolation element whose independent points are the grid
// points of the shell element it pierces (PARTPAT: a shell of PIDA or PIDB; ELPAT: SHIDA, SHIDB or a neighbour; ELEMID:
// SHIDA, SHIDB; GRIDID: the face of the grid points GA1-GA8, GB1-GB8), weighted by the shape functions of that element
// at the point (fasteners/PiercingInterpolation); ALIGN connects GA and GB directly, and ELEMID without SHIDB or GRIDID
// with SPTYP Q or T (a point-to-patch connection) connects GS directly as side B. The auxiliary points are eliminated
// at element level (PWELD MSET = OFF, the default; MSET = ON puts the same constraint equations in the m set, which
// gives the same solution and is treated alike).
//
// Stiffness: a shear-deformable (Timoshenko) beam of circular section of diameter D between A and B (A = πD²/4,
// I = πD⁴/64, J = πD⁴/32), E and G of the IsoMat of WeldProp MID, by bars_beams/BeamFlexibility (the flexibility
// method, Przemieniecki, Theory of Matrix Structural Analysis, §5.6: the transverse terms 12EI/(L³(1 + Φ)) with the
// shear relief Φ = 12EI/(k·A·G·L²)), in the weld system E (x from A to B, fastenerAxes). The shear coefficient k of a
// solid circle is Cowper's (G. R. Cowper, "The shear coefficient in Timoshenko's beam theory", Journal of Applied
// Mechanics 33, 1966, pp. 335-340): k = 6(1 + ν)/(7 + 6ν), ν = E/(2G) − 1 (weldShearCoefficient).
//
// The length of the beam (weldBeamLength) is the distance between the mid-surfaces of the two sheets: for sheets in
// contact that is the mean of their thicknesses, L = (tA + tB)/2, and when the piercing points are farther apart their
// distance, L = max(|AB|, (tA + tB)/2) (the weld spans the sheets from mid-surface to mid-surface, the usual beam
// model of a spot weld: M. Palmonella, M. I. Friswell, J. E. Mottershead, A. W. Lees, "Finite element models of spot
// welds in structural dynamics: review and updating", Computers & Structures 83, 2005, pp. 648-661). tA and tB are
// the thicknesses of the shells the card names (weldSideThickness: PARTPAT the properties PIDA, PIDB; ELPAT and
// ELEMID the elements SHIDA, SHIDB; ShellProp T, or the total thickness of a CompProp or CompPropGlobal); when a side
// has none (ALIGN, GRIDID, ELEMID without SHIDB, a shell without T) L = |AB|. L is then kept between LDMIN·D and
// LDMAX·D (PWELD: the smallest and largest ratio of length to diameter used in the stiffness). When L differs from |AB|
// the beam of length L sits at the middle of A-B with rigid arms to A and B, so that a rigid motion still gives no
// force (a choice of this implementation).
//
// Facets: stiffness; mass (ρ·A·|AB| lumped half at A, half at B on the translations, carried to the grid points of the
// pierced elements; not validated); recovery: forces in the Bar2 layout of the data contract (forces.weld:
// bending moments at A and B in planes 1 and 2, shears, axial force, torque, signs of bars_beams/BeamResponse) and
// stresses (stress.weld: axial N/A; maximum and minimum at A and B, N/A ± M·(D/2)/I with M the resultant bending
// moment there; max_shear the peak transverse shear stress of a solid circle, 4V/(3A) = 16V/(3πD²), plus the torsional
// stress at its edge, 16|T|/(πD³) (J. M. Gere, B. J. Goodno, Mechanics of Materials: the shear stresses of a beam of
// circular section and the torsion of a circular bar), and bearing V/D², V the resultant shear and T the torque).
//
// Not supported, refused by check(): MCID ≥ 0 (the weld system from a coordinate system); a material other than IsoMat.
// Problems: a PWID that is not a WeldProp, the problems of FastenerConnection (no piercing table).

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamSection.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"
#include "oos/elements/fasteners/FastenerConnection.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace oos::elements
{

// Cowper's shear coefficient of a solid circular section, k = 6(1 + ν)/(7 + 6ν) (WeldElement above): the shear area
// is k·A.
[[nodiscard]] constexpr double weldShearCoefficient(double poissonRatio) noexcept
{
    return 6.0 * (1.0 + poissonRatio) / (7.0 + 6.0 * poissonRatio);
}

// The thickness of the shell that side `id` of a Weld of form `type` names (PARTPAT: the property PIDA or PIDB;
// ELPAT, ELEMID: the element SHIDA or SHIDB, the thickness of its property): ShellProp T, or the total thickness of a
// CompProp or CompPropGlobal. Nothing for the other forms, an ID of 0, or a shell whose thickness the snapshot does not
// give.
[[nodiscard]] std::optional<double> weldSideThickness(const input::ModelSnapshot& snapshot, std::string_view type,
                                                      std::int32_t id);
// The shell property whose thickness side `id` of a Weld of form `type` reads (PARTPAT: PIDA or PIDB itself; ELPAT,
// ELEMID: the property of the element SHIDA or SHIDB), or nothing. A designed thickness of that property (PSHELL T, the
// plies of a CompProp or CompPropGlobal) changes the weld when the mean thickness sets its length (design/sensitivity/
// AffectedElements reaches the weld through it).
[[nodiscard]] std::optional<std::int32_t> weldSideProperty(const input::ModelSnapshot& snapshot, std::string_view type,
                                                           std::int32_t id);

// The length of the weld beam: max(|AB|, (tA + tB)/2) when both thicknesses are known, else |AB|, then kept between
// LDMIN·D and LDMAX·D (WeldElement above).
[[nodiscard]] double weldBeamLength(double distance, std::optional<double> thicknessA, std::optional<double> thicknessB,
                                    double diameter, double ldmin, double ldmax);

// The element family of Weld. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class WeldElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Weld"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"WeldProp", "IsoMat"}; }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The vectorised recovery (LinearRowRecovery.hpp): every column Linear, the slopes in closed form.
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;
    // The stresses of a block made from its forces (their resultants are not linear in the displacements).
    [[nodiscard]] ElementOutput vectorisedSource(ElementOutput output) const override;
    void completeRecoveredRowsFromSource(const ElementBatchView& view, ElementOutput output,
                                         std::span<const ElementResultBlock> sources,
                                         std::span<ElementResultBlock> blocks) const override;

    // One resolved Weld: its connection, the map from w = (u_A, θ_A, u_B, θ_B) in basic to the end DOFs of the beam
    // in the weld system (rigid arms included), the beam and its stiffness over its end DOFs.
    struct Weld
    {
        FastenerConnection connection;
        base::MatN<12> toBeamEnds;    // d = T·w
        base::MatN<12> beamStiffness; // over d
        BeamSection section;
        IsotropicMaterial material;
        double diameter = 0.0;
        double beamLength = 0.0;
        double distance = 0.0;         // |B − A|
        double shearCoefficient = 0.0; // Cowper's k
    };

private:
    struct Columns
    {
        std::span<const std::int32_t> pwid;
        base::Utf8View type;
        std::span<const std::int32_t> gs;
        std::span<const std::int32_t> ga;
        std::span<const std::int32_t> gb;
        std::span<const std::int32_t> matFrameId;
        std::span<const std::int32_t> ida;
        std::span<const std::int32_t> idb;
        std::optional<base::Utf8View> sptyp; // GRIDID: the shapes of the faces (Q, T: a point-to-patch connection)
        const input::CardTable* weldProp = nullptr;
        std::span<const std::int32_t> weldPropIds;
        std::span<const std::int32_t> matId;
        std::span<const double> d;
        std::span<const double> ldmin;
        std::span<const double> ldmax;
        IsotropicMaterialColumns materials;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<Weld> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
    // The section of the beam of a weld of diameter D (a circle: A = πD²/4, I1 = I2 = πD⁴/64, J = 2·I1, the recovery
    // points on the circle), at station 0.
    [[nodiscard]] static BeamStation weldStation(double diameter);
    // The WeldProp D of the weld of a row, without its piercing points (the vectorised recovery of its stresses).
    // Precondition: check() accepted the row.
    [[nodiscard]] static double diameterOf(const ElementBatchView& view, const Columns& columns, std::int64_t row);
};

} // namespace oos::elements
