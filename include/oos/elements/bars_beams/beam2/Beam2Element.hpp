#pragma once

// Beam2 — beam with a tapered section on a BeamProp:
//
//   | Beam2 | EID    | PID    | GA     | GB     | X1/G0  | X2     | X3     | OFFT   |
//   |       | PA     | PB     | W1A    | W2A    | W3A    | W1B    | W2B    | W3B    |
//   |       | SA     | SB     |        |        |        |        |        |        |
//   | BeamProp | PID    | MID    | A(A)   | I1(A)  | I2(A)  | I12(A) | J(A)   | NSM(A) |
//   |       | C1(A)  | C2(A)  | D1(A)  | D2(A)  | E1(A)  | E2(A)  | F1(A)  | F2(A)  |
//   |       | SO     | X/XB   | A      | I1     | I2     | I12    | J      | NSM    |   (per further station)
//   |       | C1     | C2     | D1     | D2     | E1     | E2     | F1     | F2     |
//   |       | K1     | K2     | S1     | S2     | NSI(A) | NSI(B) | CW(A)  | CW(B)  |
//   |       | M1(A)  | M2(A)  | M1(B)  | M2(B)  | N1(A)  | N2(A)  | N1(B)  | N2(B)  |
//
// Formulation: bars_beams/BeamFormulation, the flexibility of the tapered beam integrated along its length with the
// section varying linearly between the stations (the stiffness of a tapered beam by inversion of its flexibility,
// Euler-Bernoulli bending with shear flexibility K1·A(x)·G, K2·A(x)·G; the linear interpolation between stations is
// a modelling choice of this project). A Beam2 of uniform section is a Bar2 (same matrices). Orientation,
// offsets, OFFT and pin flags as Bar2.
//
// Facets: stiffness, mass (lumped and coupled with ρ·A(x) + NSM(x) and the NSM cards), loads (PLOAD1), recovery at the
// stations (end A, the intermediate stations whose SO is not NO, end B; one row per station with the grid point ID at
// the ends, 0 between, the station number as the layer and X/XB as the position): forces (bending moments, shears,
// axial force, total torque, warping torque 0) and stresses (the longitudinal stresses at C, D, E, F — axial plus
// bending — their maximum and minimum and the margins of safety). Thermal loads (BeamThermal, as Bar2). The
// differential stiffness (BeamDifferential) and design derivatives (Beam2Derivatives.cpp).
//
// The element axis is the axis of the shear centres (the offsets WA, WB go from the grid points to the ends of the
// shear-centre axis): the shears and the torque act there, a BeamDistributedLoad along it. The neutral axis N1, N2
// (linear in X/XB between ends A and B) carries the axial force and the bending moments (BeamFlexibility, BeamResponse:
// the output moments about it); the structural mass is on it and the NSM at its centre of gravity M1, M2 (beamGridMass;
// hypotheses there); the recovery points of the card are from the shear centre, those of a BeamStation from its
// neutral axis.
//
// Not supported, refused by check() with a diagnostic (code unsupported-field): warping (SA or SB given: the warping
// DOFs); the shear relief of taper S1, S2 not 0; the non-structural mass inertia NSI not 0; a BeamLumpedAreaProp; a
// BeamShapeProp without a sections table (as the Bar2's BarShapeProp); a material other than IsoMat; OFFT with E or
// an old numeric code. Other problems as Bar2.
//
// Taken and not used: the warping coefficients CW(A), CW(B) of a BeamProp. Without SA and SB a Beam2 has no warping
// DOF, and the input format then leaves the torsional stiffness of warping out, so CW enters no matrix and the torsion
// is G·J alone (free warping: the warping torque of the recovery is 0). check() reports one information per BeamProp
// with a CW (code ignored-field, the PID and the field), and a designed CW changes nothing (designFieldUse).
//
// A BeamShapeProp (a section type of the standard section library, its dimensions at end A and at up to nine stations
// and end B, and NSM per station) is read from the sections table of the snapshot (bars_beams/TabulatedSection): one
// row per station with the properties computed from the dimensions, the shear factors (those of end A for the element),
// the recovery points from the neutral axis and the neutral axis from the shear centre (n1, n2: the element system of
// a BeamShapeProp is at the shear centre; the snapshot's columns from O_o_Reader's section properties); the SO of the
// card's stations; its NSM at the shear centre (the derived BeamProp has no M1, M2: a hypothesis); the BeamShapeProp
// has no field of the refused list above (no warping, S1, S2, NSI).
//
// Modelling choices: those of Bar2, the linear interpolation between stations, the rows of
// the stations (no row for an intermediate station with SO = NO; the ends always have one, an end without stress points
// giving the axial stress at the four points), and the stress columns sxc-sxf including the axial stress.

#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/bars_beams/BeamResponse.hpp"
#include "oos/elements/bars_beams/TabulatedSection.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/input/tables/NestedListView.hpp"

namespace oos::elements
{

// The element family of Beam2. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class Beam2Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Beam2"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool loadFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const ElementLoadContext& loads, ElementVectorBlock& block) const override;
    bool loadFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                  const DesignedField& first, const DesignedField& second,
                                  const ElementLoadContext& loads, ElementVectorBlock& block) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override
    {
        return {"BeamProp", "BeamShapeProp", "IsoMat"};
    }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The recovery matrix of a Beam2 from one end-force map (BeamEndForces), as the Bar2's (cheaper than
    // recoveries at unit displacements), and its vectorised recovery from
    // BEAM_VECTORISED_RECOVERY_MIN_CASES cases: the extremes of each station and the margins at the station of the
    // larger stress completed from the stresses at C, D, E, F.
    // Its rows at unit displacements from one end-force map (output::SectionLoads).
    bool recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                    std::vector<ElementDof>& dofs,
                                    std::vector<ElementResultBlock>& blocks) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;

    void checkLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                        base::DiagnosticSink& sink) const override;
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    void computeLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                          ElementVectorBlock& block) const override;
    // SOL 105: bars_beams/BeamDifferential.
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
    void recoverUnderLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                               ElementOutput output, const NodalDisplacements& displacements,
                               ElementResultBlock& block) const override;

    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementMatrixBlock& block) const override;
    bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                        ElementMatrixBlock& block) const override;
    bool stiffnessFieldMixedDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& first,
                                       const DesignedField& second, ElementMatrixBlock& block) const override;
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;
    bool massFieldMixedDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& first,
                                  const DesignedField& second, const MassOptions& options,
                                  ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementOutput output, const NodalDisplacements& displacements,
                                  ElementResultBlock& rates) const override;
    bool recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row, const DesignedField& first,
                                 const DesignedField& second, ElementOutput output,
                                 const NodalDisplacements& displacements, ElementResultBlock& rates) const override;
    bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, RecoverySlopes& rates) const override;
    bool differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, const NodalDisplacements& preload,
                                              ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldMixedDerivative(
        const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
        const NodalDisplacements& preload, ElementMatrixBlock& block) const override;

private:
    // The batch's columns and those of the BeamProp table (empty when the model has no BeamProp), read once per facet
    // call.
    struct Columns
    {
        BeamConnectionColumns connection;
        std::span<const std::int32_t> sa;
        std::span<const std::int32_t> sb;
        PropertyRows properties;
        IsotropicMaterialColumns materials;
        std::span<const std::int32_t> matId;
        base::ListView<std::uint8_t> stationOutput; // SO: 0 NO, 1 YES, 2 YESA (O_o_Reader BeamStressOutput)
        base::ListView<double> stationPosition;     // X/XB
        input::NestedListView stationSection;       // A, I1, I2, I12, J, NSM
        input::NestedListView stationStress;        // C1, C2, D1, D2, E1, E2, F1, F2
        base::FixedListView<double> k;              // K1, K2
        base::FixedListView<double> s;              // S1, S2
        base::FixedListView<double> nsi;            // NSI(A), NSI(B)
        base::FixedListView<double> cw;             // CW(A), CW(B)
        base::FixedListView<double> m;              // M1(A), M2(A), M1(B), M2(B)
        base::FixedListView<double> n;              // N1(A), N2(A), N1(B), N2(B)
        PropertyRows beamlProperties;
        std::span<const std::int32_t> beamlMatId;
        base::ListView<double> beamlPositions;     // BeamShapeProp X/XB of the stations the card gives
        base::ListView<std::uint8_t> beamlOutputs; // BeamShapeProp SO of those stations
        TabulatedSectionColumns sections;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<BeamData> resolve(const ElementBatchView& view, const Columns& columns,
                                                      std::int64_t row);
    // The rest of resolve() for an element on a BeamShapeProp: `beam` holds its geometry.
    [[nodiscard]] static Resolution<BeamData> resolveBeaml(const ElementBatchView& view, const Columns& columns,
                                                           std::int64_t row, BeamData beam);
    // The station rows of one element (forces or stresses), appended to `block`.
    static void appendStationRows(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                  std::span<const BeamLineLoad> loads, ElementOutput output,
                                  const NodalDisplacements& displacements, ElementResultBlock& block);
    // The station rows of one element under `response` (appendStationRows).
    static void appendResponseRows(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                   const BeamResponse& response, ElementOutput output, ElementResultBlock& block);
};

} // namespace oos::elements
