#pragma once

// Bar2 — simple beam with its section on a BarProp:
//
//   | Bar2 | EID | PID | GA  | GB  | X1/G0 | X2  | X3  | OFFT |
//   |      | PA  | PB  | W1A | W2A | W3A   | W1B | W2B | W3B  |
//   | BarProp | PID | MID | A   | I1  | I2    | J   | NSM |      |
//   |      | C1  | C2  | D1  | D2  | E1    | E2  | F1  | F2   |
//   |      | K1  | K2  | I12 |     |       |     |     |      |
//
// Formulation: bars_beams/BeamFormulation: Euler-Bernoulli bending in
// planes 1 and 2 with I12 coupling and the transverse shear flexibility of K1·A·G and K2·A·G (the shear relief
// Φ = 12·E·I/(K·A·G·L²) of a uniform bar), axial EA/L, torsion GJ/L; orientation by X1-X3 or G0, offsets WA, WB with
// OFFT, pin flags PA, PB condensed (BeamGeometry, PinRelease); E, G, RHO and the stress limits from the IsoMat of MID.
//
// Facets: stiffness, mass (lumped and coupled, BarProp NSM and the NSM cards), loads (PLOAD1, all load and scale
// types), recovery (forces: bending moments at A and B in planes 1 and 2, shears, axial force, torque; stresses:
// bending at C, D, E, F of each end, axial, maximum and minimum, margins of safety in tension and compression). Under
// BeamDistributedLoad the Bar2 is written with the layout of its stations (the ends A and B: bending moments,
// shears, axial, torque per station; sxc-sxf, axial, smax, smin, ms): recoverUnderLineLoads gives that layout. Thermal
// loads (BeamThermal: the mean temperature of LineElemTemperatureLoad or of GA and GB, the free axial expansion held;
// the recovery subtracts it); strains as the stresses over E; the strain energy from the stiffness
// (output::StiffnessRecovery). Differential stiffness (BeamDifferential) and design derivatives
// (Bar2Derivatives.cpp).
//
// The section of a BarShapeProp (a section type of the standard section library and its dimensions) is read from the
// sections table of the snapshot (bars_beams/TabulatedSection: the properties computed from the dimensions, with
// their shear factors, a table O_o_Reader provides): the element is then the Bar2 of that BarProp-equivalent
// section, the MID and the stress limits from the BarShapeProp's IsoMat.
//
// Not supported, refused by check() with a diagnostic: a BarShapeProp without a sections table (field PID, code
// unsupported-field); a material other than IsoMat; OFFT with E or an old numeric code. Problems reported
// by check(): an unknown grid point or G0, a PID that is not a BarProp or BarShapeProp, an unknown MID, a zero or
// parallel orientation vector, a zero length, a negative or inconsistent section (I1·I2 ≤ I12²), pin flags on DOFs
// without stiffness.
//
// Modelling choices: the offset system of OFFT "O" has its x-axis from GA to GB; the mass has no rotary or torsional
// inertia and ignores the pin flags; the loads are the exact fixed-end forces of the shear-flexible bar; the margin of
// the type-100 layout is the smaller of the tension and compression margins. The mass, the thermal and
// BeamDistributedLoad loads of an offset bar are those of its length between the offset ends, at the grid points under
// ModelParams OFFDEF ELMOFF (the default; an offset bar refused in a buckling analysis), through the offsets under
// LROFF and ELMZ.

#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/bars_beams/TabulatedSection.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"
#include "oos/elements/core/PropertyLookup.hpp"

namespace oos::elements
{

// The element family of Bar2. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class Bar2Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Bar2"; }
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
        return {"BarProp", "BarShapeProp", "IsoMat"};
    }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The vectorised recovery: the extremes and margins of a row completed from its stresses at the recovery
    // points (Bar2Recovery.cpp), from BAR_VECTORISED_RECOVERY_MIN_CASES cases; its slopes from one end-force map
    // (recoverySlopes).
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    // Its rows at unit displacements from one end-force map (output::SectionLoads).
    bool recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                    std::vector<ElementDof>& dofs,
                                    std::vector<ElementResultBlock>& blocks) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
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
    [[nodiscard]] std::vector<std::string> outputColumnsUnderLineLoads(ElementOutput output) const override;
    // One element type for the ordinary layout, another under line loads (the station layout).
    [[nodiscard]] std::string resultElementType(bool underLineLoads) const override;
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
    // The batch's columns and those of the BarProp and BarShapeProp tables (empty when the model has none), and of the
    // sections table, read once per facet call.
    struct Columns
    {
        BeamConnectionColumns connection;
        PropertyRows properties;
        IsotropicMaterialColumns materials;
        std::span<const std::int32_t> matId;
        std::span<const double> a;
        std::span<const double> i1;
        std::span<const double> i2;
        std::span<const double> i12;
        std::span<const double> j;
        std::span<const double> nonStructMass;
        std::array<base::FixedListView<double>, 4> points; // C, D, E, F
        base::FixedListView<double> k;
        PropertyRows barlProperties;
        std::span<const std::int32_t> barlMatId;
        TabulatedSectionColumns sections;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<BeamData> resolve(const ElementBatchView& view, const Columns& columns,
                                                      std::int64_t row);
};

} // namespace oos::elements
