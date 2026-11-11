#pragma once

// Rod2 — rod with its section on a RodProp (CROD, RodProp):
//
//   | Rod2 | EID | PID | G1 | G2 |
//   | RodProp | PID | MID | A  | J  | C | NSM |
//
// Formulation: rods/RodFormulation (axial EA/L and torsional GJ/L stiffness, lumped or coupled translational mass with
// ρA + NSM per unit length). E, G, RHO and the stress limits from the IsoMat
// of MID.
//
// Facets: stiffness, mass, recovery (forces: axial, torsion; stresses: axial, sma, torsion, smt). Differential
// stiffness (SOL 105: rodDifferentialStiffness of the axial force). Design derivatives: the Derivatives facet.
//
// Not supported: a material other than IsoMat (refused by check()). Problems reported by check(): an unknown grid
// point, a PID that is not a RodProp, an unknown MID, coincident grid points.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/rods/RodFormulation.hpp"

namespace oos::elements
{

// The element family of Rod2. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class Rod2Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Rod2"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"RodProp", "IsoMat"}; }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override
    {
        return rodOutputColumns(output);
    }
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override
    {
        return rodOutputColumnForms(output);
    }
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    // The vectorised recovery: completeRodRows, from ROD_VECTORISED_RECOVERY_MIN_CASES cases.
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override
    {
        return rodOutputColumns(output).empty() ? 0 : ROD_VECTORISED_RECOVERY_MIN_CASES;
    }
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;
    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementMatrixBlock& block) const override;
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                        ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;
    bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, RecoverySlopes& rates) const override;
    bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementOutput output, const NodalDisplacements& displacements,
                                  ElementResultBlock& rates) const override;
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, const NodalDisplacements& preload,
                                              ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldMixedDerivative(
        const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
        const NodalDisplacements& preload, ElementMatrixBlock& block) const override;

private:
    // The batch's columns and those of the RodProp table (empty when the model has no RodProp), read once per facet
    // call.
    struct Columns
    {
        RodConnectivity connectivity;
        PropertyRows properties;
        IsotropicMaterialColumns materials;
        std::span<const std::int32_t> matId;
        std::span<const double> a;
        std::span<const double> j;
        std::span<const double> c;
        std::span<const double> nonStructMass;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<RodData> resolve(const ElementBatchView& view, const Columns& columns,
                                                     std::int64_t row);
};

} // namespace oos::elements
