#pragma once

// Tube2 — rod of circular tube section (CTUBE, TubeProp):
//
//   | Tube2 | EID | PID | G1 | G2 |
//   | TubeProp | PID | MID | OD | T  | NSM | OD2 |
//
// Formulation: rods/RodFormulation with the section of the tube (PTUBE): inner diameter D_i = OD − 2T,
// A = π(OD² − D_i²)/4, J = π(OD⁴ − D_i⁴)/32, and the torsional stress at the outer fibre, C = OD/2. T blank or 0 is
// a solid rod (O_o_Reader gives a blank T as null).
//
// Facets: stiffness, mass, recovery (forces: axial, torsion; stresses: axial, sma, torsion, smt). Differential
// stiffness (SOL 105: rodDifferentialStiffness of the axial force). Design derivatives are not written yet.
//
// Not supported: a tapered tube (OD2 other than OD: its properties vary along the length, not reproduced yet), a
// material other than IsoMat — both refused by check(). Problems reported by check(): an unknown grid point, a PID that
// is not a TubeProp, an unknown MID, coincident grid points, T above OD/2.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/rods/RodFormulation.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The section of a circular tube of outer diameter `outerDiameter` and wall `thickness` (0 for a solid rod), as
// TubeProp defines it; the NSM is left to the caller.
[[nodiscard]] RodSection tubeSection(double outerDiameter, double thickness);

// The element family of Tube2. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class Tube2Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Tube2"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"TubeProp", "IsoMat"}; }

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
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;

private:
    // The batch's columns and those of the TubeProp table (empty when the model has no TubeProp).
    struct Columns
    {
        RodConnectivity connectivity;
        PropertyRows properties;
        IsotropicMaterialColumns materials;
        std::span<const std::int32_t> matId;
        std::span<const double> od;
        std::span<const double> t;
        base::ValidityView isThicknessGiven;
        std::span<const double> nonStructMass;
        std::span<const double> od2;
        base::ValidityView isOd2Given;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<RodData> resolve(const ElementBatchView& view, const Columns& columns,
                                                     std::int64_t row);
};

} // namespace oos::elements
