#pragma once

// ShellTria6 — six-node shell (CTRIA6, ShellProp, CompProp, CompPropGlobal).
//
// Formulation: MindlinShell<QuadraticTriangleShape>, the six-node degenerated shell, flat or curved (the quadratic
// membrane and Reissner-Mindlin bending of a flat element, three-point integration, the membrane-bending coupling,
// no stiffness for the rotation about the nodal director: no K6ROT), through the facets shared with the other
// higher-order shell (HigherOrderShell.hpp, which says what is supported and refused).
// Covered by the patch, rigid-body and convergence tests.
//
// Facets: stiffness, mass, loads (pressure, thermal), recovery (centre and corner grid points; the plies of a
// laminate at the centre), differential stiffness (the membrane resultants of the preload at the centre,
// shells/ShellDifferential). Not written yet: strain energy, derivatives.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/HigherOrderShapes.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"

namespace oos::elements
{

// The element family of ShellTria6. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class ShellTria6Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "ShellTria6"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    // The closed forms in a AnisoMat2D stiffness term (HigherOrderShell's); every other field differenced.
    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, const NodalDisplacements& preload,
                                              ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    bool layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                 LayerOutput output, const NodalDisplacements& displacements,
                                 ElementResultBlock& rates) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override
    {
        return {"ShellProp", "CompProp", "CompPropGlobal", "IsoMat", "AnisoMat2D", "OrthoMat2D"};
    }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::vector<std::string> layerOutputColumns(LayerOutput output) const override;
    // The forms of the shells' columns, and the recovery matrices of the element from one plate and its place
    // states at unit local displacements (higherOrderShellSlopes).
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override
    {
        return shellOutputColumnForms(output);
    }
    [[nodiscard]] std::vector<ColumnForm> layerOutputColumnForms(LayerOutput output) const override
    {
        return laminateOutputColumnForms(output);
    }
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                             RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    // The vectorised recovery: the rows completed as the ShellQuad4's
    // (completeShellRows, completeLaminateRows, the failure indices from the ply states), from
    // SHELL_VECTORISED_RECOVERY_MIN_CASES and LAMINATE_VECTORISED_RECOVERY_MIN_CASES cases (ShellRecovery.hpp).
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override
    {
        return outputColumns(output).empty() ? 0 : SHELL_VECTORISED_RECOVERY_MIN_CASES;
    }
    [[nodiscard]] std::size_t vectorisedLayerRecoveryMinCases(LayerOutput output) const override
    {
        return laminateVectorisedRecoveryMinCases(output);
    }
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;
    [[nodiscard]] LayerOutput vectorisedLayerSource(LayerOutput output) const override
    {
        return laminateVectorisedSource(output);
    }
    void completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                    std::span<const ElementResultBlock> sources,
                                    std::span<ElementResultBlock> blocks) const override;
    void recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                       const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The rows of several outputs from one formulation of each element (the rows at rest of a block of cases).
    void recoverOutputs(const ElementBatchView& view, ElementRange range, std::span<const RecoveryOutput> outputs,
                        const NodalDisplacements& displacements, std::span<ElementResultBlock> blocks) const override;
    void checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                            base::DiagnosticSink& sink) const override;
    void computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                              ElementVectorBlock& block) const override;
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
};

} // namespace oos::elements
