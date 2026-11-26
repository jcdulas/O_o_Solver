#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::reduction
{

// An element family whose elements replaced by an external unit are left out (the unit replaces the elements of the
// zone by matrices the plugin computes): it gives what the family it wraps gives, but
//  - no stiffness, mass or differential stiffness for a replaced element (the unit's K* and M* stand for them);
//  - no result rows for one (forces, stresses, strains, layers: its interior results are the plugin's);
//  - no thermal load for one (an analysis with an external unit and a thermal load is refused: the zone's temperatures
//    are not given to the plugin in this version);
// and passes the pressure and line loads of its elements through: their equivalent loads at the interior grid points
// of the zone are the loads of the zone given to the plugin (its condensed loads f* bring them back), those at the
// contour points stay loads of the analysis (the loads of the zone are those of its interior grid points).
//
// Every assembler, recovery and sensitivity of the analysis then leaves the replaced elements out without knowing of
// external units: the element set is built on the families of ReplacedElements.
//
// Every virtual of elements::ElementFamily is overridden here, so that the elements that are not replaced are computed
// by their own family in full (a member left to the base class would answer for them as a family that gives nothing:
// no corner rows, no column forms, no closed-form derivative). A member added to ElementFamily must be added here
// too. The rule of each member:
//  - a member of the card (its name, its facets, its columns, its column forms, the fewest cases of its vectorised
//    recovery): the wrapped family's answer;
//  - a member on a range of rows: the wrapped family's on the runs of the range without a replaced element (keptRuns),
//    a replaced element appending no matrix and no row, and holding 0 in a per-row array (volumes, stabilisation
//    energies);
//  - a member on rows the caller names (the rows of result blocks to complete): the wrapped family's, the caller's
//    rows being those the members above gave, none of them a replaced element's;
//  - a member on one row: the wrapped family's for an element that is not replaced; for a replaced element the answer
//    of the base class, that of a family without that closed form (false, or the base class's known zero), so that a
//    caller that falls back on the facets above finds nothing to difference. Its load derivatives alone are the
//    wrapped family's, without the temperatures: its pressure and line loads pass through, its thermal load does not.
//
// Thread-safety: immutable; its facets are as thread-safe as those of the wrapped family. Ownership: borrows the
// wrapped family and the sorted list of replaced IDs (ReplacedElements owns both lifetimes).
class ReplacedElementFamily final : public elements::ElementFamily
{
public:
    ReplacedElementFamily(const elements::ElementFamily& inner, std::span<const base::EntityId> replaced);

    [[nodiscard]] std::string_view card() const noexcept override { return m_inner->card(); }
    [[nodiscard]] std::string_view formulation() const noexcept override { return m_inner->formulation(); }
    [[nodiscard]] elements::FacetSet facets() const noexcept override { return m_inner->facets(); }
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return m_inner->propertyCards(); }
    [[nodiscard]] std::vector<input::FieldRule> fieldRules() const override { return m_inner->fieldRules(); }

    // The checks read the cards of every element, the replaced ones included (a zone is made of elements the model
    // defines in full).
    void check(const elements::ElementBatchView& view, base::DiagnosticSink& sink) const override;
    [[nodiscard]] bool isCheckedByRow() const noexcept override { return m_inner->isCheckedByRow(); }
    void checkRows(const elements::ElementBatchView& view, elements::ElementRange range,
                   base::DiagnosticSink& sink) const override;

    void computeStiffness(const elements::ElementBatchView& view, elements::ElementRange range,
                          elements::ElementMatrixBlock& block) const override;
    void computeMass(const elements::ElementBatchView& view, elements::ElementRange range,
                     const elements::MassOptions& options, elements::ElementMatrixBlock& block) const override;
    // The volume of a replaced element is 0: its mass is in the unit's M*, as its lumped mass at unit density says
    // for a family that gives no volume (assembly::ElementVolumes).
    [[nodiscard]] bool computeVolumes(const elements::ElementBatchView& view, elements::ElementRange range,
                                      std::span<double> volumes) const override;
    // The stabilisation energy of a replaced element is 0: its stiffness is the unit's.
    [[nodiscard]] bool computeStabilisationEnergies(const elements::ElementBatchView& view,
                                                    elements::ElementRange range,
                                                    const elements::NodalDisplacements& displacements,
                                                    std::span<double> energies) const override;

    [[nodiscard]] std::vector<std::string> outputColumns(elements::ElementOutput output) const override;
    void recover(const elements::ElementBatchView& view, elements::ElementRange range, elements::ElementOutput output,
                 const elements::NodalDisplacements& displacements, elements::ElementResultBlock& block) const override;

    void checkPressureLoads(const elements::ElementBatchView& view,
                            std::span<const elements::ElementPressure> pressures,
                            base::DiagnosticSink& sink) const override;
    void computePressureLoads(const elements::ElementBatchView& view,
                              std::span<const elements::ElementPressure> pressures,
                              elements::ElementVectorBlock& block) const override;
    void checkLineLoads(const elements::ElementBatchView& view, std::span<const elements::ElementLineLoad> loads,
                        base::DiagnosticSink& sink) const override;
    void computeLineLoads(const elements::ElementBatchView& view, std::span<const elements::ElementLineLoad> loads,
                          elements::ElementVectorBlock& block) const override;
    void checkThermalLoads(const elements::ElementBatchView& view, const elements::ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const elements::ElementBatchView& view, elements::ElementRange range,
                             const elements::ElementTemperatures& temperatures,
                             elements::ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return m_inner->isThermallyInert(); }

    void checkDifferentialStiffness(const elements::ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeDifferentialStiffness(const elements::ElementBatchView& view, elements::ElementRange range,
                                      const elements::NodalDisplacements& preload,
                                      elements::ElementMatrixBlock& block) const override;
    [[nodiscard]] bool isDifferentiallyInert() const noexcept override { return m_inner->isDifferentiallyInert(); }

    [[nodiscard]] std::vector<std::string> outputColumnsUnderLineLoads(elements::ElementOutput output) const override
    {
        return m_inner->outputColumnsUnderLineLoads(output);
    }
    void recoverUnderLineLoads(const elements::ElementBatchView& view, std::span<const elements::ElementLineLoad> loads,
                               elements::ElementOutput output, const elements::NodalDisplacements& displacements,
                               elements::ElementResultBlock& block) const override;
    // The station rows of the inner family that are not replaced.
    [[nodiscard]] std::vector<std::int64_t> stationRows(const elements::ElementBatchView& view) const override;
    void recoverStationRows(const elements::ElementBatchView& view, std::span<const std::int64_t> rows,
                            std::span<const elements::ElementLineLoad> loads, elements::ElementOutput output,
                            const elements::NodalDisplacements& displacements,
                            elements::ElementResultBlock& block) const override;
    [[nodiscard]] std::string resultElementType(bool underLineLoads) const override
    {
        return m_inner->resultElementType(underLineLoads);
    }
    [[nodiscard]] std::vector<std::string> layerOutputColumns(elements::LayerOutput output) const override
    {
        return m_inner->layerOutputColumns(output);
    }
    void recoverLayers(const elements::ElementBatchView& view, elements::ElementRange range,
                       elements::LayerOutput output, const elements::NodalDisplacements& displacements,
                       elements::ElementResultBlock& block) const override;
    [[nodiscard]] std::string layerResultElementType() const override { return m_inner->layerResultElementType(); }
    // The rows of several outputs at once, the corner rows of an output asked with them (RecoveryOutput::hasCorners,
    // which this member alone reads: recover() gives the centre rows).
    void recoverOutputs(const elements::ElementBatchView& view, elements::ElementRange range,
                        std::span<const elements::RecoveryOutput> outputs,
                        const elements::NodalDisplacements& displacements,
                        std::span<elements::ElementResultBlock> blocks) const override;
    bool recoverAtUnitDisplacements(const elements::ElementBatchView& view, std::int64_t row,
                                    elements::ElementOutput output, std::vector<elements::ElementDof>& dofs,
                                    std::vector<elements::ElementResultBlock>& blocks) const override;

    [[nodiscard]] elements::FieldUse designFieldUse(const elements::DesignedField& field) const override
    {
        return m_inner->designFieldUse(field);
    }

    // The forms of the columns and the vectorised recovery: those of the card.
    [[nodiscard]] std::vector<elements::ColumnForm> outputColumnForms(elements::ElementOutput output) const override
    {
        return m_inner->outputColumnForms(output);
    }
    [[nodiscard]] std::vector<elements::ColumnForm> layerOutputColumnForms(elements::LayerOutput output) const override
    {
        return m_inner->layerOutputColumnForms(output);
    }
    bool recoverySlopes(const elements::ElementBatchView& view, std::int64_t row, elements::ElementOutput output,
                        elements::RecoverySlopes& slopes) const override;
    bool layerRecoverySlopes(const elements::ElementBatchView& view, std::int64_t row, elements::LayerOutput output,
                             elements::RecoverySlopes& slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(elements::ElementOutput output) const override
    {
        return m_inner->vectorisedRecoveryMinCases(output);
    }
    [[nodiscard]] std::size_t vectorisedLayerRecoveryMinCases(elements::LayerOutput output) const override
    {
        return m_inner->vectorisedLayerRecoveryMinCases(output);
    }
    bool recoverySlopesOfOutputs(const elements::ElementBatchView& view, std::int64_t row,
                                 std::span<const elements::RecoveryOutput> outputs,
                                 std::span<elements::RecoverySlopes> slopes) const override;
    void completeRecoveredRows(const elements::ElementBatchView& view, elements::ElementOutput output,
                               std::span<elements::ElementResultBlock> blocks) const override;
    [[nodiscard]] elements::ElementOutput vectorisedSource(elements::ElementOutput output) const override
    {
        return m_inner->vectorisedSource(output);
    }
    void completeRecoveredRowsFromSource(const elements::ElementBatchView& view, elements::ElementOutput output,
                                         std::span<const elements::ElementResultBlock> sources,
                                         std::span<elements::ElementResultBlock> blocks) const override;
    [[nodiscard]] elements::LayerOutput vectorisedLayerSource(elements::LayerOutput output) const override
    {
        return m_inner->vectorisedLayerSource(output);
    }
    void completeRecoveredLayerRows(const elements::ElementBatchView& view, elements::LayerOutput output,
                                    std::span<const elements::ElementResultBlock> sources,
                                    std::span<elements::ElementResultBlock> blocks) const override;

    // The closed-form derivatives of one element (the class comment gives the rule of a replaced one).
    bool stiffnessFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                  const elements::DesignedField& field,
                                  elements::ElementMatrixBlock& block) const override;
    void stiffnessFieldDerivatives(const elements::ElementBatchView& view, std::int64_t row,
                                   std::span<const elements::DesignedField> fields,
                                   std::span<elements::ElementMatrixBlock> blocks,
                                   std::span<std::uint8_t> isClosed) const override;
    bool massFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                             const elements::DesignedField& field, const elements::MassOptions& options,
                             elements::ElementMatrixBlock& block) const override;
    bool loadFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                             const elements::DesignedField& field, const elements::ElementLoadContext& loads,
                             elements::ElementVectorBlock& block) const override;
    bool loadFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                  const elements::DesignedField& first, const elements::DesignedField& second,
                                  const elements::ElementLoadContext& loads,
                                  elements::ElementVectorBlock& block) const override;
    bool differentialStiffnessFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                              const elements::DesignedField& field,
                                              const elements::NodalDisplacements& preload,
                                              elements::ElementMatrixBlock& block) const override;
    bool differentialStiffnessPreloadGradient(const elements::ElementBatchView& view, std::int64_t row,
                                              const elements::NodalDisplacements& shape,
                                              elements::ElementVectorBlock& block) const override;
    bool differentialStiffnessFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                   const elements::DesignedField& first,
                                                   const elements::DesignedField& second,
                                                   const elements::NodalDisplacements& preload,
                                                   elements::ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const elements::ElementBatchView& view, std::int64_t row,
                            const elements::DesignedField& field, elements::ElementOutput output,
                            const elements::NodalDisplacements& displacements,
                            elements::ElementResultBlock& rates) const override;
    void recoveryFieldRatesOfFields(const elements::ElementBatchView& view, std::int64_t row,
                                    std::span<const elements::DesignedField> fields, elements::ElementOutput output,
                                    const elements::NodalDisplacements& displacements,
                                    std::span<elements::ElementResultBlock> rates,
                                    std::span<std::uint8_t> isClosed) const override;
    bool layerRecoveryFieldRates(const elements::ElementBatchView& view, std::int64_t row,
                                 const elements::DesignedField& field, elements::LayerOutput output,
                                 const elements::NodalDisplacements& displacements,
                                 elements::ElementResultBlock& rates) const override;
    void layerRecoveryFieldRatesOfFields(const elements::ElementBatchView& view, std::int64_t row,
                                         std::span<const elements::DesignedField> fields, elements::LayerOutput output,
                                         const elements::NodalDisplacements& displacements,
                                         std::span<elements::ElementResultBlock> rates,
                                         std::span<std::uint8_t> isClosed) const override;
    bool stiffnessFieldSecondDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                        const elements::DesignedField& field,
                                        elements::ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                   const elements::DesignedField& field, const elements::MassOptions& options,
                                   elements::ElementMatrixBlock& block) const override;
    bool stiffnessFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                       const elements::DesignedField& first, const elements::DesignedField& second,
                                       elements::ElementMatrixBlock& block) const override;
    bool massFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                  const elements::DesignedField& first, const elements::DesignedField& second,
                                  const elements::MassOptions& options,
                                  elements::ElementMatrixBlock& block) const override;
    bool recoverySlopeRates(const elements::ElementBatchView& view, std::int64_t row,
                            const elements::DesignedField& field, elements::ElementOutput output,
                            elements::RecoverySlopes& rates) const override;
    bool recoveryFieldSecondRates(const elements::ElementBatchView& view, std::int64_t row,
                                  const elements::DesignedField& field, elements::ElementOutput output,
                                  const elements::NodalDisplacements& displacements,
                                  elements::ElementResultBlock& rates) const override;
    bool recoveryFieldMixedRates(const elements::ElementBatchView& view, std::int64_t row,
                                 const elements::DesignedField& first, const elements::DesignedField& second,
                                 elements::ElementOutput output, const elements::NodalDisplacements& displacements,
                                 elements::ElementResultBlock& rates) const override;

    // Whether element `elementId` is replaced.
    [[nodiscard]] bool isReplaced(base::EntityId elementId) const noexcept;

private:
    // Whether the element of batch row `row` is replaced.
    [[nodiscard]] bool isReplacedRow(const elements::ElementBatchView& view, std::int64_t row) const;
    // The runs of rows of `range` that hold no replaced element, in order.
    [[nodiscard]] std::vector<elements::ElementRange> keptRuns(const elements::ElementBatchView& view,
                                                               elements::ElementRange range) const;
    // Appends the rows of `from` to `to` (the same columns).
    static void appendRows(const elements::ElementResultBlock& from, elements::ElementResultBlock& to);

    const elements::ElementFamily* m_inner;
    std::span<const base::EntityId> m_replaced; // sorted
};

} // namespace oos::reduction
