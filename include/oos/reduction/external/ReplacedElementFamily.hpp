#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/core/ElementFamily.hpp"

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

    void check(const elements::ElementBatchView& view, base::DiagnosticSink& sink) const override;

    void computeStiffness(const elements::ElementBatchView& view, elements::ElementRange range,
                          elements::ElementMatrixBlock& block) const override;
    void computeMass(const elements::ElementBatchView& view, elements::ElementRange range,
                     const elements::MassOptions& options, elements::ElementMatrixBlock& block) const override;

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
    [[nodiscard]] elements::FieldUse designFieldUse(const elements::DesignedField& field) const override
    {
        return m_inner->designFieldUse(field);
    }

    // Whether element `elementId` is replaced.
    [[nodiscard]] bool isReplaced(base::EntityId elementId) const noexcept;

private:
    // The runs of rows of `range` that hold no replaced element, in order.
    [[nodiscard]] std::vector<elements::ElementRange> keptRuns(const elements::ElementBatchView& view,
                                                               elements::ElementRange range) const;
    // Appends the rows of `from` to `to` (the same columns).
    static void appendRows(const elements::ElementResultBlock& from, elements::ElementResultBlock& to);

    const elements::ElementFamily* m_inner;
    std::span<const base::EntityId> m_replaced; // sorted
};

} // namespace oos::reduction
