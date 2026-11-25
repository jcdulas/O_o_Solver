#pragma once

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/mixedHybrid/model/ShellTraceDefect.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements::mixedHybrid
{

[[nodiscard]] ShellInteriorDefect shellDefects(const ElementBatchView& view, std::int64_t row,
                                               const NodalDisplacements& field, bool testInterior,
                                               std::vector<ShellEdgeDefect>& edges);

// The family of a shell card (ShellQuad4, ShellTria3, ShellQuad4Drilling, ShellTria3Drilling, ShellQuad8, ShellTria6)
// in the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md sections 3.1 and 4.1): the shell of
// Neunteufel and Schöberl in its TDNNS form, its displacement shared with its neighbours on its edges and at its
// corners, its shear and edge rotations on its edges, its moments eliminated in the element.
//
// The family is thin: it reads its card's fields through the model of the formulation (MixedHybridModel, which reads
// them as the card's other families do), and calls the ShellPart of the model. The element matrices are formed over
// the element's local unknowns (grid translations in the basic system, entity DOFs with their signs) and carried to
// the displacement systems CD of the grid points (LocalMap). A shell has no rotation at its grid points.
//
// Thread-safety: immutable; facets may run concurrently on distinct ranges.
class MixedHybridShellFamily final : public ElementFamily
{
public:
    // The family of the card of `classic` (its ClassicFormulation family, kept for what the two share: the property
    // cards, the field rules, the output columns).
    explicit MixedHybridShellFamily(std::unique_ptr<ElementFamily> classic);

    [[nodiscard]] std::string_view card() const noexcept override { return m_classic->card(); }
    [[nodiscard]] std::string_view formulation() const noexcept override { return MIXED_HYBRID_FORMULATION; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Loads, Facet::Differential, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return m_classic->propertyCards(); }
    [[nodiscard]] std::vector<input::FieldRule> fieldRules() const override { return m_classic->fieldRules(); }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                 ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    // ∫ t μ0 dA on the rule of the element: the thickness of the section over the surface of the element (μ0 the area
    // of the reference surface of an offset section over that of the grid points). The displacement of a shell is not
    // that of its grid points alone (its edges and its interior carry part of a rigid translation), so its lumped
    // mass at its grid points is not its mass.
    [[nodiscard]] bool computeVolumes(const ElementBatchView& view, ElementRange range,
                                      std::span<double> volumes) const override;

    // A pressure along the normal of the element surface (its corner values interpolated over the element), or a
    // traction along the direction of the load (CID, N1-N3), per unit area of the reference surface of the section.
    void checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                            base::DiagnosticSink& sink) const override;
    void computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                              ElementVectorBlock& block) const override;
    // The free strains of the temperatures read as the card's other families read them (the change at the reference
    // surface, uniform or at the corners, and the gradient through the thickness).
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }
    // The second variation of the shells under the forces of the preload (the offsets in the section law: nothing to
    // refuse under OffsetDefinition ElemOffsetsOnly, spec section 9).
    void checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
    bool differentialStiffnessPreloadGradient(const ElementBatchView& view, std::int64_t row,
                                              const NodalDisplacements& shape,
                                              ElementVectorBlock& block) const override;

    // The rows of the card's other families (shells/ShellRecovery: forces, stresses at Z1 and Z2, strains and
    // curvatures, the plies of a laminate) at the centre and at the corners, in their output frame (x along the side
    // G1-G2 projected on the tangent plane), from the state of the element there (spec section 10.2).
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override
    {
        return m_classic->outputColumns(output);
    }
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::vector<std::string> layerOutputColumns(LayerOutput output) const override
    {
        return m_classic->layerOutputColumns(output);
    }
    void recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                       const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::string resultElementType(bool underLineLoads) const override
    {
        return m_classic->resultElementType(underLineLoads);
    }
    [[nodiscard]] std::string layerResultElementType() const override { return m_classic->layerResultElementType(); }

private:
    std::unique_ptr<ElementFamily> m_classic;
};

} // namespace oos::elements::mixedHybrid
