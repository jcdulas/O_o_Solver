#pragma once

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementFormulation.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The family of a solid card (SolidTetra, SolidPyramid, SolidPenta, SolidHexa) in the mixed hybrid formulation
// (docs/mixed-hybrid-formulation-spec.md sections 3.3 and 4.3): the hybrid stress element of degree 4, its stresses
// in equilibrium with its interior bubbles and eliminated, its displacement shared with its neighbours at its grid
// points and on its edges and faces (and its interior fields in a modal analysis).
//
// The family is thin: it reads its card's fields through the model of the formulation (MixedHybridModel, which reads
// them as the card's other families do) and calls the SolidPart of the model, which keeps the kernels it builds. The
// element matrices are formed on the element's local unknowns (grid translations in the basic system, entity DOFs) and
// carried to the displacement systems CD of the grid points (LocalMap). A solid has no rotation at its grid points.
//
// Choices of this family (spec sections 7 and 10.3):
// - the temperatures are read as the card's other families read them, at the corners (the mid-side nodes apart), and
//   interpolated over the element by the functions of its corners at the point of the straight element nearest to the
//   point of the formulation's element; an expansion that differs between directions (AnisoMat3D A1-A6) is refused
//   under a temperature load (the kernel takes one coefficient);
// - a pressure along a direction (CID, N1-N3) is taken when it is the same at every corner of the face;
// - the stresses at the centre and the corners, in the material system CORDM (the rows of the card's other families);
//   the strains are those of the stresses and of the free expansion there.
//
// Thread-safety: immutable; facets may run concurrently on distinct ranges.
class MixedHybridSolidFamily final : public ElementFamily
{
public:
    // The family of the card of `classic` (its ClassicFormulation family, kept for what the two share: the property
    // cards, the field rules, the output columns).
    explicit MixedHybridSolidFamily(std::unique_ptr<ElementFamily> classic);

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
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    // The volume of the kernel of each solid (the sum of the weights of its rule, the Jacobian of the element in
    // them): the kernels are those the stiffness keeps.
    [[nodiscard]] bool computeVolumes(const ElementBatchView& view, ElementRange range,
                                      std::span<double> volumes) const override;

    void checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                            base::DiagnosticSink& sink) const override;
    void computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                              ElementVectorBlock& block) const override;
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }

    void checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
    bool differentialStiffnessPreloadGradient(const ElementBatchView& view, std::int64_t row,
                                              const NodalDisplacements& shape,
                                              ElementVectorBlock& block) const override;

    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override
    {
        return m_classic->outputColumns(output);
    }
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::string resultElementType(bool underLineLoads) const override
    {
        return m_classic->resultElementType(underLineLoads);
    }

private:
    std::unique_ptr<ElementFamily> m_classic;
};

} // namespace oos::elements::mixedHybrid
