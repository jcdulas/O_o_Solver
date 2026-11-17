#pragma once

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementFormulation.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements::mixedHybrid
{

struct MixedLine;

// The family of a line card (Rod2, Rod2InlineSection, Tube2, Bar2, Beam2) in the mixed hybrid formulation
// (docs/mixed-hybrid-formulation-spec.md sections 3.2 and 4.2): the mixed hybrid line of the reference crate's
// document 34, section 8.1, its moments of degree k eliminated on its edge, its rotation Λ at each grid point the
// hybrid unknown (the rotation components of the grid point), its displacement between the grid points carried by
// the functions of its edge (an entity of the model).
//
// The family is thin: it reads its card's fields through the card's other family (the same property cards, field
// rules and output columns), looks its elements up in the model of the formulation (MixedHybridModel), and calls the
// line kernel (mixedHybrid/line). The element matrices are formed in the basic system for the grid components and
// carried to the displacement systems CD of the grid points.
//
// Facets: stiffness, mass (the consistent mass whatever CoupledMass asks), recovery (the force, stress and strain
// tables of the card: the forces of the section at the ends, and for a Beam2 the rows of its stations made from them
// as the card's other family makes them), the thermal loads (the free axial strain α ΔT + D/L
// of the card's other family, uniform along the line), the differential stiffness of the axial force (offsets
// included: the formulation carries them in its section law, spec section 9), the line loads of a Bar2 and a Beam2
// (BeamDistributedLoad: the work of the forces and couples per length on the displacement and rotation of the line at
// its axis, by Gauss points over the loaded part, exact for these polynomials; the rows under line loads made by the
// card's other family's response from the end forces of the formulation and the loads between the ends).
//
// Thread-safety: immutable; facets may run concurrently on distinct ranges.
class MixedHybridLineFamily final : public ElementFamily
{
public:
    // The family of the card of `classic` (its ClassicFormulation family, kept for what the two share).
    explicit MixedHybridLineFamily(std::unique_ptr<ElementFamily> classic);

    [[nodiscard]] std::string_view card() const noexcept override { return m_classic->card(); }
    [[nodiscard]] std::string_view formulation() const noexcept override { return MIXED_HYBRID_FORMULATION; }
    [[nodiscard]] FacetSet facets() const noexcept override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return m_classic->propertyCards(); }
    [[nodiscard]] std::vector<input::FieldRule> fieldRules() const override { return m_classic->fieldRules(); }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    // ∫ A μ ds along the axis of the line (μ the length of an offset axis over that of its edge): the momentum of a
    // unit translation of the line with a unit density and no non-structural mass. A line on an edge of shells moves
    // with the unknowns of that edge, which carry part of a rigid translation.
    [[nodiscard]] bool computeVolumes(const ElementBatchView& view, ElementRange range,
                                      std::span<double> volumes) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::string resultElementType(bool underLineLoads) const override
    {
        return m_classic->resultElementType(underLineLoads);
    }
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }
    void checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void checkLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                        base::DiagnosticSink& sink) const override
    {
        m_classic->checkLineLoads(view, loads, sink);
    }
    void computeLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                          ElementVectorBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumnsUnderLineLoads(ElementOutput output) const override
    {
        return m_classic->outputColumnsUnderLineLoads(output);
    }
    void recoverUnderLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                               ElementOutput output, const NodalDisplacements& displacements,
                               ElementResultBlock& block) const override;
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;

private:
    // The kind of output rows of the card.
    enum class Rows
    {
        Rod, // Rod2, Rod2InlineSection, Tube2: axial force and torque
        Bar, // Bar2: the forces and stresses at both ends
        Beam // Beam2: the rows of its stations
    };

    // The free axial strain α ΔT + D/L of a line under the temperatures of a subcase (0 without).
    [[nodiscard]] static double freeStrainOf(const ElementBatchView& view, std::int64_t row, const MixedLine& line,
                                             const ElementTemperatures* temperatures);

    std::unique_ptr<ElementFamily> m_classic;
    Rows m_rows;
};

} // namespace oos::elements::mixedHybrid
