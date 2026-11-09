#pragma once

// SolidPyramid — five-sided solid (a pyramid) with 5 corners and up to 8 mid-side nodes (CPYRAM, SolidProp):
//
//   | SolidPyramid | EID | PID | G1  | G2  | G3  | G4  | G5  | G6  |
//   |        | G7  | G8  | G9  | G10 | G11 | G12 | G13 |
//
// G1-G4 the quadrilateral base, G5 the apex; the mid-side nodes G6-G13 on the edges 1-2, 2-3, 3-4, 4-1, 1-5, 2-5, 3-5,
// 4-5 (SolidShapeFunctions).
//
// Formulation: the hexahedron (8 or 20 nodes) whose face opposite to the base is collapsed into the apex (Zienkiewicz
// & Taylor vol. 1 §8.9, degenerate elements), isoparametric with 2 × 2 × 2 (5 nodes) or 3 × 3 × 3 (13 nodes) Gauss
// points; the corner stresses are extrapolated from the 2 × 2 × 2 points (the apex has no Jacobian), the apex taking
// the mean of the four collapsed corners, and the centre is the centre of the parent hexahedron. Not validated against
// reference results: not_validated. SolidProp IN blank, ISOP blank or FULL. The material (MAT1 or AnisoMat3D) in the
// system CORDM (SolidSection).
//
// Mass: lumped (HRZ) or coupled (SolidElementKernel). Loads: FacePressureLoad on the base or a triangular face, uniform
// or with corner pressures (SolidPressure); GravityLoad through the mass. Recovery: stresses and strains at the centre
// and the five corners in the CORDM system (SolidRecovery), the rows of the result tables.
//
// Thermal loads, and what is refused: as SolidHexa (solidHexa/SolidHexaElement.hpp).

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/solids/SolidElementKernel.hpp"
#include "oos/elements/solids/SolidRecovery.hpp"

#include <array>
#include <span>

namespace oos::elements
{

// The element family of SolidPyramid. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class SolidPyramidElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "SolidPyramid"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override
    {
        return {"SolidProp", "IsoMat", "AnisoMat3D"};
    }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    // The forms of the columns and the recovery matrices from the operators of the formulation
    // (SolidElementKernel::recoverySlopesOfOutputs), and the vectorised recovery from
    // SOLID_VECTORISED_RECOVERY_MIN_CASES cases, its rows completed by completeSolidRows (SolidRecovery.hpp).
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override
    {
        return solidOutputColumnForms(output);
    }
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override
    {
        requireOutput(output);
        const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
        return KERNEL.recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
    }
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override
    {
        return KERNEL.recoverySlopesOfOutputs(view, row, outputs, slopes);
    }
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override
    {
        return outputColumns(output).empty() ? 0 : SOLID_VECTORISED_RECOVERY_MIN_CASES;
    }
    void completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override
    {
        requireOutput(output);
        completeSolidRows(output, blocks);
    }
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    void recoverOutputs(const ElementBatchView& view, ElementRange range, std::span<const RecoveryOutput> outputs,
                        const NodalDisplacements& displacements, std::span<ElementResultBlock> blocks) const override
    {
        for (const RecoveryOutput& output : outputs)
            if (!output.isLayer)
                requireOutput(output.output);
        KERNEL.recoverOutputs(view, range, outputs, displacements, blocks);
    }
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
    void checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                            base::DiagnosticSink& sink) const override;
    void computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                              ElementVectorBlock& block) const override;
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const override;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const override;

    // The formulation of one element (SolidPyramidStiffness.cpp).
    static void buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response);

private:
    static constexpr SolidElementKernel KERNEL{{SolidShape::Pyram5, SolidShape::Pyram13}, &buildResponse};
};

} // namespace oos::elements
