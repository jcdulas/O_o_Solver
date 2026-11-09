#pragma once

// SolidPenta — five-sided solid (a wedge) with 6 corners and up to 9 mid-side nodes (CPENTA, SolidProp):
//
//   | SolidPenta | EID | PID | G1  | G2  | G3  | G4  | G5  | G6  |
//   |        | G7  | G8  | G9  | G10 | G11 | G12 | G13 | G14 |
//   |        | G15 |
//
// G1-G3 one triangular face, G4-G6 the opposite one (G4 opposite G1); the mid-side nodes G7-G15 on the edges 1-2, 2-3,
// 3-1, 1-4, 2-5, 3-6, 4-5, 5-6, 6-4 (SolidShapeFunctions).
//
// Formulation: the isoparametric wedge (Zienkiewicz & Taylor vol. 1, the triangular prism; the same element as
// CalculiX C3D6 and C3D15, Code_Aster's PENTA6 and PENTA15), with the rule exact for the stiffness of a straight
// prism: 3 triangle points × 2 Gauss points for 6 nodes, a degree-4 triangle rule × 3 points for 15 nodes. SolidProp IN
// and ISOP: blank; the 6-node wedge also takes IN = BUBBLE, ISOP = REDUCED or FULL (the same element: it has one). The
// 6-node wedge, like the plain trilinear hexahedron, locks in bending (its parasitic shear): mesh bending with the
// default SolidHexa or with the 15-node wedge. The 15-node wedge is not validated against reference results. The
// material (MAT1 or AnisoMat3D) in the system CORDM (SolidSection).
//
// Mass: lumped (HRZ) or coupled (SolidElementKernel). Loads: FacePressureLoad on a triangular or quadrilateral face,
// uniform or with corner pressures (SolidPressure); GravityLoad through the mass. Recovery: stresses and strains at the
// centre and the six corners in the CORDM system (SolidRecovery), the rows of the result tables.
//
// Thermal loads, and what is refused: as SolidHexa (solidHexa/SolidHexaElement.hpp).

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/solids/SolidElementKernel.hpp"
#include "oos/elements/solids/SolidRecovery.hpp"

#include <array>
#include <span>

namespace oos::elements
{

// The element family of SolidPenta. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class SolidPentaElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "SolidPenta"; }
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

    // The formulation of one element (SolidPentaStiffness.cpp).
    static void buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response);

private:
    static constexpr SolidElementKernel KERNEL{{SolidShape::Penta6, SolidShape::Penta15}, &buildResponse};
};

} // namespace oos::elements
