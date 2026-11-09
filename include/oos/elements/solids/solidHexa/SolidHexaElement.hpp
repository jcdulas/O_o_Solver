#pragma once

// SolidHexa — six-sided solid with 8 corners and up to 12 mid-side nodes (CHEXA, SolidProp):
//
//   | SolidHexa | EID | PID | G1  | G2  | G3  | G4  | G5  | G6  |
//   |       | G7  | G8  | G9  | G10 | G11 | G12 | G13 | G14 |
//   |       | G15 | G16 | G17 | G18 | G19 | G20 |
//
// G1-G4 around one face, G5-G8 on the opposite face (G5 opposite G1), the mid-side nodes G9-G20 on the edges 1-2, 2-3,
// 3-4, 4-1, 1-5, 2-6, 3-7, 4-8, 5-6, 6-7, 7-8, 8-5 (SolidShapeFunctions).
//
// Formulation (SolidSection.hpp lists the SolidProp IN and ISOP values of each):
// - 8 nodes, by default: the trilinear hexahedron with the incompatible modes of Wilson et al. (1973) and Taylor's
//   correction (1976), condensed at element level, 2 × 2 × 2 Gauss points (solidHexa/SolidHexaIncompatibleModes.hpp; the
//   element CalculiX names C3D8I): free of shear and Poisson locking in the bending of rectangular elements;
// - 8 nodes, ISOP = FULL: the plain trilinear isoparametric element with 2 × 2 × 2 points (CalculiX C3D8), which locks
//   in bending;
// - 20 nodes: the serendipity element, by default with the full 3 × 3 × 3 rule (CalculiX C3D20), with ISOP = REDUCED
//   or IN = TWO the reduced 2 × 2 × 2 rule (CalculiX C3D20R; Zienkiewicz & Taylor vol. 1, reduced integration), its
//   stresses then extrapolated from the Gauss points.
// The material (MAT1 or AnisoMat3D) in the system CORDM (SolidSection).
//
// Mass: lumped (HRZ) or coupled (SolidElementKernel). Loads: FacePressureLoad on a face, uniform or with corner
// pressures (SolidPressure); GravityLoad through the mass (the load vector builder); the thermal loads of the grid
// point temperatures (TEMPERATURE(LOAD), IsoMat A or AnisoMat3D A1-A6 in the CORDM system; SolidElementKernel), whose
// thermal stresses the recovery subtracts (with the incompatible modes, their share of the thermal load). Recovery:
// stresses and strains at the centre and the eight corners in the CORDM system (SolidRecovery), the rows of the result
// tables; the incompatible modes' strain included (their corner values extrapolated from the Gauss points).
//
// Differential stiffness (SOL 105): the initial-stress stiffness of the centre stress of the preload
// (SolidElementKernel::computeDifferentialStiffness, the same for the four solids).
//
// Refused with a diagnostic (SolidConnectivity, SolidSection): some but not all mid-side nodes; CORDM −1 or −2 or a
// curvilinear CORDM; IN, ISOP other than above; STRESS = GAUSS; FCTN other than SMECH; materials other than IsoMat and
// AnisoMat3D; a degenerate or inverted shape; a FacePressureLoad with a direction.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/solids/SolidElementKernel.hpp"
#include "oos/elements/solids/SolidRecovery.hpp"

#include <array>
#include <span>

namespace oos::elements
{

// The element family of SolidHexa. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class SolidHexaElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "SolidHexa"; }
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

    // The formulation of one element (SolidHexaStiffness.cpp).
    static void buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response);

private:
    static constexpr SolidElementKernel KERNEL{{SolidShape::Hexa8, SolidShape::Hexa20}, &buildResponse};
};

} // namespace oos::elements
