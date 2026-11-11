#pragma once

// ShellQuad4Drilling: 4-node shell with a normal rotational (drilling) degree of freedom.
// Its element family, section operators and recovery all use the ShellQuad4Drilling root.
//
//   | ShellQuad4Drilling | EID | PID | G1 | G2 | G3 | G4 | THETA or MCID | ZOFFS |
//   | TFLAG | corner thicknesses |
//
// Formulation: ShellQuad4DrillingGeometry (element frame, warp, A and B), ShellQuad4DrillingMembrane (the membrane with a drilling
// DOF), ShellQuad4DrillingBending (Hellinger-Reissner curvatures with the Pian-Sumihara moments, NASTRAN-95 edge shear),
// ShellQuad4DrillingWarpOffsets (the rigid offsets that join a warped element, formed flat in its mean plane, to its grid
// points) and ShellQuad4DrillingPlate (the sum, the mass, the loads and the recovery). The section is a ShellProp with IsoMat,
// AnisoMat2D or OrthoMat2D materials (MID1 to MID3, MID2 = −1), or a laminate CompProp or CompPropGlobal with no
// membrane-bending coupling (ShellSection, Laminate). ZOFFS is a rigid offset of the reference plane; the corner
// thicknesses T1-T4 follow TFLAG (not for a laminate); THETA and MCID set the material axis, in which the material
// matrices and the ply angles are given.
//
// Mass (checked on public modal decks): lumped (COUPMASS ≤ 0, the default) the row sums of the consistent
// translational mass, ∫ m N_i dA, no rotary inertia; coupled (COUPMASS > 0) the consistent mass of the bilinear
// functions on the three translations and the rotary inertia ρ t³/12 on θx, θy. The mass per unit area is
// ρ t(ξ, η) + NSM (+ the NSM cards' share), ρ of MID1 (of MID2 when MID1 is blank; Σ ρ_k t_k of the plies for a
// laminate); with ZOFFS the mass lies on the offset plane.
//
// Pressure (PLOAD4, ShellPressureLoad through pressureTable): p(ξ, η) = Σ N_i p_i from the corner pressures, the
// consistent nodal forces F_i = ∫ N_i p dA on the mean plane, along +n, or along the vector N of CID (ShellPressure).
// SORL and LDIR (edge loads) are not in pressureTable.
//
// Thermal loads (TEMPERATURE(LOAD); ShellQuad4DrillingPlate): the temperature change of ShellTemperature, uniform over the
// element, through the thermal resultants of the section.
//
// Recovery (in the element frame): at the centre (node 0) and at the corners G1 to G4 (their grid IDs), the corner
// values extrapolated bilinearly from the 2 × 2 Gauss points (STRESS(CORNER) = BILIN); stresses and fibre strains
// at Z1 and Z2 (layers 1 and 2), strains and curvatures (STRCUR), forces per unit length (ShellRecovery). The centre
// rows alone are the STRESS(CENTER) output. A laminate gives the ply stresses, strains and failure indices at its
// centre (recoverLayers, element type "ShellQuad4DrillingComposite") and, by PARAM NOCOMPS, no homogeneous stresses or strains.
//
// Differential stiffness (SOL 105; ShellQuad4DrillingPlate::differentialStiffness): the membrane resultants of the preload at
// the centre on the in-plane gradients of the three translations (shells/ShellDifferential).
//
// Facets: stiffness, mass, differential stiffness, loads (pressure, thermal), recovery (with the layers), design
// derivatives.
//
// Not supported (refused by check() with a diagnostic naming the field): a membrane-bending coupling (MID4, a laminate
// with B ≠ 0), a material other than IsoMat, AnisoMat2D, OrthoMat2D, LAM = SMCORE and the Hashin failure theories of a
// laminate (ShellSection, Laminate); a cylindrical or spherical MCID. Refused by checkPressureLoads(): a direction in a
// curvilinear system or of zero length. Problems reported by check(): an unknown grid point, a grid point named twice,
// a PID that is not a ShellProp, CompProp or CompPropGlobal, an unknown material, parallel diagonals, a folded or
// collapsed corner, a shape whose edge shear is undefined, a thickness that is not positive, an MCID whose x axis is
// normal to the element. Warning: a warp above 0.05.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/ShellNodeFrames.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"

namespace oos::elements
{

// The element family of ShellQuad4Drilling. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class ShellQuad4DrillingElement final : public ElementFamily
{
public:
    ShellQuad4DrillingElement() = default;
    [[nodiscard]] std::string_view card() const noexcept override { return "ShellQuad4Drilling"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override
    {
        return {"ShellProp", "CompProp", "CompPropGlobal", "IsoMat", "AnisoMat2D", "OrthoMat2D"};
    }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    [[nodiscard]] bool isCheckedByRow() const noexcept override { return true; }
    void checkRows(const ElementBatchView& view, ElementRange range, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::vector<std::string> layerOutputColumns(LayerOutput output) const override;
    void recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                       const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The rows of several outputs from one formulation of each element (the rows at rest of a block of cases).
    void recoverOutputs(const ElementBatchView& view, ElementRange range, std::span<const RecoveryOutput> outputs,
                        const NodalDisplacements& displacements, std::span<ElementResultBlock> blocks) const override;
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    [[nodiscard]] std::vector<ColumnForm> layerOutputColumnForms(LayerOutput output) const override;
    // Its rows at unit displacements from one plate (output::SectionLoads).
    bool recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                    std::vector<ElementDof>& dofs,
                                    std::vector<ElementResultBlock>& blocks) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                             RecoverySlopes& slopes) const override;
    // The vectorised recovery: the rows completed by completeShellRows, from
    // SHELL_VECTORISED_RECOVERY_MIN_CASES cases; the ply tables by completeLaminateRows and the failure indices
    // from the ply states, from LAMINATE_VECTORISED_RECOVERY_MIN_CASES (ShellRecovery.hpp).
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    [[nodiscard]] std::size_t vectorisedLayerRecoveryMinCases(LayerOutput output) const override
    {
        return laminateVectorisedRecoveryMinCases(output);
    }
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;
    [[nodiscard]] LayerOutput vectorisedLayerSource(LayerOutput output) const override
    {
        return laminateVectorisedSource(output);
    }
    void completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                    std::span<const ElementResultBlock> sources,
                                    std::span<ElementResultBlock> blocks) const override;
    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementMatrixBlock& block) const override;
    void stiffnessFieldDerivatives(const ElementBatchView& view, std::int64_t row,
                                   std::span<const DesignedField> fields, std::span<ElementMatrixBlock> blocks,
                                   std::span<std::uint8_t> isClosed) const override;
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, const NodalDisplacements& preload,
                                              ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldMixedDerivative(
        const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
        const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    void recoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                    std::span<const DesignedField> fields, ElementOutput output,
                                    const NodalDisplacements& displacements, std::span<ElementResultBlock> rates,
                                    std::span<std::uint8_t> isClosed) const override;
    bool layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                 LayerOutput output, const NodalDisplacements& displacements,
                                 ElementResultBlock& rates) const override;
    void layerRecoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                         std::span<const DesignedField> fields, LayerOutput output,
                                         const NodalDisplacements& displacements, std::span<ElementResultBlock> rates,
                                         std::span<std::uint8_t> isClosed) const override;
    bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                        ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;
    bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, RecoverySlopes& rates) const override;
    bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementOutput output, const NodalDisplacements& displacements,
                                  ElementResultBlock& rates) const override;
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

    // Everything the facets need about one element.
    struct Data
    {
        ShellQuad4DrillingGeometry geometry;
        ShellSection section; // in the element frame
        std::array<double, 4> thickness{};
        std::array<std::int32_t, 4> gridIds{};
        ShellNodeFrames<4> frames;
        double materialAngle = 0.0; // the material x axis from the element x axis, radians
    };

private:
    // The batch's columns and those of the property tables, read once per facet call.
    struct Columns
    {
        ShellConnectivity connectivity;
        ShellSectionColumns sections;
        std::span<const double> theta;
        std::span<const std::int32_t> matFrameId;
        std::span<const double> offset;
        std::span<const std::int32_t> thicknessFlag;
        base::FixedListView<double> cornerThickness;

        explicit Columns(const ElementBatchView& view);
    };

    // The data of the element of a row; a membrane-bending coupling is a problem (a warped element is not:
    // ShellQuad4DrillingPlate forms it flat with the warp offsets).
    [[nodiscard]] Resolution<Data> resolve(const ElementBatchView& view, const Columns& columns,
                                           std::int64_t row) const;
    // Whether the family forms a membrane-bending coupling: no (the drilling membrane refuses one), so the closed-form
    // rates refuse a MID4 term or a ply thickness that would make a coupling zero at the point non-zero
    // (shellSectionRate, laminateSectionRate).
    [[nodiscard]] static constexpr bool takesCoupling() noexcept { return false; }

};

} // namespace oos::elements
