#pragma once

// ShellTria3Drilling: 3-node shell with a normal rotational (drilling) degree of freedom.
// Its element family, section operators and recovery all use the ShellTria3Drilling root.
//
//   | ShellTria3Drilling | EID | PID | G1 | G2 | G3 | THETA or MCID | ZOFFS |
//   | TFLAG | corner thicknesses |
//
// Formulation (ShellTria3DrillingGeometry, ShellTria3DrillingPlate): the membrane with a drilling DOF (ShellTria3DrillingMembrane), the
// bending on linear rotations with the edge shear of the NASTRAN-95 TRIA3 (MID3 in its compliance); the sections
// (PSHELL with IsoMat, AnisoMat2D, OrthoMat2D, MID1 to MID3, MID2 = −1; CompProp, CompPropGlobal with no
// membrane-bending coupling), ZOFFS, the corner thicknesses and THETA/MCID as in ShellQuad4DrillingElement.hpp, the element x
// axis along G1→G2.
//
// Mass, pressure and thermal loads as in ShellQuad4DrillingElement (lumped HRZ or coupled with rotary inertia; F_i = ∫ N_i p dA
// along +n or the direction of CID; the thermal resultants of the mean temperature change). Recovery at the centre only
// (node 0; a triangle has no corner output): stresses and fibre strains at Z1 and Z2, strains and curvatures, forces
// per unit length, the plies of a laminate (ShellRecovery).
//
// Facets: stiffness, mass, differential stiffness (the membrane resultants of the preload, shells/ShellDifferential),
// loads (pressure, thermal), recovery (with the layers), design derivatives.
//
// Not supported, and problems reported: as in ShellQuad4DrillingElement (a membrane-bending coupling among them), with collinear
// grid points the degenerate geometry of a triangle.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/ShellNodeFrames.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"

namespace oos::elements
{

// The element family of ShellTria3Drilling. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class ShellTria3DrillingElement final : public ElementFamily
{
public:
    ShellTria3DrillingElement() = default;
    [[nodiscard]] std::string_view card() const noexcept override { return "ShellTria3Drilling"; }
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
        ShellTria3DrillingGeometry geometry;
        ShellSection section; // in the element frame
        std::array<double, 3> thickness{};
        std::array<std::int32_t, 3> gridIds{};
        ShellNodeFrames<3> frames;
        double materialAngle = 0.0; // the material x axis from the element x axis, radians
    };

private:
    // The batch's columns and those of the ShellProp table, read once per facet call.
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

    // The data of the element of a row; a membrane-bending coupling is a problem.
    [[nodiscard]] Resolution<Data> resolve(const ElementBatchView& view, const Columns& columns,
                                           std::int64_t row) const;
    // Whether the family forms a membrane-bending coupling: no (the drilling membrane refuses one), so the closed-form
    // rates refuse a MID4 term or a ply thickness that would make a coupling zero at the point non-zero
    // (shellSectionRate, laminateSectionRate).
    [[nodiscard]] static constexpr bool takesCoupling() noexcept { return false; }

};

} // namespace oos::elements
