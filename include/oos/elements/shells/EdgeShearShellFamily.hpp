#pragma once

// The element families of the NASTRAN-95 ShellQuad4 and ShellTria3 (shellQuad4/ShellQuad4Element,
// shellTria3/ShellTria3Element): the facets the solver calls, written once over the kernel of each element (ShellQuad4,
// ShellTria3: "Shell" below, which gives NODES, DOFS, CARD, its Geometry and the functions stiffness, mass, recovery,
// pressureLoads, thermalLoads, differentialStiffness).
//
// What the family adds to the kernels:
// - the inputs: grid points and their displacement systems, the section (PSHELL with IsoMat, AnisoMat2D, OrthoMat2D;
// CompProp and CompPropGlobal
//   as their equivalent section: PlateConstitutive.hpp), the corner thicknesses (T1-T4, TFLAG), the material axis
//   (THETA from the user x axis, or the projection of MCID's x axis), the offset ZOFFS;
// - the passage between the element frame and the displacement systems of the grid points, with the offset
//   (ShellNodeFrames: u_reference = u + θ × (ZOFFS n), spec §2.11 and §3.8);
// - the output rows (ShellRecovery's tables): at the centre only, in the user frame (x along G1→G2, spec §7.1); a
//   STRESS(CORNER) request gets the centre rows, NASTRAN-95 having no corner output for these elements;
// - the design derivatives in closed form, by the complex step (ShellScalar.hpp) through the same kernels.
//
// Choices of this project (NASTRAN-95 differs or has no equivalent):
// - mass: the lumped mass of the element on the grid points through the rotations alone, and the pressure loads
//   without the offset, unless ModelParams OFFDEF asks for them on the offset (ShellNodeFrames), as the other shells;
//   the thermal loads always through the offset, consistently with the stiffness (NASTRAN-95's QUAD4 leaves the offset
//   out of its thermal loads, its TRIA3 does not);
// - THETA and MCID: the material axis is measured and applied in one frame. NASTRAN-95's QUAD4 measures an MCID's
//   angle in its element (bisector) frame but applies it from the user frame (G1→G2), which turns the material of a
//   skewed element by the angle between the two; a THETA of exactly 0.0 on the element would take the ShellProp's MCSID
//   in NASTRAN-95, a field this solver's ShellProp does not have;
// - ShellProp T without MID1: the bending-only ShellQuad4 keeps its thickness (NASTRAN-95 forces 1.0E-14 there);
// - the recovery of a coupled section (MID4, an unsymmetric laminate): the ShellQuad4's membrane forces take the
// coupling
//   t² G4 κ (ShellQuad4Recovery.cpp), and the ShellTria3's recovery reads G4 correctly and subtracts the thermal
//   curvature, so that the forces balance the loads; the other recovery choices of NASTRAN-95's QUAD4 are kept (its
//   shear factors of the recovery, G2 without 12I/T³ in the stresses);
// - the differential stiffness (SOL 105): NASTRAN-95 forms it with the sub-triangles of its older QUAD1/TRIA1
//   elements; here the membrane forces recovered by the element (ShellCentreRecovery, the thermal force removed) act on
//   the in-plane gradients of u, v, w (ShellDifferential), ∫ (Nx w,x² + 2 Nxy w,x w,y + Ny w,y²) dA and the same terms
//   on u and v;
// - no K6ROT: these elements have no drilling stiffness (the solver's automatic constraint of singular DOFs, PARAM
//   AUTOSPC, removes θz of a flat mesh).
//
// Design derivatives: the kernels are templates on the scalar and are evaluated in complex numbers along the rates of
// their inputs (the section values and the corner thicknesses), which gives the exact first derivatives (complex step,
// h = 1e-30); the second derivatives with respect to the ShellProp T are central differences of complex-step first
// derivatives (a 4-point rule in T with step 1e-3·T: truncation about 1e-12 relative, round-off about 1e-13). Fields
// with closed forms: the ShellProp T when every corner follows it, a AnisoMat2D stiffness term of the ShellProp's
// materials, a ply thickness of a CompProp or CompPropGlobal (forces and ply rows only); none under a temperature
// change. The others are differenced by the sensitivity analysis on the facets.
//
// Thread-safety: immutable; facets may run concurrently on distinct ranges.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/PlateConstitutive.hpp"
#include "oos/elements/shells/ShellCentreRecovery.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/ShellNodeFrames.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements
{

template <typename Shell>
class EdgeShearShellFamily : public ElementFamily
{
public:
    static constexpr std::size_t NODES = Shell::NODES;
    static constexpr std::size_t DOFS = Shell::DOFS;

    [[nodiscard]] std::string_view card() const noexcept override { return Shell::CARD; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Loads, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] std::vector<std::string> propertyCards() const override
    {
        return {"ShellProp", "CompProp", "CompPropGlobal", "IsoMat", "AnisoMat2D", "OrthoMat2D"};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    [[nodiscard]] bool isCheckedByRow() const noexcept override { return true; }
    void checkRows(const ElementBatchView& view, ElementRange range, base::DiagnosticSink& sink) const override;

    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
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
    [[nodiscard]] bool isThermallyInert() const noexcept override { return false; }

    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    [[nodiscard]] std::vector<std::string> layerOutputColumns(LayerOutput output) const override;
    void recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                       const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    void recoverOutputs(const ElementBatchView& view, ElementRange range, std::span<const RecoveryOutput> outputs,
                        const NodalDisplacements& displacements, std::span<ElementResultBlock> blocks) const override;
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    [[nodiscard]] std::vector<ColumnForm> layerOutputColumnForms(LayerOutput output) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                             RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    [[nodiscard]] std::size_t vectorisedLayerRecoveryMinCases(LayerOutput output) const override
    {
        return laminateVectorisedRecoveryMinCases(output);
    }
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
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, const NodalDisplacements& preload,
                                              ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    bool layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                 LayerOutput output, const NodalDisplacements& displacements,
                                 ElementResultBlock& rates) const override;
    bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                        ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;
    bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, RecoverySlopes& rates) const override;
    bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementOutput output, const NodalDisplacements& displacements,
                                  ElementResultBlock& rates) const override;

    // Everything the facets need about one element.
    struct Data
    {
        typename Shell::Geometry geometry;
        ShellSection section;                   // in its material frame (not rotated)
        PlateSectionValues<double> values;      // the section as the kernel reads it (material frame)
        PlateConstitutive<double> constitutive; // in the element frame
        std::array<double, NODES> thickness{};
        std::array<std::int32_t, NODES> gridIds{};
        ShellNodeFrames<NODES> frames;
        double materialAngle = 0.0;    // the material x axis from the element x axis, radians
        bool followsThickness = false; // every corner thickness proportional to the ShellProp T
    };

    // The rates of an element's inputs along one designed field (a direction of the complex step).
    struct Direction
    {
        PlateSectionValues<double> section;
        std::array<double, NODES> thickness{};
        std::optional<LaminateSectionRate> laminate; // a ply thickness
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

    [[nodiscard]] Resolution<Data> resolve(const ElementBatchView& view, const Columns& columns,
                                           std::int64_t row) const;
    [[nodiscard]] Data resolved(const ElementBatchView& view, std::int64_t row) const;
    // The direction of a designed field with a closed form for this element, or nothing.
    [[nodiscard]] std::optional<Direction> directionOf(const Data& data, const DesignedField& field) const;
};

} // namespace oos::elements
