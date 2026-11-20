#pragma once

// The element families of the ShellQuad4 and ShellTria3 (shellQuad4/ShellQuad4Element,
// shellTria3/ShellTria3Element): the facets the solver calls, written once over the kernel of each element (ShellQuad4,
// ShellTria3: "Shell" below, which gives NODES, DOFS, CARD, its Geometry and the functions stiffness, mass, recovery,
// pressureLoads, thermalLoads, differentialStiffness). The kernels are the elements of
// docs/shell-quad4-formulation.md and docs/shell-tria3-formulation.md.
//
// What the family adds to the kernels:
// - the inputs: grid points and their displacement systems, the section (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D;
//   CompProp and CompPropGlobal as their equivalent section: PlateConstitutive.hpp), the corner thicknesses (T1-T4,
//   TFLAG), the material axis (THETA from the reference direction G1→G2, or the projection of MCID's x axis), the
//   offset ZOFFS;
// - the passage between the element frame and the displacement systems of the grid points, with the offset
//   (ShellNodeFrames: u_reference = u + θ × (ZOFFS n); section 11 of the ShellQuad4 document);
// - the output rows (ShellRecovery's tables), in the reference frame (x along G1→G2): those of the centre, and for
//   an output asked with its corners (RecoveryOutput::hasCorners: the CornerResults describer of a request) those of
//   the four grid points of a ShellQuad4 after them (section 12 of its document: the values of the 2 × 2 points read
//   at the corners); a ShellTria3 has its centre rows only, and the plies of a laminate are those of the centre;
// - the design derivatives in closed form, by the complex step (ShellScalar.hpp) through the same kernels.
//
// Choices of this project:
// - mass: the lumped mass of the element on the grid points through the rotations alone, and the pressure loads
//   without the offset, unless ModelParams OffsetDefinition asks for them on the offset (ShellNodeFrames), as the other
//   shells; the thermal loads always through the offset, consistently with the stiffness;
// - thermal loads: the dilatation of the free strain (its mean in-plane strain) loads the element as K q, q its
//   homothety, so that a uniform temperature rise leaves a warped mesh, or one whose normals are shared, without
//   stress (DirectorShellKinematics.hpp, addDilatationLoads);
// - THETA and MCID: the material axis is measured and applied in one frame, the element frame;
// - ShellProp T without MID1: the bending-only section keeps its thickness;
// - a section without MID3 (or a laminate whose plies give no transverse shear modulus) has no transverse shear
//   flexibility: the kernels take the shear stiffness of section 4 of the ShellQuad4 document;
// - the recovery: the forces and moments of a coupled section (MID4, an unsymmetric laminate) take the coupling, and
//   the thermal strain and curvature are subtracted, so that the forces balance the loads; the fibre stresses are
//   those of a plate of bending material G2 (without 12I/T³); the transverse shear forces are those conjugate to
//   the mean shear strain in the energy of the element;
// - the differential stiffness (SOL 105): the membrane forces recovered by the element (ShellCentreRecovery, the
//   thermal force removed) act on the in-plane gradients of u, v, w (ShellDifferential),
//   ∫ (Nx w,x² + 2 Nxy w,x w,y + Ny w,y²) dA and the same terms on u and v;
// - the normals of the grid points: the directors of the elements are the normals their grid points share with
//   the neighbouring ShellQuad4, ShellTria3, ShellQuad8 and ShellTria6 within PARAM ShellNormalAngle (20 degrees by
//   default), in place of each element's own (ShellDirectors.hpp); a flat mesh, a fold and a junction keep the
//   elements' own normals;
// - the rotation about the normal: a section with a membrane and a bending stiffness has the drilling term of the
//   ShellQuad4 document (section 8) with c_d = 1e-6 times PARAM ShellDrillingStiffness (100 by default); a section
//   without one of them, or a model with ShellDrillingStiffness 0, has none (the solver's automatic constraint of
//   singular DOFs, PARAM AutoSingularityConstr, removes what has no stiffness). The term is a stabilisation: an
//   inertia or a moment about the normal put on one grid point of a shell is held by it alone, and belongs on a rigid
//   or interpolation element that spreads it over the neighbouring grid points (a modal analysis points out the
//   PointMassInertia in that case: solutions::AnalysisModel).
//
// Design derivatives: the kernels are templates on the scalar and are evaluated in complex numbers along the rates of
// their inputs (the section values and the corner thicknesses), which gives the exact first derivatives (complex step,
// h = 1e-30). The second thickness rates use SecondOrderScalar to propagate the chain rule exactly. Fields
// with closed forms: the ShellProp T when every corner follows it, an AnisoMat2D stiffness term of the ShellProp's
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
    // The energy of the drilling term of each element (Shell::drillingEnergy).
    [[nodiscard]] bool computeStabilisationEnergies(const ElementBatchView& view, ElementRange range,
                                                    const NodalDisplacements& displacements,
                                                    std::span<double> energies) const override;

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
    bool differentialStiffnessFieldMixedDerivative(
        const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
        const NodalDisplacements& preload, ElementMatrixBlock& block) const override;
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
