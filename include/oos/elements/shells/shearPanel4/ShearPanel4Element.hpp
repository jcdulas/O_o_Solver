#pragma once

// ShearPanel4 — shear panel (CSHEAR, ShearPanelProp):
//
//   | ShearPanel4 | EID | PID | G1 | G2 | G3 | G4 |
//   | ShearPanelProp | PID | MID | T  | NSM | F1 | F2 |
//
// Formulation: ShearPanel4Panel (the statically determinate shear flows of Garvey's quadrilateral panel, the energy
// of the mean flow, the extensional rods of F1 and F2). G, E and RHO from the IsoMat of MID; translations of the four
// grid points only (the panel has no rotational stiffness).
//
// Recovery (the layouts of the O_o_Reader data contract): stresses max_shear (the edge shear
// stress τ_e = q_e / T of largest magnitude), avg_shear (the mean of the four), margin (SS of the IsoMat against
// max_shear); strains the same divided by G; forces the corner forces F_ij = q_e L_e / 2 at the two ends of each
// edge, the kick forces (0 on a flat panel) and the shear flows q12, q23, q34, q41.
//
// Facets: stiffness, mass (lumped quarters, the coupled mass the same), recovery. No thermal load (a pure shear
// state; ElementFamily::isThermallyInert), no pressure. Differential stiffness: the mean shear flow of the preload as
// a membrane resultant Nxy (ShearPanel4Panel::differentialStiffness). Not written yet: derivatives.
//
// Not supported, and problems reported by check(): an unknown grid point, a grid point named twice, a PID that is not a
// ShearPanelProp, a material other than IsoMat, a T or G that is not positive, parallel diagonals, a folded corner, a
// warped panel (the kick forces are not formed; refused with its diagnostic), ModelParams SHEARP HARDER (not read: the
// default Garvey panel is formed).

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/shearPanel4/ShearPanel4Panel.hpp"

namespace oos::elements
{

// The element family of ShearPanel4. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class ShearPanel4Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "ShearPanel4"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Differential, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"ShearPanelProp", "IsoMat"}; }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The vectorised recovery: the recovery matrix from one resolution of the
    // panel (recoverySlopes), from SHEAR_PANEL_VECTORISED_RECOVERY_MIN_CASES cases; the largest shear and the margin
    // completed from the average shear (the mean flow of the panel).
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const override;

    // Everything the facets need about one panel.
    struct Data
    {
        ShearPanel4Panel panel;
        ShearPanelSection section;
        double shearLimit = 0.0; // SS of the IsoMat
        std::array<std::int32_t, 4> gridRows{};
        std::array<base::Mat3, 4> cdFrames{};

        // The translations of the four grid points (components 1-3), node by node.
        [[nodiscard]] std::array<ElementDof, 12> dofs() const;
        // A local matrix (element frame) in the displacement systems: K_CD(a, b) = R_aᵀ E K(a, b) Eᵀ R_b.
        [[nodiscard]] base::MatN<12> toDisplacementSystems(const base::MatN<12>& local) const;
        // The local translations q (element frame) of the displacements of the model.
        [[nodiscard]] std::array<double, 12> localDisplacements(const NodalDisplacements& displacements) const;
    };

private:
    struct Columns
    {
        ShellConnectivity connectivity;
        PropertyRows properties;
        IsotropicMaterialColumns materials;
        std::span<const std::int32_t> matId;
        std::span<const double> t, nonStructMass, f1, f2;
        base::ValidityView isNonStructMassGiven, isF1Given, isF2Given;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<Data> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
};

} // namespace oos::elements
