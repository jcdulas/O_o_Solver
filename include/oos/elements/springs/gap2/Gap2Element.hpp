#pragma once

// Gap2 — gap element, in a linear analysis (CGAP, GapProp):
//
//   | Gap2 | EID | PID | GA | GB | X1/GO | X2  | X3  | CID   |
//   | GapProp | PID | U0  | F0 | KA | KB    | KT  | MU1 | MU2   |
//   |      | TMAX| MAR | TRMIN |
//
// Formulation: in a linear solution sequence the gap is a linear spring in the state the initial opening
// U0 gives it (PGAP: the gap is closed when U0 ≤ 0, open when U0 > 0; the nonlinear contact is for SOL 106 and
// 400): closed, the axial stiffness KA and the transverse stiffness KT (O_o_Reader gives KT = MU1·KA when blank) in the
// element y and z directions; open, the axial stiffness KB and no transverse stiffness. The springs act on the
// translations of GA and GB only (the Gap2 has no rotational stiffness), on the relative displacement
// Δ = Eᵀ·(u_B − u_A) in the element system E: K = Hᵀ·diag(k_x, k_t, k_t)·H. The element axes are those of CID when CID
// is given (needed when GA and GB coincide), else x from GA to GB and y, z from the orientation vector GO or X1-X3
// (SpringAxes). A transverse stiffness between distinct grid points is not in moment equilibrium (coincident grid
// points with CID avoid this).
//
// Facets: stiffness; recovery of the forces in the layout of the data contract (forces.gap2: fx the axial force,
// positive in compression, −k_x·Δx; sfy, sfz the transverse forces k_t·Δy, k_t·Δz; u, v, w the relative displacements
// Δx, Δy, Δz; sv, sw the slips, 0 in a linear analysis). The signs of the gap output are assumed (no reference
// result of a gap in a linear solution was available): not validated.
//
// Not supported: a preload F0 other than 0 (support catalog field rule: in a linear analysis the preload would be a
// load, whose linear treatment is not defined here); a cylindrical or spherical CID. Problems: an unknown grid
// point or GO, a PID that is not a GapProp, CID missing for coincident grid points, an orientation parallel to GA-GB.

#include "oos/base/math/MatN.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of Gap2. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class Gap2Element final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Gap2"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                 const DesignedField& first, const DesignedField& second,
                                 ElementOutput output, const NodalDisplacements& displacements,
                                 ElementResultBlock& rates) const override;
    bool stiffnessFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                       const DesignedField& first, const DesignedField& second,
                                       ElementMatrixBlock& block) const override;
    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, ElementMatrixBlock& block) const override;
    bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, RecoverySlopes& rates) const override;
    bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementOutput output, const NodalDisplacements& displacements,
                                  ElementResultBlock& rates) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"GapProp"}; }
    [[nodiscard]] std::vector<input::FieldRule> fieldRules() const override;

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The vectorised recovery (LinearRowRecovery.hpp): every column Linear, the slopes in closed form.
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;

    // One resolved Gap2: its grid rows, H (the relative displacement in the element system over the translations of
    // GA, then GB) and the stiffnesses of its initial state.
    struct Gap
    {
        std::int32_t rowA = 0;
        std::int32_t rowB = 0;
        base::MatN<3, 6> relativeMotion;   // H
        std::array<double, 3> stiffness{}; // k_x, k_t, k_t
        bool isClosed = false;
    };

private:
    struct Columns
    {
        std::span<const std::int32_t> ga;
        std::span<const std::int32_t> gb;
        std::span<const std::int32_t> go;
        base::FixedListView<double> x;
        base::ValidityView isXGiven;
        std::span<const std::int32_t> frameId;
        PropertyRows properties;
        std::span<const double> u0;
        std::span<const double> ka;
        std::span<const double> kb;
        std::span<const double> kt;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<Gap> resolve(const ElementBatchView& view, const Columns& columns,
                                                 std::int64_t row);
};

} // namespace oos::elements
