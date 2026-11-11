#pragma once

// Bushing1D — rod-type spring and damper (CBUSH1D, BushProp1D):
//
//   | Bushing1D | EID | PID | GA | GB | CID |
//   | BushProp1D | PID | K   | C  | M  |     | SA | SE |
//
// Formulation (CBUSH1D, BushProp1D, the linear part): one axial spring K along the element x-axis between GA and GB,
// on their translations only (no rotational stiffness, like a Rod2 without torsion): e = x·(u_B − u_A), the axial
// displacement, f = K·e, the element force; K = K·hᵀ·h with h the row of x·(u_B − u_A) over the translations of GA
// and GB in their displacement systems. A blank GB grounds the second end. The x-axis runs from GA to GB, or is the
// x-axis of CID when CID is given (needed when GB is blank or GA and GB coincide: SpringAxes).
//
// Facets: stiffness; mass (PBUSH1D M, the mass of the element, lumped on the translations: M/2 at GA and at GB, M at GA
// when GB is blank — the split is assumed, not validated); recovery: the stress and strain tables of the data contract
// (stress.bushing1D, strain.bushing1D: element force, axial displacement, axial velocity 0 in statics, axial stress
// SA·f, axial strain SE·e, plastic strain 0, failed 0; BushProp1D: SA, SE the stress and strain recovery coefficients).
// C, the viscous damping, plays no part in SOL 101, 103, 105. The data contract has no force table of Bushing1D (the
// stress table holds the force).
// The stiffness, mass and recovery rates of K, M, SA and SE are closed form.
//
// Not supported, refused by check(): the nonlinear groups of BushProp1D (SHOCKA, SPRING, DAMPER, GENER: field groups
// not 0); a cylindrical or spherical CID. Problems: an unknown grid point, a PID that is not a BushProp1D, CID missing
// where it is needed, an unknown CID.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of Bushing1D. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class Bushing1DElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Bushing1D"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                 const DesignedField& first, const DesignedField& second,
                                 ElementOutput output, const NodalDisplacements& displacements,
                                 ElementResultBlock& rates) const override;
    bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementMatrixBlock& block) const override;
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, const NodalDisplacements& displacements,
                            ElementResultBlock& rates) const override;
    bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                        const DesignedField& field, ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;
    bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                            ElementOutput output, RecoverySlopes& rates) const override;
    bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                  ElementOutput output, const NodalDisplacements& displacements,
                                  ElementResultBlock& rates) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"BushProp1D"}; }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
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

    // One resolved Bushing1D: its grid rows, h (the axial displacement over the translations of GA, then GB), and the
    // BushProp1D constants.
    struct Bush
    {
        std::int32_t rowA = 0;
        std::int32_t rowB = 0;
        bool isGrounded = false;
        std::array<double, 6> axialMap{}; // h; entries 3-5 are zero when grounded
        double stiffness = 0.0;           // K
        double mass = 0.0;                // M
        double stressCoefficient = 0.0;   // SA
        double strainCoefficient = 0.0;   // SE
    };

    // The DOFs of h: the translations of GA, then those of GB unless grounded.
    [[nodiscard]] static std::span<const ElementDof> translationDofs(const Bush& bush, std::array<ElementDof, 6>& dofs);

private:
    struct Columns
    {
        std::span<const std::int32_t> ga;
        std::span<const std::int32_t> gb;
        std::span<const std::int32_t> frameId;
        PropertyRows properties;
        std::span<const double> k;
        std::span<const double> m;
        std::span<const double> sa;
        std::span<const double> se;
        std::span<const std::uint8_t> groups;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<Bush> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
};

} // namespace oos::elements
