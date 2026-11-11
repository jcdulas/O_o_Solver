#pragma once

// Bushing — generalised spring and damper (CBUSH, BushProp):
//
//   | Bushing | EID | PID  | GA | GB | GO/X1 | X2 | X3 | CID |
//   |       | S   | OCID | S1 | S2 | S3    |    |    |     |
//   | BushProp | PID | "K"  | K1 | K2 | K3    | K4 | K5 | K6  |
//
// Formulation (CBUSH, BushProp): six uncoupled springs K1-K6 on the
// relative motion of GB and GA, in the element system, acting at the spring point p:
//
//   Δu = (u_B + θ_B × (p − x_B)) − (u_A + θ_A × (p − x_A)),   Δθ = θ_B − θ_A,
//   f = diag(K1..K6)·Eᵀ(Δu, Δθ),                              K = Hᵀ·diag(K1..K6)·H,
//
// E the element axes in the basic system and H the map from the grid point DOFs (in their displacement systems) to
// the element components of (Δu, Δθ): rigid arms carry each grid point's motion to p, so a rigid motion gives no force.
// A blank GB grounds the second end (u_B = 0).
//
// Element system:
// - CID ≥ 0: the axes of CID, whatever the geometry;
// - CID blank: x from GA to GB; the orientation vector v from GO (v = x_GO − x_GA) or from X1-X3 (components in the
//   displacement system of GA); z = x × v / |x × v|, y = z × x (the Bar2 convention). CID must be given when GB is
//   blank or when GA and GB are closer than BUSH_COINCIDENT_DISTANCE.
// Spring point: OCID −1 (blank) puts it on the line GA-GB at p = x_A + S·(x_B − x_A) (S default 0.5; x_A itself when
// GB is blank); OCID ≥ 0 puts it at (S1, S2, S3) in the system OCID.
//
// Facets: stiffness; mass (PBUSH M, lumped on the translations: M/2 at GA and at GB, all of M at GA when GB is blank
// — the input format gives M as the mass of the element without the split, which is assumed here as for Bushing1D; not
// validated); recovery: forces fx, fy, fz, mx, my, mz in the element system (f above), stresses and strains by the
// recovery coefficients of BushProp RCV (PBUSH: SA, ST the stress coefficients of the translational and rotational
// components, EA, ET the strain coefficients; stress_i = SA·f_i for i = 1-3, ST·m_i for 4-6; strain_i = EA·Δu_i,
// ET·Δθ_i, the element displacements; their default 1.0 is given by O_o_Reader). B and GE (damping) play no part in
// SOL 101, 103, 105. Stiffness, mass and recovery have exact field derivatives in K1-K6, M and RCV.
//
// Not supported: a cylindrical or spherical CID or OCID (the element axes are then those of the curvilinear system
// at GA, which this element does not compute); CID blank with neither GO nor X1-X3 — refused by check(). Problems
// reported by check(): an unknown grid point, a PID that is not a BushProp, an unknown CID or OCID, an orientation
// vector parallel to GA-GB, CID missing where the input format requires it.

#include "oos/base/math/MatN.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// With CID blank, GA and GB must be at least this far apart (model length units) for the GA-GB line
// to define the element x-axis; closer, CID must be given.
inline constexpr double BUSH_COINCIDENT_DISTANCE = 1.0e-4;

// The orientation vector is refused as parallel to the axis when |x × v| ≤ this factor times |v|: the element y and z
// axes are then undefined (as for Bar2, such an orientation is refused).
inline constexpr double BUSH_PARALLEL_ORIENTATION_FACTOR = 1.0e-12;

// The element family of Bushing. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class BushingElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Bushing"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Recovery, Facet::Derivatives};
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
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"BushProp"}; }

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

    // One resolved Bushing: its grid point rows, the map H from the DOFs (GA 1-6, then GB 1-6 unless grounded) to the
    // element components of (Δu, Δθ), and K1-K6. Public for the tests of the element frame.
    struct Bush
    {
        std::int32_t rowA = 0;
        std::int32_t rowB = 0;
        bool isGrounded = false;
        base::MatN<6, 12> relativeMotion; // H; columns 6-11 are zero when grounded
        std::array<double, 6> stiffness{};
        double mass = 0.0;                // BushProp M
        std::array<double, 4> recovery{}; // BushProp RCV: SA, ST, EA, ET
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
        std::span<const double> s;
        std::span<const std::int32_t> offsetFrameId;
        base::FixedListView<double> si;
        PropertyRows properties;
        base::FixedListView<double> k;   // BushProp K1-K6 (empty without a BushProp table)
        base::FixedListView<double> rcv; // BushProp SA, ST, EA, ET
        std::span<const double> mass;    // BushProp M
        base::FixedListView<double>
            thermal; // BushProp "T": ALPHA, TREF, COINL (empty when the table has no such column)

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<Bush> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
};

} // namespace oos::elements
