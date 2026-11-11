#pragma once

// SpringDofInline — scalar spring with its stiffness on the card:
//
//   | SpringDofInline | EID | K | G1 | C1 | G2 | C2 | GE | S |
//
// Formulation: a spring of stiffness K between the DOF (G1, C1) and the DOF (G2, C2), each in the displacement system
// of its grid point (or the DOF of a scalar point, C blank); a blank G grounds that end. Element force f = K (u1 − u2),
// element stress S · f (CELAS2 fields K, S).
//
// Facets: stiffness, recovery (forces, stresses). No mass (a scalar spring has none). GE, the structural damping
// coefficient, is read by dynamic solutions only and plays no part in SOL 101, 103, 105. The
// differential stiffness and loads are zero. The derivatives of K and S are closed form.
//
// Not supported: nothing on this card. Problems reported by check(): an unknown point, a component outside 1-6 on a
// grid point or other than 0 on a scalar point, both ends blank, both ends on the same DOF.
//
// This is the reference element for the elements without geometry: copy its files to add one.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ScalarConnection.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of SpringDofInline: one batch holds every SpringDofInline of a model.
//
// Thread-safety: immutable; facets may run concurrently on distinct ranges.
class SpringDofInlineElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "SpringDofInline"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                 const DesignedField& first, const DesignedField& second,
                                 ElementOutput output, const NodalDisplacements& displacements,
                                 ElementResultBlock& rates) const override;
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

private:
    // The columns of the batch, read once per facet call (each accessor checks the column's type).
    struct Columns
    {
        std::span<const double> k;
        std::span<const std::int32_t> g1;
        std::span<const std::int32_t> c1;
        std::span<const std::int32_t> g2;
        std::span<const std::int32_t> c2;
        std::span<const double> s;

        explicit Columns(const base::ContractReader& reader);
    };

    struct Spring
    {
        ScalarConnection connection;
        double stiffness = 0.0;
        double stressCoefficient = 0.0;
    };

    [[nodiscard]] static Resolution<Spring> resolve(const ElementBatchView& view, const Columns& columns,
                                                    std::int64_t row);
};

} // namespace oos::elements
