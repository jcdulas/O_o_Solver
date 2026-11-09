#pragma once

// SpringScalarInline — scalar spring between scalar points, its stiffness on the card:
//
//   | SpringScalarInline | EID | K | S1 | S2 |
//
// Formulation: as SpringDofInline (springs/springDofInline) between the single DOFs of the scalar points S1 and S2 (a blank S
// grounds that end). Element force f = K (u1 − u2). The card has no stress coefficient and no damping field, so the
// element gives forces only.
//
// Facets: stiffness, recovery (forces). No mass. Differential stiffness, loads and design derivatives are not written
// yet.
//
// Not supported: nothing on this card. Problems reported by check(): an S that is not a scalar point, both ends blank
// or equal.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ScalarConnection.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of SpringScalarInline. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class SpringScalarInlineElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "SpringScalarInline"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;

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
    struct Columns
    {
        std::span<const double> k;
        std::span<const std::int32_t> s1;
        std::span<const std::int32_t> s2;

        explicit Columns(const base::ContractReader& reader);
    };

    struct Spring
    {
        ScalarConnection connection;
        double stiffness = 0.0;
    };

    [[nodiscard]] static Resolution<Spring> resolve(const ElementBatchView& view, const Columns& columns,
                                                    std::int64_t row);
};

} // namespace oos::elements
