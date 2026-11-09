#pragma once

// SpringDofProperty — scalar spring with its stiffness on a SpringProp (CELAS1, SpringProp):
//
//   | SpringDofProperty | EID | PID | G1 | C1 | G2 | C2 |
//   | SpringProp  | PID | K   | GE | S  |
//
// Formulation: as SpringDofInline (springs/springDofInline), K, GE and S taken from the SpringProp. Element force f = K (u1 −
// u2), element stress S · f.
//
// Facets: stiffness, recovery (forces, stresses). No mass. GE plays no part in SOL 101, 103, 105. Differential
// stiffness, loads and design derivatives are not written yet.
//
// Not supported: nothing on these cards (the second SpringProp of a card is a separate SpringProp row in O_o_Reader).
// Problems reported by check(): those of SpringDofInline, and a PID that is not a SpringProp.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/core/ScalarConnection.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of SpringDofProperty. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class SpringDofPropertyElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "SpringDofProperty"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"SpringProp"}; }

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
        std::span<const std::int32_t> g1;
        std::span<const std::int32_t> c1;
        std::span<const std::int32_t> g2;
        std::span<const std::int32_t> c2;
        PropertyRows properties;

        explicit Columns(const ElementBatchView& view);
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
