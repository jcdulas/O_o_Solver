#pragma once

#include "oos/api/Optimization.hpp"

namespace oos::api
{
// Fixed CSR layout for normalized constraints (feasible when g <= 0).
// Omit rowColumns to include every variable. A supplied pattern must cover
// structural dependencies, including entries that vanish at the initial point.
class ConstraintJacobian
{
public:
    explicit ConstraintJacobian(const OptimizationEvaluation& point,
                                const std::optional<std::vector<std::vector<std::size_t>>>& rowColumns = {});
    [[nodiscard]] const auto& rowOffsets() const noexcept { return m_offsets; }
    [[nodiscard]] const auto& columnIndices() const noexcept { return m_columns; }
    [[nodiscard]] std::size_t variableCount() const noexcept { return m_variables; }
    // Order matches columnIndices(). A failure invalidates the output of this call.
    void values(OptimizationEvaluation& point, std::span<double> output) const;
    // Check the constraint identities before returning trial values to a fixed-size optimizer.
    void constraints(const OptimizationEvaluation& point, std::span<double> output) const;

private:
    void checkPoint(const OptimizationEvaluation& point) const;
    std::shared_ptr<const OptimizationProblem> m_problem;
    std::vector<design::ConstraintValue> m_rows;
    std::size_t m_variables;
    std::vector<std::size_t> m_offsets, m_columns;
};
} // namespace oos::api
