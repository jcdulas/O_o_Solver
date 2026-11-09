#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstdint>
#include <span>

namespace oos::assembly
{

// The load matrix of one family in its compact form, on which the load basis tests the rank: one
// column per load case (or per load set as written), one row per loaded DOF (concentrated loads) or per value written
// on a card (distributed loads: a corner pressure of an element), sparse by column (compressed columns), each value
// with the precision it was written with (0 when only the round-off is considered).
//
// Built column by column in order: the column and nonzero counts are known beforehand (the builder counts them in a
// first pass), so the arrays are budgeted once.
//
// Thread-safety: immutable once filled; concurrent reads are safe. Ownership: owns its arrays (budgeted).
class CompactLoadMatrix
{
public:
    CompactLoadMatrix() = default;
    // Throws std::invalid_argument for a negative count; MemoryBudgetError.
    CompactLoadMatrix(base::MemoryBudget& budget, std::int64_t rowCount, std::int64_t columnCount,
                      std::int64_t nonzeroCount);

    // Appends the next column. Throws std::invalid_argument when all columns are already there, when the rows,
    // values and precisions differ in length, a row is out of range, or the nonzeros would exceed the count given.
    void appendColumn(std::span<const std::int64_t> rows, std::span<const double> values,
                      std::span<const double> precisions);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_rowCount; }
    [[nodiscard]] std::int64_t columnCount() const noexcept { return m_columnCount; }
    [[nodiscard]] std::int64_t filledColumnCount() const noexcept { return m_filled; }
    [[nodiscard]] std::span<const std::int64_t> rowsOf(std::int64_t column) const;
    [[nodiscard]] std::span<const double> valuesOf(std::int64_t column) const;
    [[nodiscard]] std::span<const double> precisionsOf(std::int64_t column) const;
    // The Euclidean norm of a column.
    [[nodiscard]] double normOf(std::int64_t column) const;

private:
    std::int64_t m_rowCount = 0;
    std::int64_t m_columnCount = 0;
    std::int64_t m_filled = 0;
    base::BudgetedBuffer<std::int64_t> m_starts; // columnCount + 1
    base::BudgetedBuffer<std::int64_t> m_rows;
    base::BudgetedBuffer<double> m_values;
    base::BudgetedBuffer<double> m_precisions;
};

} // namespace oos::assembly
