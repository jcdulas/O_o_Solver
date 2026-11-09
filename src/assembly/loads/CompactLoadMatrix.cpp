#include "oos/assembly/loads/CompactLoadMatrix.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::assembly
{

CompactLoadMatrix::CompactLoadMatrix(base::MemoryBudget& budget, std::int64_t rowCount, std::int64_t columnCount,
                                     std::int64_t nonzeroCount)
    : m_rowCount(rowCount), m_columnCount(columnCount)
{
    if (rowCount < 0 || columnCount < 0 || nonzeroCount < 0)
        throw std::invalid_argument(base::concatenate({"CompactLoadMatrix: expected nonnegative counts, found ",
                                                       std::to_string(rowCount), " rows, ", std::to_string(columnCount),
                                                       " columns, ", std::to_string(nonzeroCount), " nonzeros"}));
    m_starts = base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(columnCount) + 1,
                                                  "compact load matrix columns");
    m_rows =
        base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(nonzeroCount), "compact load matrix rows");
    m_values =
        base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(nonzeroCount), "compact load matrix values");
    m_precisions =
        base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(nonzeroCount), "compact load matrix precisions");
}

void CompactLoadMatrix::appendColumn(std::span<const std::int64_t> rows, std::span<const double> values,
                                     std::span<const double> precisions)
{
    if (m_filled >= m_columnCount)
        throw std::invalid_argument(base::concatenate(
            {"CompactLoadMatrix: expected at most ", std::to_string(m_columnCount), " columns, found one more"}));
    if (rows.size() != values.size() || rows.size() != precisions.size())
        throw std::invalid_argument(base::concatenate(
            {"CompactLoadMatrix: expected as many rows, values and precisions, found ", std::to_string(rows.size()),
             ", ", std::to_string(values.size()), " and ", std::to_string(precisions.size())}));
    const auto start = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(m_filled)]);
    if (start + rows.size() > m_rows.size())
        throw std::invalid_argument(base::concatenate(
            {"CompactLoadMatrix: expected at most ", std::to_string(m_rows.size()), " nonzeros, found more"}));
    for (std::size_t k = 0; k < rows.size(); ++k)
    {
        if (rows[k] < 0 || rows[k] >= m_rowCount)
            throw std::invalid_argument(
                base::concatenate({"CompactLoadMatrix: expected a row in [0, ", std::to_string(m_rowCount), "), found ",
                                   std::to_string(rows[k])}));
        m_rows[start + k] = rows[k];
        m_values[start + k] = values[k];
        m_precisions[start + k] = precisions[k];
    }
    ++m_filled;
    m_starts[static_cast<std::size_t>(m_filled)] = static_cast<std::int64_t>(start + rows.size());
}

std::span<const std::int64_t> CompactLoadMatrix::rowsOf(std::int64_t column) const
{
    const auto begin = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(column)]);
    const auto end = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(column) + 1]);
    return m_rows.span().subspan(begin, end - begin);
}

std::span<const double> CompactLoadMatrix::valuesOf(std::int64_t column) const
{
    const auto begin = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(column)]);
    const auto end = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(column) + 1]);
    return m_values.span().subspan(begin, end - begin);
}

std::span<const double> CompactLoadMatrix::precisionsOf(std::int64_t column) const
{
    const auto begin = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(column)]);
    const auto end = static_cast<std::size_t>(m_starts[static_cast<std::size_t>(column) + 1]);
    return m_precisions.span().subspan(begin, end - begin);
}

double CompactLoadMatrix::normOf(std::int64_t column) const
{
    double sum = 0.0;
    for (const double value : valuesOf(column))
        sum += value * value;
    return std::sqrt(sum);
}

} // namespace oos::assembly
