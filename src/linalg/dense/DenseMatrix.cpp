#include "oos/linalg/dense/DenseMatrix.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <cstdint>
#include <limits>
#include <string>

namespace oos::linalg
{
namespace
{

// rows × columns as a count of values, refused before any reservation when it does not fit 64 bits.
std::size_t valueCount(const base::MemoryBudget& budget, std::size_t leadingDimension, std::size_t columns,
                       const std::string& label)
{
    if (columns != 0 && leadingDimension > std::numeric_limits<std::size_t>::max() / columns)
        throw base::MemoryBudgetError(
            base::concatenate({"memory budget exceeded: the dense matrix '", label, "' of ",
                               std::to_string(leadingDimension), " × ", std::to_string(columns),
                               " values has more than 2^64 values; the size is wrong or the matrix must be split: "
                               "check the size, or condense the model in units"}),
            base::MemoryShortfall{std::numeric_limits<std::uint64_t>::max(), budget.available(), budget.limit(), {}});
    return leadingDimension * columns;
}

} // namespace

std::size_t DenseMatrix::alignedLeadingDimension(std::size_t rows)
{
    if (rows <= COLUMN_ALIGNMENT_DOUBLES)
        return COLUMN_ALIGNMENT_DOUBLES;
    if (rows > std::numeric_limits<std::size_t>::max() - COLUMN_ALIGNMENT_DOUBLES)
        return rows; // a size that large is refused by the budget anyway
    return (rows + COLUMN_ALIGNMENT_DOUBLES - 1) / COLUMN_ALIGNMENT_DOUBLES * COLUMN_ALIGNMENT_DOUBLES;
}

DenseMatrix::DenseMatrix(base::MemoryBudget& budget, std::size_t rows, std::size_t columns, const std::string& label)
    : m_rows(rows),
      m_columns(columns),
      m_leadingDimension(alignedLeadingDimension(rows)),
      m_values(budget, valueCount(budget, m_leadingDimension, columns, label), label)
{
}

DenseMatrix::DenseMatrix(base::MemoryBudget& budget, std::size_t rows, std::size_t columns, const std::string& label,
                         base::ThreadPool* pool)
    : m_rows(rows),
      m_columns(columns),
      m_leadingDimension(alignedLeadingDimension(rows)),
      m_values(budget, valueCount(budget, m_leadingDimension, columns, label), label, base::FOR_OVERWRITE)
{
    base::fillOnPool(pool, m_values.span(), 0.0);
}

} // namespace oos::linalg
