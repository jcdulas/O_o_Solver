#include "oos/linalg/dense/SymmetricPacked.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

// n(n+1)/2, refused before any reservation when it does not fit 64 bits.
std::size_t packedValueCount(const base::MemoryBudget& budget, std::size_t size, const std::string& label)
{
    const std::size_t even = size % 2 == 0 ? size : size + 1; // one of n, n+1 is even: halve it first
    const std::size_t other = size % 2 == 0 ? size + 1 : size;
    const std::size_t half = even / 2;
    if (size == std::numeric_limits<std::size_t>::max() ||
        (other != 0 && half > std::numeric_limits<std::size_t>::max() / other))
        throw base::MemoryBudgetError(
            base::concatenate({"memory budget exceeded: the packed symmetric matrix '", label, "' of order ",
                               std::to_string(size), " has more than 2^64 values; the size is wrong: check it"}),
            base::MemoryShortfall{std::numeric_limits<std::uint64_t>::max(), budget.available(), budget.limit(), {}});
    return half * other;
}

void requireSquare(std::size_t size, std::size_t rows, std::size_t columns, const char* function)
{
    if (rows != size || columns != size)
        throw std::invalid_argument(base::concatenate(
            {"SymmetricPacked::", function, ": expected a ", std::to_string(size), " × ", std::to_string(size),
             " matrix, found ", std::to_string(rows), " × ", std::to_string(columns)}));
}

} // namespace

SymmetricPacked::SymmetricPacked(base::MemoryBudget& budget, std::size_t size, const std::string& label)
    : m_size(size), m_values(budget, packedValueCount(budget, size, label), label)
{
}

void SymmetricPacked::copyLowerFrom(ConstDenseMatrixView source)
{
    requireSquare(m_size, source.rows, source.columns, "copyLowerFrom");
    std::size_t position = 0;
    for (std::size_t j = 0; j < m_size; ++j)
        for (std::size_t i = j; i < m_size; ++i)
            m_values[position++] = source(i, j);
}

void SymmetricPacked::copyTo(DenseMatrixView target) const
{
    requireSquare(m_size, target.rows, target.columns, "copyTo");
    std::size_t position = 0;
    for (std::size_t j = 0; j < m_size; ++j)
        for (std::size_t i = j; i < m_size; ++i)
        {
            const double value = m_values[position++];
            target(i, j) = value;
            target(j, i) = value;
        }
}

} // namespace oos::linalg
