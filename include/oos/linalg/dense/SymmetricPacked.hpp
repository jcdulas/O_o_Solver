#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <string>

namespace oos::linalg
{

// A symmetric matrix stored as its lower triangle packed by columns (the LAPACK 'L' packed layout): column j holds
// rows j .. n-1, and starts after the n + (n-1) + ... + (n-j+1) values of the columns before it, so that element
// (i, j), i >= j, is at j·n − j(j−1)/2 + (i − j). Half the memory of a full dense matrix: the form in which a condensed
// matrix K* or a reduced mass is kept when it is not being factorised.
//
// The memory is reserved in a MemoryBudget before it is allocated. Thread-safety: as a std::vector. Ownership: owns
// its memory and its reservation; movable, not copyable. The budget must outlive the matrix.
class SymmetricPacked
{
public:
    SymmetricPacked() = default;
    // Throws MemoryBudgetError (nothing allocated) when n(n+1)/2 values do not fit the budget or 64 bits.
    SymmetricPacked(base::MemoryBudget& budget, std::size_t size, const std::string& label);

    [[nodiscard]] std::size_t size() const noexcept { return m_size; }
    [[nodiscard]] std::size_t valueCount() const noexcept { return m_values.size(); }
    [[nodiscard]] double* data() noexcept { return m_values.data(); }
    [[nodiscard]] const double* data() const noexcept { return m_values.data(); }

    // The position of element (i, j) or (j, i) in the packed array (either triangle: the matrix is symmetric).
    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j) const noexcept
    {
        if (i < j)
            return index(j, i);
        // For j = 0 the product j·(j − 1) wraps to 0·(2⁶⁴ − 1) = 0: no special case.
        return j * m_size - j * (j - 1) / 2 + (i - j);
    }
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept { return m_values[index(i, j)]; }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept { return m_values[index(i, j)]; }

    // Copies the lower triangle of the square `source` (its upper triangle is not read).
    void copyLowerFrom(ConstDenseMatrixView source);
    // Writes the full symmetric matrix into the square `target` (both triangles).
    void copyTo(DenseMatrixView target) const;

private:
    std::size_t m_size = 0;
    base::BudgetedBuffer<double> m_values;
};

} // namespace oos::linalg
