#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <string>

namespace oos::linalg
{

// A dense column-major matrix of doubles, zero-initialised, its memory reserved in a MemoryBudget before it is
// allocated (a buffer whose size grows with the model is budgeted).
//
// Invariants: the leading dimension is the row count rounded up to a multiple of COLUMN_ALIGNMENT_DOUBLES (8 doubles,
// 64 bytes), so that every column starts on a 64-byte boundary, the alignment of the buffer (base::BUFFER_ALIGNMENT):
// the SIMD kernels may then load the first rows of any column with aligned loads, and two columns never share a cache
// line. The padding rows hold zeros and are never read by the kernels.
//
// Thread-safety: as a std::vector. Ownership: owns its memory and its reservation; movable, not copyable. The budget
// must outlive the matrix.
class DenseMatrix
{
public:
    static constexpr std::size_t COLUMN_ALIGNMENT_DOUBLES = 8;

    DenseMatrix() = default;
    // Throws MemoryBudgetError (nothing allocated) when rows × columns does not fit the budget, or does not fit 64
    // bits.
    DenseMatrix(base::MemoryBudget& budget, std::size_t rows, std::size_t columns, const std::string& label);
    // The same zero matrix, its values zeroed on `pool` (base::fillOnPool: the first touch of a large matrix spread
    // over the threads; null: the calling thread).
    DenseMatrix(base::MemoryBudget& budget, std::size_t rows, std::size_t columns, const std::string& label,
                base::ThreadPool* pool);

    [[nodiscard]] std::size_t rows() const noexcept { return m_rows; }
    [[nodiscard]] std::size_t columns() const noexcept { return m_columns; }
    [[nodiscard]] std::size_t leadingDimension() const noexcept { return m_leadingDimension; }
    [[nodiscard]] double* data() noexcept { return m_values.data(); }
    [[nodiscard]] const double* data() const noexcept { return m_values.data(); }

    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept
    {
        return m_values[i + j * m_leadingDimension];
    }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept
    {
        return m_values[i + j * m_leadingDimension];
    }

    [[nodiscard]] DenseMatrixView view() noexcept { return {m_values.data(), m_rows, m_columns, m_leadingDimension}; }
    [[nodiscard]] ConstDenseMatrixView view() const noexcept
    {
        return {m_values.data(), m_rows, m_columns, m_leadingDimension};
    }

    // The leading dimension DenseMatrix uses for `rows` rows (at least one alignment unit).
    [[nodiscard]] static std::size_t alignedLeadingDimension(std::size_t rows);

private:
    std::size_t m_rows = 0;
    std::size_t m_columns = 0;
    std::size_t m_leadingDimension = COLUMN_ALIGNMENT_DOUBLES;
    base::BudgetedBuffer<double> m_values;
};

} // namespace oos::linalg
