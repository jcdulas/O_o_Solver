#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// A symmetric matrix of scalar equations stored as its lower triangle in compressed sparse columns:
// `int32` row indices, `int64` column offsets. Column j holds the rows i >= j, ascending, the diagonal first. It is
// the input of the factorisation: the node blocks of a BlockSparseMatrix with the eliminated DOFs removed and the
// equations in the order of a DofSelection.
//
// The pattern is structural: a term of a stored block is kept even when its value is zero, so that the pattern of a
// conversion depends on the node graph and the selection only, never on the values (the symbolic factor of the
// pattern then serves every matrix assembled on it).
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted); the budget must outlive it. Movable,
// not copyable.
class SymmetricCsc
{
public:
    SymmetricCsc() = default;

    // The kept DOFs of `matrix`, in the equation order of `selection`. The columns are sorted on the pool (each
    // column by one thread: the result does not depend on the number of threads).
    // Throws std::invalid_argument when the selection was not made for the pattern of the matrix (other node DOF
    // counts), MemoryBudgetError when the result does not fit.
    // control: checked at every node column and between pieces of the sort (OperationCancelledError); its counter is
    // advanced by the equations sorted, dimension() in all.
    [[nodiscard]] static SymmetricCsc fromBlocks(const BlockSparseMatrix& matrix, const DofSelection& selection,
                                                 base::MemoryBudget& budget, base::ThreadPool* pool = nullptr,
                                                 const base::OperationControl& control = {});

    [[nodiscard]] std::int32_t dimension() const noexcept { return m_dimension; }
    [[nodiscard]] std::int64_t nonZeroCount() const noexcept { return static_cast<std::int64_t>(m_rows.size()); }
    [[nodiscard]] std::span<const std::int64_t> columnOffsets() const noexcept { return m_columnOffsets.span(); }
    [[nodiscard]] std::span<const std::int32_t> rowIndices() const noexcept { return m_rows.span(); }
    [[nodiscard]] std::span<const double> values() const noexcept { return m_values.span(); }

    // alpha·a + beta·b for two matrices of the same structure (the stiffness and the mass assembled on one pattern):
    // the shifted matrix K − σM of the eigenvalue solvers, whose symbolic factor is that of the pattern. Each value is
    // alpha·a_k + beta·b_k, rounded once per product and once for the sum. Throws std::invalid_argument when the
    // structures differ (hasSameStructure), MemoryBudgetError when the result does not fit.
    [[nodiscard]] static SymmetricCsc combine(double alpha, const SymmetricCsc& a, double beta, const SymmetricCsc& b,
                                              base::MemoryBudget& budget);

    // The leading principal block of `matrix`, its first `size` equations: the interior block K_oo of a unit whose
    // boundary equations come last (the fixed-interface modes of Craig-Bampton). Its pattern is the pattern
    // of those columns restricted to those rows, so the leading blocks of two matrices of one structure have one
    // structure. Throws std::invalid_argument for a size outside [0, dimension()], MemoryBudgetError when it does not
    // fit.
    [[nodiscard]] static SymmetricCsc leadingBlock(const SymmetricCsc& matrix, std::int32_t size,
                                                   base::MemoryBudget& budget);

    // The matrix of the given lower triangle by columns (each column's rows ascending, from its diagonal on): the
    // matrices a caller gathers itself (the units of multilevel substructuring). Throws std::invalid_argument when the
    // arrays are not such a triangle, MemoryBudgetError.
    [[nodiscard]] static SymmetricCsc fromCompressedColumns(std::int32_t dimension,
                                                            std::span<const std::int64_t> offsets,
                                                            std::span<const std::int32_t> rows,
                                                            std::span<const double> values, base::MemoryBudget& budget);
    // True when `other` has the same dimension, column offsets and row indices.
    [[nodiscard]] bool hasSameStructure(const SymmetricCsc& other) const noexcept;

    // y = A·x with A symmetric (both triangles implied). Serial, fixed order: for checks and tests.
    // Throws std::invalid_argument when a size is not dimension().
    void multiply(std::span<const double> x, std::span<double> y) const;

    // Y = A·X for a block of vectors (X and Y dimension() × r, distinct storage). Each column is computed as
    // multiply() computes it, in the same order of terms, whatever the block: the columns are the pieces of the
    // parallel work (COLUMNS_PER_MULTIPLY_PIECE per piece), so the bits do not depend on the pool. Throws
    // std::invalid_argument when a size differs; OperationCancelledError between pieces.
    void multiply(ConstDenseMatrixView x, DenseMatrixView y, base::ThreadPool* pool = nullptr,
                  const base::CancellationToken* cancellation = nullptr) const;

    // The columns of one piece of the parallel block product.
    static constexpr std::size_t COLUMNS_PER_MULTIPLY_PIECE = 1;

private:
    std::int32_t m_dimension = 0;
    base::BudgetedBuffer<std::int64_t> m_columnOffsets; // dimension + 1
    base::BudgetedBuffer<std::int32_t> m_rows;
    base::BudgetedBuffer<double> m_values;
};

} // namespace oos::linalg
