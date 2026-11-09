#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace oos::linalg
{

// A symmetric matrix on a SparsePattern of node blocks, as the assembler fills it. Only the blocks of
// the lower triangle are stored; a diagonal block is stored complete (both of its triangles), so that an element
// matrix is added to it without a test per term, and a conversion reads its lower triangle.
//
// Thread-safety: as a std::vector of doubles; the assembler writes disjoint blocks from several threads. Ownership:
// owns its values (budgeted); the pattern is shared and must outlive the matrix, as must the budget. Movable, not
// copyable.
class BlockSparseMatrix
{
public:
    // A zero matrix on `pattern`. Throws MemoryBudgetError when its values do not fit.
    BlockSparseMatrix(const SparsePattern& pattern, base::MemoryBudget& budget, const std::string& label);
    // The same zero matrix, its values zeroed on `pool` (base::fillOnPool; null: the calling thread).
    BlockSparseMatrix(const SparsePattern& pattern, base::MemoryBudget& budget, const std::string& label,
                      base::ThreadPool* pool);

    [[nodiscard]] const SparsePattern& pattern() const noexcept { return *m_pattern; }

    // The values of a block, column-major, dofCount(row node) rows × dofCount(column node) columns.
    [[nodiscard]] std::span<double> blockValues(std::int64_t block) noexcept;
    [[nodiscard]] std::span<const double> blockValues(std::int64_t block) const noexcept;
    [[nodiscard]] std::span<double> values() noexcept { return m_values.span(); }
    [[nodiscard]] std::span<const double> values() const noexcept { return m_values.span(); }

    void setZero() noexcept;

    // y = A·x with A symmetric, x and y node-major DOF vectors of pattern().dofCount() values (std::invalid_argument
    // otherwise). Serial, in a fixed order: for checks and tests, not for solvers.
    void multiply(std::span<const double> x, std::span<double> y) const;
    // Y = A·X for every column of X (pattern().dofCount() rows, column-major; Y of the same shape): each column gets
    // the terms of multiply() in its order, so the bits of a call of multiply() per column whatever the threads or the
    // grouping. The columns are shared between the threads of `pool` (null: the caller's) in groups of contiguous
    // columns, each group reading the matrix once. Throws std::invalid_argument for a shape that does not match.
    void multiply(ConstDenseMatrixView x, DenseMatrixView y, base::ThreadPool* pool = nullptr) const;

private:
    const SparsePattern* m_pattern; // not owned
    base::BudgetedBuffer<double> m_values;
};

} // namespace oos::linalg
