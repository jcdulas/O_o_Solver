#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace oos::linalg
{

namespace
{

// The DOFs of one node in a vector: the first and how many (1 to SparsePattern::MAX_NODE_DOFS).
struct NodeDofs
{
    std::size_t first;
    std::size_t count;
};

// y += V*x for one stored block V (column-major, the DOFs of the node `rows` by those of the node `columns`) and, off
// the diagonal (isTransposed), y += transpose(V)*x for the upper block it stands for. Each element of y receives the
// block's terms in the order of the textbook loops (for c, for r: y(rows + r) += V(r, c)*x(columns + c), and off the
// diagonal y(columns + c) += V(r, c)*x(rows + r)): an element of the lower part by c increasing, one of the transpose
// by r increasing, each product rounded before it is added (linalg: -ffp-contract=off). The two parts write the DOFs
// of different nodes, so applying the transpose apart keeps the order of every element's terms (the same bits).
// PERFORMANCE: each part sums in local accumulators, where the interleaved loops read and wrote y through memory the
// compiler had to assume aliased (the products of the stiffness with the basis vectors of every static analysis: its
// SPC forces and EPSILON residuals).
// Measured on a stiffened barrel panel model.
void applyBlock(const double* values, NodeDofs rows, NodeDofs columns, bool isTransposed, const double* x,
                double* y) noexcept
{
    constexpr auto MAX_DOFS = static_cast<std::size_t>(SparsePattern::MAX_NODE_DOFS);
    std::array<double, MAX_DOFS> lower{};
    for (std::size_t r = 0; r < rows.count; ++r)
        lower[r] = y[rows.first + r];
    for (std::size_t c = 0; c < columns.count; ++c)
    {
        const double xc = x[columns.first + c];
        const double* column = values + c * rows.count;
        for (std::size_t r = 0; r < rows.count; ++r)
            lower[r] += column[r] * xc;
    }
    for (std::size_t r = 0; r < rows.count; ++r)
        y[rows.first + r] = lower[r];
    if (!isTransposed)
        return;
    for (std::size_t c = 0; c < columns.count; ++c)
    {
        const double* column = values + c * rows.count;
        double sum = y[columns.first + c];
        for (std::size_t r = 0; r < rows.count; ++r)
            sum += column[r] * x[rows.first + r];
        y[columns.first + c] = sum;
    }
}

} // namespace

BlockSparseMatrix::BlockSparseMatrix(const SparsePattern& pattern, base::MemoryBudget& budget, const std::string& label)
    : m_pattern(&pattern), m_values(budget, static_cast<std::size_t>(pattern.valueCount()), label)
{
}

BlockSparseMatrix::BlockSparseMatrix(const SparsePattern& pattern, base::MemoryBudget& budget, const std::string& label,
                                     base::ThreadPool* pool)
    : m_pattern(&pattern), m_values(budget, static_cast<std::size_t>(pattern.valueCount()), label, base::FOR_OVERWRITE)
{
    base::fillOnPool(pool, m_values.span(), 0.0);
}

std::span<double> BlockSparseMatrix::blockValues(std::int64_t block) noexcept
{
    const std::int64_t begin = m_pattern->blockValueOffset(block);
    return m_values.span().subspan(static_cast<std::size_t>(begin),
                                   static_cast<std::size_t>(m_pattern->blockValueOffset(block + 1) - begin));
}

std::span<const double> BlockSparseMatrix::blockValues(std::int64_t block) const noexcept
{
    const std::int64_t begin = m_pattern->blockValueOffset(block);
    return m_values.span().subspan(static_cast<std::size_t>(begin),
                                   static_cast<std::size_t>(m_pattern->blockValueOffset(block + 1) - begin));
}

void BlockSparseMatrix::setZero() noexcept
{
    std::fill_n(m_values.data(), m_values.size(), 0.0);
}

void BlockSparseMatrix::multiply(std::span<const double> x, std::span<double> y) const
{
    const auto dofs = static_cast<std::size_t>(m_pattern->dofCount());
    if (x.size() != dofs || y.size() != dofs)
        throw std::invalid_argument(
            base::concatenate({"BlockSparseMatrix::multiply: expected vectors of ", std::to_string(dofs),
                               " values, found ", std::to_string(x.size()), " and ", std::to_string(y.size())}));
    std::fill(y.begin(), y.end(), 0.0);
    for (std::int32_t column = 0; column < m_pattern->nodeCount(); ++column)
    {
        const NodeDofs columnDofs{static_cast<std::size_t>(m_pattern->firstDof(column)),
                                  static_cast<std::size_t>(m_pattern->nodeDofCount(column))};
        for (std::int64_t block = m_pattern->columnBegin(column); block < m_pattern->columnBegin(column + 1); ++block)
        {
            const std::int32_t row = m_pattern->blockRow(block);
            const NodeDofs rowDofs{static_cast<std::size_t>(m_pattern->firstDof(row)),
                                   static_cast<std::size_t>(m_pattern->nodeDofCount(row))};
            applyBlock(blockValues(block).data(), rowDofs, columnDofs, row != column, x.data(), y.data());
        }
    }
}

namespace
{

// The most columns one task of the block product takes: the 6 × 6 block of the matrix stays in the first-level cache
// while it is applied to them, and the accesses to the columns' rows stay few enough for the cache too.
constexpr std::size_t MAX_COLUMNS_PER_TASK = 16;

} // namespace

// PERFORMANCE: the products of a symmetric stiffness with many vectors (the SPC forces and the EPSILON residuals of the
// basis vectors of a static analysis) read the matrix once per group of columns instead of once per column, and the
// groups run on the pool; each column's terms come in the order of multiply(), with the same separately rounded
// operations (linalg is compiled with -ffp-contract=off), so the bits are those of the products one by one.
// Measured on the output recovery of large models.
void BlockSparseMatrix::multiply(ConstDenseMatrixView x, DenseMatrixView y, base::ThreadPool* pool) const
{
    const auto dofs = static_cast<std::size_t>(m_pattern->dofCount());
    if (x.rows != dofs || y.rows != dofs || x.columns != y.columns)
        throw std::invalid_argument(base::concatenate(
            {"BlockSparseMatrix::multiply: expected blocks of ", std::to_string(dofs),
             " rows with as many columns, found ", std::to_string(x.rows), " × ", std::to_string(x.columns), " and ",
             std::to_string(y.rows), " × ", std::to_string(y.columns)}));
    const std::size_t columns = x.columns;
    if (columns == 0)
        return;
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    const std::size_t perTask = std::clamp<std::size_t>((columns + threads - 1) / threads, 1, MAX_COLUMNS_PER_TASK);
    const std::size_t tasks = (columns + perTask - 1) / perTask;
    const auto group = [&](std::size_t task)
    {
        const std::size_t first = task * perTask;
        const std::size_t last = std::min(columns, first + perTask);
        for (std::size_t j = first; j < last; ++j)
            std::fill_n(&y(0, j), dofs, 0.0);
        for (std::int32_t column = 0; column < m_pattern->nodeCount(); ++column)
        {
            const NodeDofs columnDofs{static_cast<std::size_t>(m_pattern->firstDof(column)),
                                      static_cast<std::size_t>(m_pattern->nodeDofCount(column))};
            for (std::int64_t block = m_pattern->columnBegin(column); block < m_pattern->columnBegin(column + 1);
                 ++block)
            {
                const std::int32_t row = m_pattern->blockRow(block);
                const NodeDofs rowDofs{static_cast<std::size_t>(m_pattern->firstDof(row)),
                                       static_cast<std::size_t>(m_pattern->nodeDofCount(row))};
                const double* values = blockValues(block).data();
                // The block of multiply() on each column, so that each column receives its terms in the same order.
                for (std::size_t j = first; j < last; ++j)
                    applyBlock(values, rowDofs, columnDofs, row != column, x.pointer(0, j), &y(0, j));
            }
        }
    };
    if (pool != nullptr && tasks > 1)
        pool->runBatch(tasks, group);
    else
        for (std::size_t task = 0; task < tasks; ++task)
            group(task);
}

} // namespace oos::linalg
