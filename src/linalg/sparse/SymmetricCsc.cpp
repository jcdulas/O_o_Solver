#include "oos/linalg/sparse/SymmetricCsc.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

// The owners of columns per piece of the parallel conversion (fromBlocks), some 6 equations each.
constexpr std::size_t OWNERS_PER_PIECE = 1024;
// A node without a kept DOF (fromBlocks).
constexpr std::int32_t NO_EQUATION = -1;
// The most DOFs of a node (a grid point's six).
constexpr std::size_t MAX_NODE_DOFS = 6;

void requireMatchingSelection(const SparsePattern& pattern, const DofSelection& selection)
{
    bool isMatching = selection.nodeCount() == pattern.nodeCount();
    for (std::int32_t node = 0; isMatching && node <= pattern.nodeCount(); ++node)
        isMatching =
            selection.firstDof(node) == (node < pattern.nodeCount() ? pattern.firstDof(node) : pattern.dofCount());
    if (!isMatching)
        throw std::invalid_argument(base::concatenate(
            {"SymmetricCsc::fromBlocks: the DOF selection (", std::to_string(selection.nodeCount()),
             " nodes) was not made for the pattern of the matrix (", std::to_string(pattern.nodeCount()),
             " nodes, other DOF counts); build it from the same node DOF counts"}));
}

} // namespace

SymmetricCsc SymmetricCsc::fromBlocks(const BlockSparseMatrix& matrix, const DofSelection& selection,
                                      base::MemoryBudget& budget, base::ThreadPool* pool,
                                      const base::OperationControl& control)
{
    requireMatchingSelection(matrix.pattern(), selection);
    const SparsePattern& pattern = matrix.pattern();
    SymmetricCsc result;
    result.m_dimension = selection.keptDofCount();
    const auto dimension = static_cast<std::size_t>(result.m_dimension);
    const auto nodeCount = static_cast<std::size_t>(pattern.nodeCount());
    // PERFORMANCE: the terms gathered by the node that owns their columns, on the pool: the kept DOFs of a node are
    // consecutive equations (DofSelection), so the lower-triangle terms of a node block (R, C) all fall in the columns
    // of the node with the lower equations (C on the diagonal), and each column is written by its owner alone; every
    // column is then sorted by row, so the result is the serial conversion's whatever the threads (each (row, column)
    // appears once). The two passes over the blocks ran on one thread: 0.15 of the 0.18 s of the conversion of the
    // 1 M DOF plate.
    // Measured on the 1 M DOF plate.
    base::BudgetedBuffer<std::int32_t> firstEquation(budget, nodeCount, "symmetric CSC: node equations",
                                                     base::FOR_OVERWRITE);
    base::BudgetedBuffer<std::int32_t> nodeOfEquation(budget, dimension, "symmetric CSC: node equations",
                                                      base::FOR_OVERWRITE);
    std::fill(nodeOfEquation.data(), nodeOfEquation.data() + dimension, NO_EQUATION);
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        firstEquation[node] = NO_EQUATION;
        for (int c = 0; c < pattern.nodeDofCount(static_cast<std::int32_t>(node)); ++c)
            if (const std::int32_t q = selection.equation(static_cast<std::int32_t>(node), c);
                q != DofSelection::ELIMINATED)
            {
                firstEquation[node] = q;
                nodeOfEquation[static_cast<std::size_t>(q)] = static_cast<std::int32_t>(node);
                break;
            }
    }
    // The owners in the order of their equations.
    std::size_t ownerCount = 0;
    for (std::size_t q = 0; q < dimension; ++q)
        if (nodeOfEquation[q] != NO_EQUATION)
            nodeOfEquation[ownerCount++] = nodeOfEquation[q];
    // The blocks of each owner, in block order.
    base::BudgetedBuffer<std::int64_t> blockStart(budget, nodeCount + 1, "symmetric CSC: blocks by owner");
    const auto ownerOf = [&](std::int32_t row, std::int32_t column)
    {
        const std::int32_t rowFirst = firstEquation[static_cast<std::size_t>(row)];
        const std::int32_t columnFirst = firstEquation[static_cast<std::size_t>(column)];
        if (rowFirst == NO_EQUATION || columnFirst == NO_EQUATION)
            return NO_EQUATION;
        return row == column || columnFirst < rowFirst ? column : row;
    };
    for (std::int32_t column = 0; column < pattern.nodeCount(); ++column)
        for (std::int64_t block = pattern.columnBegin(column); block < pattern.columnBegin(column + 1); ++block)
            if (const std::int32_t owner = ownerOf(pattern.blockRow(block), column); owner != NO_EQUATION)
                ++blockStart[static_cast<std::size_t>(owner) + 1];
    for (std::size_t node = 0; node < nodeCount; ++node)
        blockStart[node + 1] += blockStart[node];
    base::BudgetedBuffer<std::int64_t> blocksOf(budget, static_cast<std::size_t>(blockStart[nodeCount]),
                                                "symmetric CSC: blocks by owner", base::FOR_OVERWRITE);
    base::BudgetedBuffer<std::int32_t> columnsOf(budget, static_cast<std::size_t>(blockStart[nodeCount]),
                                                 "symmetric CSC: blocks by owner", base::FOR_OVERWRITE);
    {
        base::BudgetedBuffer<std::int64_t> next(budget, nodeCount, "symmetric CSC: blocks by owner",
                                                base::FOR_OVERWRITE);
        std::copy_n(blockStart.data(), nodeCount, next.data());
        for (std::int32_t column = 0; column < pattern.nodeCount(); ++column)
            for (std::int64_t block = pattern.columnBegin(column); block < pattern.columnBegin(column + 1); ++block)
                if (const std::int32_t owner = ownerOf(pattern.blockRow(block), column); owner != NO_EQUATION)
                {
                    const auto at = static_cast<std::size_t>(next[static_cast<std::size_t>(owner)]++);
                    blocksOf[at] = block;
                    columnsOf[at] = column;
                }
    }
    // Calls visit(row equation, column equation, value) for the kept terms of the lower triangle of the blocks of
    // `owner` (their columns its equations).
    const auto forEachTermOf = [&](std::int32_t owner, auto&& visit)
    {
        for (std::int64_t k = blockStart[static_cast<std::size_t>(owner)];
             k < blockStart[static_cast<std::size_t>(owner) + 1]; ++k)
        {
            const std::int64_t block = blocksOf[static_cast<std::size_t>(k)];
            const std::int32_t row = pattern.blockRow(block);
            const std::int32_t column = columnsOf[static_cast<std::size_t>(k)];
            const int columnDofs = pattern.nodeDofCount(column);
            const int rowDofs = pattern.nodeDofCount(row);
            const std::span<const double> values = matrix.blockValues(block);
            for (int c = 0; c < columnDofs; ++c)
            {
                const std::int32_t q = selection.equation(column, c);
                if (q == DofSelection::ELIMINATED)
                    continue;
                // In a diagonal block, the lower triangle of the node (the DOFs of a node keep their order).
                for (int r = row == column ? c : 0; r < rowDofs; ++r)
                {
                    const std::int32_t p = selection.equation(row, r);
                    if (p == DofSelection::ELIMINATED)
                        continue;
                    const double value = values[static_cast<std::size_t>(c * rowDofs + r)];
                    if (p >= q)
                        visit(p, q, value);
                    else
                        visit(q, p, value);
                }
            }
        }
    };
    result.m_columnOffsets = base::BudgetedBuffer<std::int64_t>(budget, dimension + 1, "symmetric CSC: offsets");
    base::parallelFor(
        pool, ownerCount, OWNERS_PER_PIECE,
        [&](std::size_t first, std::size_t end)
        {
            for (std::size_t k = first; k < end; ++k)
                forEachTermOf(nodeOfEquation[k], [&](std::int32_t, std::int32_t column, double)
                              { ++result.m_columnOffsets[static_cast<std::size_t>(column) + 1]; });
        },
        control.cancellation);
    for (std::size_t j = 0; j < dimension; ++j)
        result.m_columnOffsets[j + 1] += result.m_columnOffsets[j];
    const auto nonZeros = static_cast<std::size_t>(result.m_columnOffsets[dimension]);
    result.m_rows = base::BudgetedBuffer<std::int32_t>(budget, nonZeros, "symmetric CSC: rows", base::FOR_OVERWRITE);
    result.m_values = base::BudgetedBuffer<double>(budget, nonZeros, "symmetric CSC: values", base::FOR_OVERWRITE);
    base::parallelFor(
        pool, ownerCount, OWNERS_PER_PIECE,
        [&](std::size_t first, std::size_t end)
        {
            std::vector<std::pair<std::int32_t, double>> column; // bounded by a column
            for (std::size_t k = first; k < end; ++k)
            {
                const std::int32_t owner = nodeOfEquation[k];
                const auto firstColumn = static_cast<std::size_t>(firstEquation[static_cast<std::size_t>(owner)]);
                // The owner's columns: its consecutive equations (at most a node's DOFs).
                std::array<std::int64_t, MAX_NODE_DOFS> cursor{};
                std::size_t columns = 0;
                for (int c = 0; c < pattern.nodeDofCount(owner); ++c)
                    columns += selection.equation(owner, c) != DofSelection::ELIMINATED ? 1 : 0;
                for (std::size_t j = 0; j < columns; ++j)
                    cursor[j] = result.m_columnOffsets[firstColumn + j];
                forEachTermOf(owner,
                              [&](std::int32_t row, std::int32_t columnEquation, double value)
                              {
                                  const auto at = static_cast<std::size_t>(
                                      cursor[static_cast<std::size_t>(columnEquation) - firstColumn]++);
                                  result.m_rows[at] = row;
                                  result.m_values[at] = value;
                              });
                // Each (row, column) appears once (one block per node pair, one term per DOF pair): sorting by row is
                // total.
                for (std::size_t j = firstColumn; j < firstColumn + columns; ++j)
                {
                    const auto begin = static_cast<std::size_t>(result.m_columnOffsets[j]);
                    const auto stop = static_cast<std::size_t>(result.m_columnOffsets[j + 1]);
                    column.clear();
                    for (std::size_t t = begin; t < stop; ++t)
                        column.emplace_back(result.m_rows[t], result.m_values[t]);
                    std::sort(column.begin(), column.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    for (std::size_t t = begin; t < stop; ++t)
                    {
                        result.m_rows[t] = column[t - begin].first;
                        result.m_values[t] = column[t - begin].second;
                    }
                }
                control.advance(columns);
            }
        },
        control.cancellation);
    return result;
}

SymmetricCsc SymmetricCsc::combine(double alpha, const SymmetricCsc& a, double beta, const SymmetricCsc& b,
                                   base::MemoryBudget& budget)
{
    if (!a.hasSameStructure(b))
        throw std::invalid_argument(base::concatenate(
            {"SymmetricCsc::combine: the matrices must have the same structure, found dimensions ",
             std::to_string(a.m_dimension), " and ", std::to_string(b.m_dimension), " with ",
             std::to_string(a.nonZeroCount()), " and ", std::to_string(b.nonZeroCount()),
             " terms or other rows; assemble both on one sparse pattern and convert them with one DOF selection"}));
    SymmetricCsc result;
    result.m_dimension = a.m_dimension;
    const auto dimension = static_cast<std::size_t>(a.m_dimension);
    const std::size_t nonZeros = a.m_rows.size();
    result.m_columnOffsets =
        base::BudgetedBuffer<std::int64_t>(budget, dimension + 1, "symmetric CSC: offsets", base::FOR_OVERWRITE);
    result.m_rows = base::BudgetedBuffer<std::int32_t>(budget, nonZeros, "symmetric CSC: rows", base::FOR_OVERWRITE);
    result.m_values = base::BudgetedBuffer<double>(budget, nonZeros, "symmetric CSC: values", base::FOR_OVERWRITE);
    std::copy_n(a.m_columnOffsets.data(), dimension + 1, result.m_columnOffsets.data());
    std::copy_n(a.m_rows.data(), nonZeros, result.m_rows.data());
    for (std::size_t k = 0; k < nonZeros; ++k)
    {
        // Two products and a sum, each rounded (no fused multiply-add is allowed by -ffp-contract=off): the same bits
        // on every machine.
        const double left = alpha * a.m_values[k];
        const double right = beta * b.m_values[k];
        result.m_values[k] = left + right;
    }
    return result;
}

void SymmetricCsc::scaleSymmetric(std::span<const double> scale)
{
    if (scale.size() != static_cast<std::size_t>(m_dimension))
        throw std::invalid_argument(
            base::concatenate({"SymmetricCsc::scaleSymmetric: expected ", std::to_string(m_dimension),
                               " scales (one per equation), found ", std::to_string(scale.size())}));
    for (std::size_t j = 0; j < scale.size(); ++j)
        for (auto k = m_columnOffsets[j]; k < m_columnOffsets[j + 1]; ++k)
        {
            const auto term = static_cast<std::size_t>(k);
            m_values[term] = scale[static_cast<std::size_t>(m_rows[term])] * m_values[term] * scale[j];
        }
}

SymmetricCsc SymmetricCsc::leadingBlock(const SymmetricCsc& matrix, std::int32_t size, base::MemoryBudget& budget)
{
    if (size < 0 || size > matrix.m_dimension)
        throw std::invalid_argument(
            base::concatenate({"SymmetricCsc::leadingBlock: expected a size in [0, ",
                               std::to_string(matrix.m_dimension), "], found ", std::to_string(size)}));
    const auto columns = static_cast<std::size_t>(size);
    // The rows of a column are ascending: those of the block are a prefix of each column.
    std::size_t nonZeros = 0;
    for (std::size_t j = 0; j < columns; ++j)
        for (auto k = matrix.m_columnOffsets[j]; k < matrix.m_columnOffsets[j + 1]; ++k)
            nonZeros += matrix.m_rows[static_cast<std::size_t>(k)] < size ? 1 : 0;
    SymmetricCsc result;
    result.m_dimension = size;
    result.m_columnOffsets =
        base::BudgetedBuffer<std::int64_t>(budget, columns + 1, "symmetric CSC: offsets", base::FOR_OVERWRITE);
    result.m_rows = base::BudgetedBuffer<std::int32_t>(budget, nonZeros, "symmetric CSC: rows", base::FOR_OVERWRITE);
    result.m_values = base::BudgetedBuffer<double>(budget, nonZeros, "symmetric CSC: values", base::FOR_OVERWRITE);
    std::size_t next = 0;
    result.m_columnOffsets[0] = 0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        for (auto k = matrix.m_columnOffsets[j]; k < matrix.m_columnOffsets[j + 1]; ++k)
        {
            const auto entry = static_cast<std::size_t>(k);
            if (matrix.m_rows[entry] >= size)
                continue;
            result.m_rows[next] = matrix.m_rows[entry];
            result.m_values[next] = matrix.m_values[entry];
            ++next;
        }
        result.m_columnOffsets[j + 1] = static_cast<std::int64_t>(next);
    }
    return result;
}

SymmetricCsc SymmetricCsc::fromCompressedColumns(std::int32_t dimension, std::span<const std::int64_t> offsets,
                                                 std::span<const std::int32_t> rows, std::span<const double> values,
                                                 base::MemoryBudget& budget)
{
    const auto columns = static_cast<std::size_t>(std::max(dimension, 0));
    if (dimension < 0 || offsets.size() != columns + 1 || offsets.front() != 0 ||
        static_cast<std::size_t>(offsets.back()) != rows.size() || rows.size() != values.size())
        throw std::invalid_argument(base::concatenate(
            {"SymmetricCsc::fromCompressedColumns: expected ", std::to_string(columns + 1), " offsets from 0 to the ",
             std::to_string(rows.size()), " rows and as many values, found ", std::to_string(offsets.size()),
             " offsets and ", std::to_string(values.size()), " values; give the lower triangle by columns"}));
    for (std::size_t j = 0; j < columns; ++j)
    {
        for (auto k = offsets[j]; k < offsets[j + 1]; ++k)
        {
            const std::int32_t row = rows[static_cast<std::size_t>(k)];
            const bool isAscending = k == offsets[j] || rows[static_cast<std::size_t>(k) - 1] < row;
            if (row < static_cast<std::int32_t>(j) || row >= dimension || !isAscending)
                throw std::invalid_argument(base::concatenate(
                    {"SymmetricCsc::fromCompressedColumns: column ", std::to_string(j), " has the row ",
                     std::to_string(row), "; give each column's rows of the lower triangle, ascending and below ",
                     std::to_string(dimension)}));
        }
    }
    SymmetricCsc result;
    result.m_dimension = dimension;
    result.m_columnOffsets =
        base::BudgetedBuffer<std::int64_t>(budget, columns + 1, "symmetric CSC: offsets", base::FOR_OVERWRITE);
    result.m_rows = base::BudgetedBuffer<std::int32_t>(budget, rows.size(), "symmetric CSC: rows", base::FOR_OVERWRITE);
    result.m_values = base::BudgetedBuffer<double>(budget, values.size(), "symmetric CSC: values", base::FOR_OVERWRITE);
    std::copy(offsets.begin(), offsets.end(), result.m_columnOffsets.data());
    std::copy(rows.begin(), rows.end(), result.m_rows.data());
    std::copy(values.begin(), values.end(), result.m_values.data());
    return result;
}

SymmetricCsc SymmetricCsc::withoutZeroTerms(const SymmetricCsc& matrix, base::MemoryBudget& budget)
{
    const auto columns = static_cast<std::size_t>(matrix.m_dimension);
    const auto kept = static_cast<std::size_t>(matrix.nonZeroCount() - matrix.zeroTermCount());
    SymmetricCsc result;
    result.m_dimension = matrix.m_dimension;
    result.m_columnOffsets =
        base::BudgetedBuffer<std::int64_t>(budget, columns + 1, "symmetric CSC: offsets", base::FOR_OVERWRITE);
    result.m_rows = base::BudgetedBuffer<std::int32_t>(budget, kept, "symmetric CSC: rows", base::FOR_OVERWRITE);
    result.m_values = base::BudgetedBuffer<double>(budget, kept, "symmetric CSC: values", base::FOR_OVERWRITE);
    std::size_t next = 0;
    result.m_columnOffsets[0] = 0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        for (auto k = static_cast<std::size_t>(matrix.m_columnOffsets[j]);
             k < static_cast<std::size_t>(matrix.m_columnOffsets[j + 1]); ++k)
        {
            if (matrix.m_values[k] == 0.0)
                continue;
            result.m_rows[next] = matrix.m_rows[k];
            result.m_values[next] = matrix.m_values[k];
            ++next;
        }
        result.m_columnOffsets[j + 1] = static_cast<std::int64_t>(next);
    }
    return result;
}

std::int64_t SymmetricCsc::zeroTermCount() const noexcept
{
    return std::count(m_values.data(), m_values.data() + m_values.size(), 0.0);
}

bool SymmetricCsc::hasSameStructure(const SymmetricCsc& other) const noexcept
{
    if (m_dimension != other.m_dimension || m_rows.size() != other.m_rows.size())
        return false;
    const auto offsets = static_cast<std::size_t>(m_dimension) + (m_columnOffsets.empty() ? 0 : 1);
    return std::equal(m_columnOffsets.data(), m_columnOffsets.data() + offsets, other.m_columnOffsets.data()) &&
           std::equal(m_rows.data(), m_rows.data() + m_rows.size(), other.m_rows.data());
}

void SymmetricCsc::multiply(ConstDenseMatrixView x, DenseMatrixView y, base::ThreadPool* pool,
                            const base::CancellationToken* cancellation) const
{
    const auto dimension = static_cast<std::size_t>(m_dimension);
    if (x.rows != dimension || y.rows != dimension || x.columns != y.columns)
        throw std::invalid_argument(base::concatenate(
            {"SymmetricCsc::multiply: expected blocks of ", std::to_string(dimension),
             " rows with the same columns, found ", std::to_string(x.rows), " × ", std::to_string(x.columns), " and ",
             std::to_string(y.rows), " × ", std::to_string(y.columns)}));
    // PERFORMANCE: the columns of a group in one pass over the terms. A pass per column reads the whole matrix for
    // each (12 bytes per term, far more than the caches hold) and makes one sum after the other; a pass per group
    // reads it once, and the sums of its columns, independent of each other, go side by side. The bits of a column do
    // not depend on its group. Measured on one thread: the mass products of the eigenvalue solve of 1 024 shells of
    // the mixed hybrid formulation (89 859 equations, 8.1 M terms, blocks of 7 columns), 15.8 s by column and 6.3 s
    // by groups, of a solve of 36.5 s then 26.5 s.
    const std::size_t threads = pool != nullptr ? std::max<std::size_t>(pool->threadCount(), 1) : 1;
    const std::size_t groupColumns =
        std::clamp<std::size_t>(base::pieceCount(x.columns, threads), 1, MULTIPLY_GROUP_COLUMNS);
    base::parallelFor(
        pool, base::pieceCount(x.columns, groupColumns), 1,
        [&](std::size_t firstGroup, std::size_t endGroup)
        {
            for (std::size_t group = firstGroup; group < endGroup; ++group)
            {
                const std::size_t first = group * groupColumns;
                multiplyGroup(x, y, first, std::min(groupColumns, x.columns - first));
            }
        },
        cancellation);
}

namespace
{

// Columns [first, first + Width) of Y = A·X in one pass over the terms of A (its lower triangle by columns: offsets,
// rows, values). Each column of Y gets the sums SymmetricCsc::multiply(span, span) makes, in the same order: a value
// of row j is first added to by the columns before j (their term of row j), then by column j.
template <std::size_t Width>
void multiplyColumnsTogether(std::span<const std::int64_t> offsets, std::span<const std::int32_t> rows,
                             std::span<const double> values, ConstDenseMatrixView x, DenseMatrixView y,
                             std::size_t first)
{
    std::array<const double*, Width> xColumns;
    std::array<double*, Width> yColumns;
    for (std::size_t c = 0; c < Width; ++c)
    {
        xColumns[c] = x.pointer(0, first + c);
        yColumns[c] = &y(0, first + c);
        std::fill_n(yColumns[c], y.rows, 0.0);
    }
    for (std::size_t j = 0; j + 1 < offsets.size(); ++j)
    {
        // Row j of X, and row j of Y while column j adds to it (no other row of this column is row j).
        std::array<double, Width> xOfColumn;
        std::array<double, Width> yOfColumn;
        for (std::size_t c = 0; c < Width; ++c)
        {
            xOfColumn[c] = xColumns[c][j];
            yOfColumn[c] = yColumns[c][j];
        }
        for (auto k = static_cast<std::size_t>(offsets[j]); k < static_cast<std::size_t>(offsets[j + 1]); ++k)
        {
            const auto i = static_cast<std::size_t>(rows[k]);
            const double value = values[k];
            if (i == j)
            {
                for (std::size_t c = 0; c < Width; ++c)
                    yOfColumn[c] += value * xOfColumn[c];
                continue;
            }
            for (std::size_t c = 0; c < Width; ++c)
            {
                yColumns[c][i] += value * xOfColumn[c];
                yOfColumn[c] += value * xColumns[c][i];
            }
        }
        for (std::size_t c = 0; c < Width; ++c)
            yColumns[c][j] = yOfColumn[c];
    }
}

} // namespace

void SymmetricCsc::multiplyGroup(ConstDenseMatrixView x, DenseMatrixView y, std::size_t first,
                                 std::size_t count) const
{
    const std::span<const std::int64_t> offsets = columnOffsets();
    const std::span<const std::int32_t> rows = rowIndices();
    const std::span<const double> terms = values();
    // One function per width, so that the sums of the columns of a group are written side by side.
    switch (count)
    {
    case 1:
        return multiplyColumnsTogether<1>(offsets, rows, terms, x, y, first);
    case 2:
        return multiplyColumnsTogether<2>(offsets, rows, terms, x, y, first);
    case 3:
        return multiplyColumnsTogether<3>(offsets, rows, terms, x, y, first);
    case 4:
        return multiplyColumnsTogether<4>(offsets, rows, terms, x, y, first);
    case 5:
        return multiplyColumnsTogether<5>(offsets, rows, terms, x, y, first);
    case 6:
        return multiplyColumnsTogether<6>(offsets, rows, terms, x, y, first);
    case 7:
        return multiplyColumnsTogether<7>(offsets, rows, terms, x, y, first);
    default:
        OOS_ASSERT(count == MULTIPLY_GROUP_COLUMNS);
        return multiplyColumnsTogether<8>(offsets, rows, terms, x, y, first);
    }
}

void SymmetricCsc::multiply(std::span<const double> x, std::span<double> y) const
{
    const auto dimension = static_cast<std::size_t>(m_dimension);
    if (x.size() != dimension || y.size() != dimension)
        throw std::invalid_argument(
            base::concatenate({"SymmetricCsc::multiply: expected vectors of ", std::to_string(dimension),
                               " values, found ", std::to_string(x.size()), " and ", std::to_string(y.size())}));
    std::fill(y.begin(), y.end(), 0.0);
    for (std::size_t j = 0; j < dimension; ++j)
    {
        for (auto k = static_cast<std::size_t>(m_columnOffsets[j]);
             k < static_cast<std::size_t>(m_columnOffsets[j + 1]); ++k)
        {
            const auto i = static_cast<std::size_t>(m_rows[k]);
            y[i] += m_values[k] * x[j];
            if (i != j)
                y[j] += m_values[k] * x[i];
        }
    }
}

} // namespace oos::linalg
