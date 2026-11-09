#include "BlockedLdlt.hpp"

#include "LdltArithmetic.hpp"

#include "oos/base/threading/CancellationToken.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

// D of the columns [first, first + count) of a factor in progress: d on the diagonal of `a`, e in `subdiagonal`.
BlockDiagonalView pivotsOf(DenseMatrixView a, const double* subdiagonal, std::size_t first, std::size_t count)
{
    return {&a(first, first), a.leadingDimension + 1, subdiagonal == nullptr ? nullptr : subdiagonal + first, count};
}

// Between two blocks: a block of a large front is a fraction of a second of work, the interruption delay.
void throwIfCancelled(const base::CancellationToken* cancellation)
{
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

// The current value of column `column` of the matrix being factorised, rows [firstRow, firstRow + values.size()),
// given `values` (its values after the updates of the blocks before firstBlockColumn): subtracts the terms
// L(i, p)·W(column, p) of the termCount columns p of the current block from firstBlockColumn, in increasing p (the
// chain of the reference family for the stored elements (i, column), i >= column).
void subtractBlockColumns(const KernelFamily& family, DenseMatrixView a, const double* subdiagonal,
                          std::size_t firstBlockColumn, std::size_t termCount, std::size_t firstRow, std::size_t column,
                          std::span<double> values, std::vector<double>& productRow)
{
    if (termCount == 0 || values.empty())
        return;
    computeProductRow(a, column, firstBlockColumn, pivotsOf(a, subdiagonal, firstBlockColumn, termCount),
                      std::span<double>(productRow.data(), termCount));
    ColumnProductOperands operands;
    operands.y = values.data();
    operands.a = &a(firstRow, firstBlockColumn);
    operands.lda = a.leadingDimension;
    operands.v = productRow.data();
    operands.rowCount = values.size();
    operands.innerCount = termCount;
    family.subtractColumnProducts(operands);
}

// The current value of the candidate column r of the pivot search, rows [firstRow, n): its rows i >= r are the stored
// elements (i, r), whose chain is Σ L(i, p)·W(r, p) (subtractBlockColumns); its rows i < r are the stored elements
// (r, i) of row r, whose chain is Σ L(r, p)·W(i, p): the rows of W are formed here, so that each element follows the
// chain of the element of the lower triangle it is (the product L(r, p)·W(i, p) is not L(i, p)·W(r, p) in floating
// point).
void subtractBlockColumnsFromCandidate(const KernelFamily& family, DenseMatrixView a, const double* subdiagonal,
                                       std::size_t firstBlockColumn, std::size_t firstRow, std::size_t candidate,
                                       std::span<double> values, std::vector<double>& productRow,
                                       std::vector<double>& productRows)
{
    const std::size_t termCount = firstRow - firstBlockColumn;
    const std::size_t above = candidate - firstRow;
    subtractBlockColumns(family, a, subdiagonal, firstBlockColumn, termCount, candidate, candidate,
                         values.subspan(above), productRow);
    if (termCount == 0 || above == 0)
        return;
    const BlockDiagonalView pivots = pivotsOf(a, subdiagonal, firstBlockColumn, termCount);
    productRows.resize(above * termCount);
    for (std::size_t i = 0; i < above; ++i)
    {
        computeProductRow(a, firstRow + i, firstBlockColumn, pivots, std::span<double>(productRow.data(), termCount));
        for (std::size_t p = 0; p < termCount; ++p)
            productRows[i + p * above] = productRow[p];
    }
    for (std::size_t p = 0; p < termCount; ++p)
        productRow[p] = a(candidate, firstBlockColumn + p);
    ColumnProductOperands operands;
    operands.y = values.data();
    operands.a = productRows.data();
    operands.lda = above;
    operands.v = productRow.data();
    operands.rowCount = above;
    operands.innerCount = termCount;
    family.subtractColumnProducts(operands);
}

// The largest |values[i]| for i >= first, and its first index (or values.size() when there is none).
struct ColumnMaximum
{
    double value = 0.0;
    std::size_t index = 0;
};

ColumnMaximum findMaximum(std::span<const double> values, std::size_t first, std::size_t skipped)
{
    ColumnMaximum maximum{0.0, values.size()};
    for (std::size_t i = first; i < values.size(); ++i)
    {
        if (i != skipped && std::abs(values[i]) > maximum.value)
        {
            maximum.value = std::abs(values[i]);
            maximum.index = i;
        }
    }
    return maximum;
}

} // namespace

std::size_t BlockedLdlt::factorWithoutPivoting(DenseMatrixView a, std::size_t pivotColumnCount) const
{
    const std::size_t n = a.rows;
    std::vector<double> productRow(LDLT_BLOCK_COLUMNS);
    const std::size_t eliminated = factorColumns(a, 0, pivotColumnCount, productRow);
    if (eliminated < pivotColumnCount || pivotColumnCount == n)
        return eliminated;
    // The trailing block (the Schur complement of a partial factorisation) receives the terms of every eliminated
    // column at once, p increasing.
    throwIfCancelled(m_kernels.options().cancellation);
    m_kernels.updateSymmetric(a.block(pivotColumnCount, pivotColumnCount, n - pivotColumnCount, n - pivotColumnCount),
                              a.block(pivotColumnCount, 0, n - pivotColumnCount, pivotColumnCount),
                              pivotsOf(a, nullptr, 0, pivotColumnCount));
    return pivotColumnCount;
}

std::size_t BlockedLdlt::factorColumns(DenseMatrixView a, std::size_t first, std::size_t end,
                                       std::vector<double>& productRow) const
{
    // PERFORMANCE: k-blocking (the recursive LDLᵀ of Gustavson, "Recursion leads to automatic variable blocking for
    // dense linear-algebra algorithms", IBM J. Res. Dev. 41, 1997, with the blocks of LDLT_BLOCK_COLUMNS at the
    // leaves). The columns after a block of 96 received its terms block after block before (one sweep of the later
    // columns of a panel per block, k = 96 each); here the right half of a part receives the terms of its whole left
    // half in one update (k up to half the panel's columns), so the panel is swept about log2(c/96) times and the
    // micro-kernels run on full inner blocks, with few large tiled calls for the threads. Every element still receives
    // its terms in increasing p, the chain of the flat blocking (the same bits).
    // Measured (with the shared panels): the LDLᵀ of 4 000 on one thread 73 → 80 % of the
    // peak, a front of 2 000 rows and 500 columns on six threads 135-145 → 157-167 GFLOP/s.
    const std::size_t width = end - first;
    if (width <= LDLT_BLOCK_COLUMNS)
    {
        throwIfCancelled(m_kernels.options().cancellation);
        return factorBlockColumns(a, first, end, productRow);
    }
    const std::size_t blocks = (width + LDLT_BLOCK_COLUMNS - 1) / LDLT_BLOCK_COLUMNS;
    const std::size_t middle = first + blocks / 2 * LDLT_BLOCK_COLUMNS;
    const std::size_t left = factorColumns(a, first, middle, productRow);
    if (left < middle - first)
        return left;
    // The right part's columns, over all their rows from `middle` down, receive the left part's terms: their diagonal
    // part by the symmetric update, the rows below the part by the rectangular one.
    const std::size_t n = a.rows;
    const BlockDiagonalView pivots = pivotsOf(a, nullptr, first, middle - first);
    const DenseMatrixView rightRows = a.block(middle, first, end - middle, middle - first);
    m_kernels.updateSymmetric(a.block(middle, middle, end - middle, end - middle), rightRows, pivots);
    if (end < n)
        m_kernels.updateRectangular(a.block(end, middle, n - end, end - middle),
                                    a.block(end, first, n - end, middle - first), rightRows, pivots);
    return middle - first + factorColumns(a, middle, end, productRow);
}

std::size_t BlockedLdlt::factorBlocked(DenseMatrixView a, std::size_t pivotColumnCount, std::size_t blockColumns) const
{
    const std::size_t n = a.rows;
    std::vector<double> productRow(blockColumns);
    for (std::size_t firstColumn = 0; firstColumn < pivotColumnCount; firstColumn += blockColumns)
    {
        throwIfCancelled(m_kernels.options().cancellation);
        const std::size_t endColumn = std::min(pivotColumnCount, firstColumn + blockColumns);
        const std::size_t blockWidth = endColumn - firstColumn;
        const std::size_t eliminated = factorBlockColumns(a, firstColumn, endColumn, productRow);
        if (eliminated < blockWidth)
            return firstColumn + eliminated;
        if (endColumn == n)
            continue;
        // The update of the trailing lower triangle by the whole block.
        m_kernels.updateSymmetric(a.block(endColumn, endColumn, n - endColumn, n - endColumn),
                                  a.block(endColumn, firstColumn, n - endColumn, blockWidth),
                                  pivotsOf(a, nullptr, firstColumn, blockWidth));
    }
    return pivotColumnCount;
}

std::size_t BlockedLdlt::factorBlockColumns(DenseMatrixView a, std::size_t firstColumn, std::size_t endColumn,
                                            std::vector<double>& productRow) const
{
    const std::size_t n = a.rows;
    const std::size_t blockWidth = endColumn - firstColumn;
    if (blockWidth > INNER_BLOCK_COLUMNS)
    {
        // The diagonal block by the same algorithm with narrower blocks: the chain of every element is the same.
        const std::size_t eliminated =
            factorBlocked(a.block(firstColumn, firstColumn, blockWidth, blockWidth), blockWidth, INNER_BLOCK_COLUMNS);
        if (eliminated < blockWidth)
            return eliminated;
    }
    else
    {
        // The diagonal block, column by column (left-looking inside the block).
        for (std::size_t k = firstColumn; k < endColumn; ++k)
        {
            subtractBlockColumns(m_family, a, nullptr, firstColumn, k - firstColumn, k, k,
                                 std::span<double>(&a(k, k), endColumn - k), productRow);
            const double pivot = a(k, k);
            if (pivot == 0.0 || !std::isfinite(pivot))
                return k - firstColumn;
            for (std::size_t i = k + 1; i < endColumn; ++i)
                a(i, k) = a(i, k) / pivot;
        }
    }
    // The rows below the block.
    if (endColumn < n)
        m_kernels.solvePanel(a.block(endColumn, firstColumn, n - endColumn, blockWidth),
                             a.block(firstColumn, firstColumn, blockWidth, blockWidth),
                             pivotsOf(a, nullptr, firstColumn, blockWidth));
    return blockWidth;
}

std::size_t BlockedLdlt::factorFront(DenseMatrixView panel, DenseMatrixView contribution) const
{
    // PERFORMANCE: the front in two parts. The panel's columns are eliminated over the panel's rows alone
    // (factorColumns), and the trailing block of the front receives the update of all the columns at once: one sweep
    // of the micro-kernels over it, k = c terms per load and store of each element, instead of one sweep per block of
    // LDLT_BLOCK_COLUMNS; the chain of every element is the same. The front then needs no buffer of its own:
    // the panel is the factor's, the trailing block the update matrix handed to the parent.
    // Measured with the front in place of the multifrontal factorisation (one thread): the shell of 100 000 DOFs
    // 0.52 -> 0.50 s, the 3D grid of 50 000 DOFs 1.64 -> 1.45 s.
    const std::size_t columns = panel.columns;
    const std::size_t n = panel.rows;
    const std::size_t rowsBelow = n - columns;
    std::vector<double> productRow(LDLT_BLOCK_COLUMNS);
    const std::size_t eliminated = factorColumns(panel, 0, columns, productRow);
    if (eliminated < columns)
        return eliminated;
    if (rowsBelow > 0)
    {
        throwIfCancelled(m_kernels.options().cancellation);
        m_kernels.updateSymmetric(contribution, panel.block(columns, 0, rowsBelow, columns),
                                  pivotsOf(panel, nullptr, 0, columns));
    }
    return columns;
}

std::size_t BlockedLdlt::factorPanel(DenseMatrixView panel) const
{
    std::vector<double> productRow(LDLT_BLOCK_COLUMNS);
    return factorColumns(panel, 0, panel.columns, productRow);
}

std::size_t BlockedLdlt::factorWithPivoting(DenseMatrixView a, std::size_t candidateCount,
                                            std::span<std::size_t> permutation, std::span<double> subdiagonal) const
{
    // Bunch-Kaufman partial pivoting by blocks (the scheme of LAPACK dlasyf, with the decisions of dsytf2 and of the
    // reference family): inside a block, the current column k and, when needed, the candidate column r are computed
    // left-looking; the interchange is applied at once to the whole matrix (rows of L already computed and trailing
    // rows and columns, which have all received the updates of the earlier blocks only, so they stay consistent);
    // the trailing update of the block runs after its last column. A block grows by one column when its last pivot
    // is 2 × 2. A delayed candidate (kernels/DenseKernelEntryPoints.hpp, PivotedLdltOperands) is interchanged with
    // the last untried candidate the same way; it then receives the trailing updates like any other trailing column.
    const std::size_t n = a.rows;
    for (std::size_t i = 0; i < candidateCount; ++i)
    {
        permutation[i] = i;
        subdiagonal[i] = 0.0;
    }
    std::vector<double> columnK(n);
    std::vector<double> columnR(n);
    std::vector<double> productRow(PIVOTED_BLOCK_COLUMNS + 1);
    std::vector<double> productRows;

    std::size_t k = 0;
    std::size_t end = candidateCount; // untried candidates [k, end); delayed ones [end, candidateCount)
    std::size_t passStart = 0;
    while (true)
    {
        while (k < end)
        {
            throwIfCancelled(m_kernels.options().cancellation);
            const std::size_t firstColumn = k;
            while (k < end && k - firstColumn < PIVOTED_BLOCK_COLUMNS)
            {
                const std::size_t rows = n - k;
                const std::span<double> valuesK(columnK.data(), rows);
                for (std::size_t i = 0; i < rows; ++i)
                    valuesK[i] = a(k + i, k);
                subtractBlockColumns(m_family, a, subdiagonal.data(), firstColumn, k - firstColumn, k, k, valuesK,
                                     productRow);

                const double absakk = std::abs(valuesK[0]);
                const ColumnMaximum colmax = findMaximum(valuesK, 1, rows);
                if (absakk == 0.0 && colmax.value == 0.0)
                {
                    // A zero column: a zero 1 × 1 pivot, L(:, k) = 0.
                    for (std::size_t i = 0; i < rows; ++i)
                        a(k + i, k) = 0.0;
                    k += 1;
                    continue;
                }

                std::size_t blockSize = 1;
                std::size_t swapWith = k;
                const std::span<double> valuesR(columnR.data(), rows);
                if (absakk < BUNCH_KAUFMAN_ALPHA * colmax.value && k + colmax.index >= end)
                {
                    if (absakk < DELAYED_PIVOT_THRESHOLD * colmax.value)
                    {
                        --end;
                        if (k != end)
                        {
                            swapSymmetric(a, k, end);
                            std::swap(permutation[k], permutation[end]);
                        }
                        continue;
                    }
                }
                else if (absakk < BUNCH_KAUFMAN_ALPHA * colmax.value)
                {
                    const std::size_t r = k + colmax.index;
                    for (std::size_t i = 0; i < rows; ++i)
                    {
                        const std::size_t row = k + i;
                        valuesR[i] = row < r ? a(r, row) : a(row, r);
                    }
                    subtractBlockColumnsFromCandidate(m_family, a, subdiagonal.data(), firstColumn, k, r, valuesR,
                                                      productRow, productRows);
                    const double rowmax = findMaximum(valuesR, 0, colmax.index).value;
                    if (absakk >= BUNCH_KAUFMAN_ALPHA * colmax.value * (colmax.value / rowmax))
                        swapWith = k;
                    else if (std::abs(valuesR[colmax.index]) >= BUNCH_KAUFMAN_ALPHA * rowmax)
                        swapWith = r;
                    else
                    {
                        swapWith = r;
                        blockSize = 2;
                    }
                }

                const std::size_t moved = k + blockSize - 1;
                if (swapWith != moved)
                {
                    swapSymmetric(a, moved, swapWith);
                    std::swap(permutation[moved], permutation[swapWith]);
                    // The computed columns follow the interchange of rows `moved` and `swapWith`.
                    std::swap(valuesR[moved - k], valuesR[swapWith - k]);
                    if (blockSize == 1)
                        std::copy(valuesR.begin(), valuesR.end(), valuesK.begin());
                    else
                        std::swap(valuesK[moved - k], valuesK[swapWith - k]);
                }

                if (blockSize == 1)
                {
                    const double pivot = valuesK[0];
                    a(k, k) = pivot;
                    for (std::size_t i = 1; i < rows; ++i)
                        a(k + i, k) = valuesK[i] / pivot;
                }
                else
                {
                    const double d0 = valuesK[0];
                    const double e = valuesK[1];
                    const double d1 = valuesR[1];
                    const TwoByTwoInverse inverse = invertTwoByTwo(d0, e, d1);
                    for (std::size_t i = 2; i < rows; ++i)
                    {
                        const double r0 = valuesK[i];
                        const double r1 = valuesR[i];
                        a(k + i, k) = inverse.s * (inverse.d11 * r0 - r1);
                        a(k + i, k + 1) = inverse.s * (inverse.d22 * r1 - r0);
                    }
                    a(k, k) = d0;
                    a(k + 1, k) = 0.0;
                    a(k + 1, k + 1) = d1;
                    subdiagonal[k] = e;
                }
                k += blockSize;
            }
            if (k < n)
                m_kernels.updateSymmetric(a.block(k, k, n - k, n - k), a.block(k, firstColumn, n - k, k - firstColumn),
                                          pivotsOf(a, subdiagonal.data(), firstColumn, k - firstColumn));
        }
        // Retry the delayed candidates after the pivots of this pass, as long as a pass eliminates something.
        if (end == candidateCount || k == passStart)
            break;
        passStart = k;
        end = candidateCount;
    }
    return k;
}

} // namespace oos::linalg
