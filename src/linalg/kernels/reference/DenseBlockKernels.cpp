// The scalar reference family of the dense kernels: the textbook algorithms, unblocked, with a fixed loop order, the
// oracle of the SIMD families (the factorisation of a front in two parts by groups of columns and register tiles, each
// element keeping the chain of the unblocked algorithm: factorFrontWithoutPivoting). Compiled with -ffp-contract=off
// (cmake/OosKernels.cmake): every product is rounded before it is added, so the results are the same bits on
// every machine that follows IEEE 754 double arithmetic. The chains of terms are those of
// kernels/DenseKernelEntryPoints.hpp.

#include "oos/linalg/kernels/DenseKernelEntryPoints.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace oos::linalg::kernels::reference
{
namespace
{

double readA(const ProductTileOperands& operands, std::size_t i, std::size_t p)
{
    return operands.isATransposed ? operands.a[p + i * operands.lda] : operands.a[i + p * operands.lda];
}

double readB(const ProductTileOperands& operands, std::size_t j, std::size_t p)
{
    return operands.isBTransposed ? operands.b[p + j * operands.ldb] : operands.b[j + p * operands.ldb];
}

// Ŵ(j, p) = (B̂·D)(j, p) by the formula of ProductTileOperands.
double readW(const ProductTileOperands& operands, std::size_t j, std::size_t p)
{
    const BlockDiagonalView& d = operands.d;
    if (d.diagonal == nullptr)
        return readB(operands, j, p);
    const double dp = d.diagonal[p * d.diagonalStride];
    if (d.subdiagonal != nullptr)
    {
        if (p + 1 < d.size && d.subdiagonal[p] != 0.0)
            return readB(operands, j, p) * dp + readB(operands, j, p + 1) * d.subdiagonal[p];
        if (p > 0 && d.subdiagonal[p - 1] != 0.0)
            return readB(operands, j, p - 1) * d.subdiagonal[p - 1] + readB(operands, j, p) * dp;
    }
    return readB(operands, j, p) * dp;
}

bool startsTwoByTwo(const BlockDiagonalView& d, std::size_t p)
{
    return d.subdiagonal != nullptr && p + 1 < d.size && d.subdiagonal[p] != 0.0;
}

// The coefficients of the inverse of a 2 × 2 pivot block [[a, b], [b, c]] in the form of LAPACK dsytf2, which avoids
// forming the determinant a·c − b² (it can underflow or cancel): x0 = s·(d11·r0 − r1), x1 = s·(d22·r1 − r0).
struct TwoByTwoInverse
{
    double d11;
    double d22;
    double s;
};

TwoByTwoInverse invertTwoByTwo(double a, double b, double c)
{
    const double d11 = c / b;
    const double d22 = a / b;
    const double t = 1.0 / (d11 * d22 - 1.0);
    return {d11, d22, t / b};
}

// The symmetric interchange of rows and columns s < t of the lower-stored `a`: the rows s and t of the columns before s
// (the part of L already computed and the trailing rows), the two diagonal terms, the column s and the row t between
// them, and the columns s and t below t. a(t, s) is its own mirror and stays.
void swapSymmetric(double* a, std::size_t lda, std::size_t size, std::size_t s, std::size_t t)
{
    for (std::size_t j = 0; j < s; ++j)
        std::swap(a[s + j * lda], a[t + j * lda]);
    std::swap(a[s + s * lda], a[t + t * lda]);
    for (std::size_t i = s + 1; i < t; ++i)
        std::swap(a[i + s * lda], a[t + i * lda]);
    for (std::size_t i = t + 1; i < size; ++i)
        std::swap(a[i + s * lda], a[i + t * lda]);
}

// The columns of a front's panel eliminated right-looking among themselves before their terms reach the columns after
// them (factorFrontWithoutPivoting).
constexpr std::size_t FRONT_GROUP_COLUMNS = 32;
// The register tiles of subtractLowerTerms: LOWER_TILE_ROWS × LOWER_TILE_COLUMNS elements kept in local accumulators
// over a block of LOWER_TERM_BLOCK terms, whose W is formed once per block of columns.
constexpr std::size_t LOWER_TILE_ROWS = 4;
constexpr std::size_t LOWER_TILE_COLUMNS = 4;
constexpr std::size_t LOWER_TERM_BLOCK = 64;

// The operands of subtractLowerTerms for one block of terms: T(i, j) at t[i + j·ldt], L(i, p) at l[i + p·ldl] for the
// block's terms p = firstTerm + q, and W(j, p) at w[q·LOWER_TILE_COLUMNS + j − firstColumn] for the columns of a tile.
struct LowerTermBlock
{
    double* t;
    std::size_t ldt;
    const double* l;
    std::size_t ldl;
    const double* w;
    std::size_t firstTerm;
    std::size_t termCount;
    std::size_t firstColumn;
};

// The ROWS × WIDTH elements of rows [firstRow, firstRow + ROWS) and the tile's columns, all below the diagonal: each
// accumulator starts from T and receives the block's terms in increasing p, acc = acc − L·W (a rounded product, then
// a rounded subtraction: -ffp-contract=off).
template <std::size_t ROWS, std::size_t WIDTH>
void subtractTileTerms(const LowerTermBlock& block, std::size_t firstRow) noexcept
{
    double acc[WIDTH][ROWS];
    for (std::size_t c = 0; c < WIDTH; ++c)
        for (std::size_t r = 0; r < ROWS; ++r)
            acc[c][r] = block.t[firstRow + r + (block.firstColumn + c) * block.ldt];
    for (std::size_t q = 0; q < block.termCount; ++q)
    {
        const double* lColumn = block.l + firstRow + (block.firstTerm + q) * block.ldl;
        const double* wRow = block.w + q * LOWER_TILE_COLUMNS;
        for (std::size_t c = 0; c < WIDTH; ++c)
            for (std::size_t r = 0; r < ROWS; ++r)
                acc[c][r] -= lColumn[r] * wRow[c];
    }
    for (std::size_t c = 0; c < WIDTH; ++c)
        for (std::size_t r = 0; r < ROWS; ++r)
            block.t[firstRow + r + (block.firstColumn + c) * block.ldt] = acc[c][r];
}

template <std::size_t WIDTH>
void subtractRowsTerms(const LowerTermBlock& block, std::size_t firstRow, std::size_t rowCount) noexcept
{
    std::size_t i = firstRow;
    for (; i + LOWER_TILE_ROWS <= firstRow + rowCount; i += LOWER_TILE_ROWS)
        subtractTileTerms<LOWER_TILE_ROWS, WIDTH>(block, i);
    for (; i < firstRow + rowCount; ++i)
        subtractTileTerms<1, WIDTH>(block, i);
}

// T(i, j) −= Σ_{p ∈ [firstTerm, lastTerm)} L(i, p)·W(j, p), p increasing, for the lower trapezoid of T (columns
// j < columnCount, rows j ≤ i < rowCount), with L(i, p) = l[i + p·ldl] and W(j, p) = L(j, p)·d_p, d_p =
// front[p + p·ldf]: the terms factorFrontWithoutPivoting's unblocked algorithm subtracts from these elements, with
// the same operations in the same order (the blocks of terms only store an element and load it back).
void subtractLowerTerms(double* t, std::size_t ldt, std::size_t rowCount, std::size_t columnCount, const double* l,
                        std::size_t ldl, const double* front, std::size_t ldf, std::size_t firstTerm,
                        std::size_t lastTerm) noexcept
{
    std::array<double, LOWER_TERM_BLOCK * LOWER_TILE_COLUMNS> w{};
    for (std::size_t p0 = firstTerm; p0 < lastTerm; p0 += LOWER_TERM_BLOCK)
    {
        const std::size_t termCount = std::min(LOWER_TERM_BLOCK, lastTerm - p0);
        for (std::size_t j0 = 0; j0 < columnCount; j0 += LOWER_TILE_COLUMNS)
        {
            const std::size_t width = std::min(LOWER_TILE_COLUMNS, columnCount - j0);
            for (std::size_t q = 0; q < termCount; ++q)
            {
                const std::size_t p = p0 + q;
                const double pivot = front[p + p * ldf];
                for (std::size_t c = 0; c < width; ++c)
                    w[q * LOWER_TILE_COLUMNS + c] = l[j0 + c + p * ldl] * pivot;
            }
            const LowerTermBlock block{t, ldt, l, ldl, w.data(), p0, termCount, j0};
            // The tile's diagonal block element by element (on and below the diagonal only), then the rows below it.
            for (std::size_t i = j0; i < j0 + width && i < rowCount; ++i)
                for (std::size_t c = 0; c < width && j0 + c <= i; ++c)
                {
                    double& target = t[i + (j0 + c) * ldt];
                    double acc = target;
                    for (std::size_t q = 0; q < termCount; ++q)
                        acc -= l[i + (p0 + q) * ldl] * w[q * LOWER_TILE_COLUMNS + c];
                    target = acc;
                }
            const std::size_t below = j0 + width;
            const std::size_t rowsBelow = rowCount > below ? rowCount - below : 0;
            switch (width)
            {
            case 1:
                subtractRowsTerms<1>(block, below, rowsBelow);
                break;
            case 2:
                subtractRowsTerms<2>(block, below, rowsBelow);
                break;
            case 3:
                subtractRowsTerms<3>(block, below, rowsBelow);
                break;
            default:
                subtractRowsTerms<LOWER_TILE_COLUMNS>(block, below, rowsBelow);
                break;
            }
        }
    }
}

} // namespace

void subtractProductTile(const ProductTileOperands& operands, const PackingWorkspace& /*workspace*/) noexcept
{
    // Loop order j, p, i: each column of C receives the terms one after the other, so every c(i, j) is summed in the
    // order of p; the i loop runs over independent elements.
    const std::size_t k = operands.innerCount;
    for (std::size_t j = 0; j < operands.columnCount; ++j)
    {
        std::size_t firstRow = 0;
        if (operands.isLowerOnly)
        {
            // i + diagonalOffset >= j.
            const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(j) - operands.diagonalOffset;
            firstRow = limit > 0 ? static_cast<std::size_t>(limit) : 0;
        }
        double* cColumn = operands.c + j * operands.ldc;
        for (std::size_t t = 0; t < k; ++t)
        {
            const std::size_t p = operands.isInnerReversed ? k - 1 - t : t;
            const double w = readW(operands, j, p);
            for (std::size_t i = firstRow; i < operands.rowCount; ++i)
                cColumn[i] -= readA(operands, i, p) * w;
        }
    }
}

void subtractColumnProducts(const ColumnProductOperands& operands) noexcept
{
    for (std::size_t p = 0; p < operands.innerCount; ++p)
    {
        const double v = operands.v[p];
        const double* aColumn = operands.a + p * operands.lda;
        for (std::size_t i = 0; i < operands.rowCount; ++i)
            operands.y[i] -= aColumn[i] * v;
    }
}

void solvePanel(const PanelSolveOperands& operands) noexcept
{
    const std::size_t m = operands.rowCount;
    const auto column = [&](std::size_t j) { return operands.p + j * operands.ldp; };
    const auto subtractSolvedColumns = [&](std::size_t j, std::size_t count)
    {
        double* target = column(j);
        for (std::size_t q = 0; q < count; ++q)
        {
            const double w = operands.w[j + q * operands.ldw];
            const double* solved = column(q);
            for (std::size_t i = 0; i < m; ++i)
                target[i] -= solved[i] * w;
        }
    };
    std::size_t j = 0;
    while (j < operands.columnCount)
    {
        const BlockDiagonalView& d = operands.d;
        if (startsTwoByTwo(d, j))
        {
            subtractSolvedColumns(j, j);
            subtractSolvedColumns(j + 1, j);
            const TwoByTwoInverse inverse = invertTwoByTwo(d.diagonal[j * d.diagonalStride], d.subdiagonal[j],
                                                           d.diagonal[(j + 1) * d.diagonalStride]);
            double* first = column(j);
            double* second = column(j + 1);
            for (std::size_t i = 0; i < m; ++i)
            {
                const double r0 = first[i];
                const double r1 = second[i];
                first[i] = inverse.s * (inverse.d11 * r0 - r1);
                second[i] = inverse.s * (inverse.d22 * r1 - r0);
            }
            j += 2;
        }
        else
        {
            subtractSolvedColumns(j, j);
            const double pivot = d.diagonal[j * d.diagonalStride];
            double* target = column(j);
            for (std::size_t i = 0; i < m; ++i)
                target[i] = target[i] / pivot;
            j += 1;
        }
    }
}

void solveUnitLowerBlock(const TriangleSolveOperands& operands) noexcept
{
    // Column sweep: once x(q) is final it is subtracted from the rows below, so each x(i) receives its terms in
    // increasing q.
    for (std::size_t c = 0; c < operands.rhsCount; ++c)
    {
        double* x = operands.b + c * operands.ldb;
        for (std::size_t q = 0; q < operands.size; ++q)
        {
            const double xq = x[q];
            const double* lColumn = operands.l + q * operands.ldl;
            for (std::size_t i = q + 1; i < operands.size; ++i)
                x[i] -= lColumn[i] * xq;
        }
    }
}

void solveUnitLowerTransposedBlock(const TriangleSolveOperands& operands) noexcept
{
    // Row by row from the last: x(i) = b(i) − Σ l(q, i)·x(q), q from the last row down to i + 1.
    for (std::size_t c = 0; c < operands.rhsCount; ++c)
    {
        double* x = operands.b + c * operands.ldb;
        for (std::size_t i = operands.size; i-- > 0;)
        {
            const double* lColumn = operands.l + i * operands.ldl;
            double sum = x[i];
            for (std::size_t q = operands.size; q-- > i + 1;)
                sum -= lColumn[q] * x[q];
            x[i] = sum;
        }
    }
}

void subtractRowProducts(const RowProductOperands& operands) noexcept
{
    // Target by target, term by term: each target row receives its terms in increasing p, the columns of a row being
    // independent elements.
    for (std::size_t i = 0; i < operands.targetCount; ++i)
    {
        const std::ptrdiff_t targetAt = static_cast<std::ptrdiff_t>(i) * operands.targetStep;
        double* target = operands.x + static_cast<std::size_t>(operands.targets[targetAt]) * operands.ldx;
        const double* lRow = operands.l + static_cast<std::ptrdiff_t>(i) * operands.targetStride;
        for (std::size_t p = 0; p < operands.termCount; ++p)
        {
            const std::ptrdiff_t sourceAt = static_cast<std::ptrdiff_t>(p) * operands.sourceStep;
            const double* source = operands.x + static_cast<std::size_t>(operands.sources[sourceAt]) * operands.ldx;
            const double value = lRow[static_cast<std::ptrdiff_t>(p) * operands.termStride];
            for (std::size_t c = 0; c < operands.width; ++c)
                target[c] -= value * source[c];
        }
    }
}

void solveRowTriangle(const RowTriangleOperands& operands) noexcept
{
    const auto row = [&](std::size_t i)
    { return operands.x + static_cast<std::size_t>(operands.rows[i]) * operands.ldx; };
    if (!operands.isTransposed)
    {
        // Row i receives the terms of the rows before it, q increasing.
        for (std::size_t i = 1; i < operands.size; ++i)
        {
            double* target = row(i);
            for (std::size_t q = 0; q < i; ++q)
            {
                const double value = operands.l[i + q * operands.ldl];
                const double* solved = row(q);
                for (std::size_t c = 0; c < operands.width; ++c)
                    target[c] -= value * solved[c];
            }
        }
        return;
    }
    // From the last row up: row i receives the terms of the rows after it, q decreasing.
    for (std::size_t i = operands.size; i-- > 0;)
    {
        double* target = row(i);
        const double* lColumn = operands.l + i * operands.ldl;
        for (std::size_t q = operands.size; q-- > i + 1;)
        {
            const double value = lColumn[q];
            const double* solved = row(q);
            for (std::size_t c = 0; c < operands.width; ++c)
                target[c] -= value * solved[c];
        }
    }
}

std::size_t factorWithoutPivoting(const LdltOperands& operands) noexcept
{
    // Right-looking LDLᵀ (Golub & Van Loan Algorithm 4.1.2, outer product form): column k is scaled by its pivot, then
    // its rank-one term L(:, k)·d_k·L(:, k)ᵀ is subtracted from the trailing lower triangle at once. Element (i, j)
    // thus receives the terms k = 0, 1, ... in increasing order, each L(i, k)·W(j, k) with W(j, k) = L(j, k)·d_k.
    double* a = operands.a;
    const std::size_t lda = operands.lda;
    const std::size_t n = operands.size;
    for (std::size_t k = 0; k < operands.pivotColumnCount; ++k)
    {
        const double pivot = a[k + k * lda];
        if (pivot == 0.0 || !std::isfinite(pivot))
            return k;
        double* lColumn = a + k * lda;
        for (std::size_t i = k + 1; i < n; ++i)
            lColumn[i] = lColumn[i] / pivot;
        for (std::size_t j = k + 1; j < n; ++j)
        {
            const double w = lColumn[j] * pivot;
            double* target = a + j * lda;
            for (std::size_t i = j; i < n; ++i)
                target[i] -= lColumn[i] * w;
        }
    }
    return operands.pivotColumnCount;
}

std::size_t factorFrontWithoutPivoting(const SplitLdltOperands& operands) noexcept
{
    // factorWithoutPivoting on the panel's columns, then the terms of the c columns on the trailing block, k
    // increasing: column k of L and d_k are final once step k is done, so each element of the trailing block receives
    // the operations of the whole front's factorisation in their order (the same bits), and none when a pivot is zero.
    //
    // PERFORMANCE: the panel's columns are eliminated by groups of FRONT_GROUP_COLUMNS: each group right-looking on its
    // own columns, then its terms applied to the columns after it, and the terms of every column to the trailing block
    // at the end, by register tiles (subtractLowerTerms). Each element still receives its terms one by one in
    // increasing k, each product rounded before it is subtracted (the chains of the unblocked algorithm, the same
    // bits); the rank-one updates of the unblocked algorithm read and wrote the whole trailing triangle for every
    // column.
    // Measured: the full base of a stiffened barrel panel model (10 980 equations, 100 cases) on one thread,
    // the numeric factorisation 0.81 → 0.38 s.
    double* panel = operands.panel;
    const std::size_t ldp = operands.ldp;
    const std::size_t c = operands.pivotColumnCount;
    const std::size_t n = c + operands.rowCount;
    for (std::size_t first = 0; first < c; first += FRONT_GROUP_COLUMNS)
    {
        const std::size_t last = std::min(c, first + FRONT_GROUP_COLUMNS);
        for (std::size_t k = first; k < last; ++k)
        {
            const double pivot = panel[k + k * ldp];
            if (pivot == 0.0 || !std::isfinite(pivot))
            {
                // The columns after the group get the terms of the group's eliminated columns, as the unblocked
                // algorithm leaves them when it stops at k.
                subtractLowerTerms(panel + last + last * ldp, ldp, n - last, c - last, panel + last, ldp, panel, ldp,
                                   first, k);
                return k;
            }
            double* lColumn = panel + k * ldp;
            for (std::size_t i = k + 1; i < n; ++i)
                lColumn[i] = lColumn[i] / pivot;
            for (std::size_t j = k + 1; j < last; ++j)
            {
                const double w = lColumn[j] * pivot;
                double* target = panel + j * ldp;
                for (std::size_t i = j; i < n; ++i)
                    target[i] -= lColumn[i] * w;
            }
        }
        subtractLowerTerms(panel + last + last * ldp, ldp, n - last, c - last, panel + last, ldp, panel, ldp, first,
                           last);
    }
    if (operands.contribution == nullptr)
        return c;
    subtractLowerTerms(operands.contribution, operands.ldc, operands.rowCount, operands.rowCount, panel + c, ldp, panel,
                       ldp, 0, c);
    return c;
}

std::size_t factorWithPivoting(const PivotedLdltOperands& operands) noexcept
{
    // Bunch-Kaufman partial pivoting, unblocked and right-looking (LAPACK dsytf2, lower): at column k, with
    // colmax = max |a(i, k)| over i > k (row imax) and rowmax = max |a(imax, j)| over j != imax:
    // a 1 × 1 pivot on k if |a(k, k)| >= α·colmax or |a(k, k)|·rowmax >= α·colmax²; else a 1 × 1 pivot on imax if
    // |a(imax, imax)| >= α·rowmax; else a 2 × 2 pivot on (k, imax). Ties of the maxima go to the first index.
    // Restricted to the untried candidates [k, end) (PivotedLdltOperands): when imax is not one of them, a 1 × 1 pivot
    // on k if |a(k, k)| >= u·colmax (DELAYED_PIVOT_THRESHOLD), otherwise k is delayed: swapped with the last untried
    // candidate.
    double* a = operands.a;
    const std::size_t lda = operands.lda;
    const std::size_t n = operands.size;
    const std::size_t candidates = operands.candidateCount;
    const auto at = [&](std::size_t i, std::size_t j) -> double& { return a[i + j * lda]; };
    for (std::size_t i = 0; i < candidates; ++i)
    {
        operands.permutation[i] = i;
        operands.subdiagonal[i] = 0.0;
    }

    std::size_t k = 0;
    std::size_t end = candidates; // untried candidates [k, end); delayed ones [end, candidates)
    std::size_t passStart = 0;
    while (true)
    {
        while (k < end)
        {
            const double absakk = std::abs(at(k, k));
            double colmax = 0.0;
            std::size_t imax = k;
            for (std::size_t i = k + 1; i < n; ++i)
            {
                if (std::abs(at(i, k)) > colmax)
                {
                    colmax = std::abs(at(i, k));
                    imax = i;
                }
            }
            if (absakk == 0.0 && colmax == 0.0)
            {
                // A zero column: a zero pivot, nothing to eliminate (L(:, k) = 0).
                k += 1;
                continue;
            }

            std::size_t blockSize = 1;
            std::size_t swapWith = k;
            if (absakk < BUNCH_KAUFMAN_ALPHA * colmax && imax >= end)
            {
                if (absakk < DELAYED_PIVOT_THRESHOLD * colmax)
                {
                    --end;
                    if (k != end)
                    {
                        swapSymmetric(a, lda, n, k, end);
                        std::swap(operands.permutation[k], operands.permutation[end]);
                    }
                    continue;
                }
            }
            else if (absakk < BUNCH_KAUFMAN_ALPHA * colmax)
            {
                double rowmax = 0.0;
                for (std::size_t j = k; j < imax; ++j)
                    rowmax = std::fmax(rowmax, std::abs(at(imax, j)));
                for (std::size_t i = imax + 1; i < n; ++i)
                    rowmax = std::fmax(rowmax, std::abs(at(i, imax)));
                if (absakk >= BUNCH_KAUFMAN_ALPHA * colmax * (colmax / rowmax))
                    swapWith = k;
                else if (std::abs(at(imax, imax)) >= BUNCH_KAUFMAN_ALPHA * rowmax)
                    swapWith = imax;
                else
                {
                    swapWith = imax;
                    blockSize = 2;
                }
            }

            const std::size_t moved = k + blockSize - 1;
            if (swapWith != moved)
            {
                swapSymmetric(a, lda, n, moved, swapWith);
                std::swap(operands.permutation[moved], operands.permutation[swapWith]);
            }

            if (blockSize == 1)
            {
                const double pivot = at(k, k);
                for (std::size_t i = k + 1; i < n; ++i)
                    at(i, k) = at(i, k) / pivot;
                for (std::size_t j = k + 1; j < n; ++j)
                {
                    const double w = at(j, k) * pivot;
                    for (std::size_t i = j; i < n; ++i)
                        at(i, j) -= at(i, k) * w;
                }
            }
            else
            {
                const double d0 = at(k, k);
                const double e = at(k + 1, k);
                const double d1 = at(k + 1, k + 1);
                const TwoByTwoInverse inverse = invertTwoByTwo(d0, e, d1);
                for (std::size_t i = k + 2; i < n; ++i)
                {
                    const double r0 = at(i, k);
                    const double r1 = at(i, k + 1);
                    at(i, k) = inverse.s * (inverse.d11 * r0 - r1);
                    at(i, k + 1) = inverse.s * (inverse.d22 * r1 - r0);
                }
                operands.subdiagonal[k] = e;
                at(k + 1, k) = 0.0;
                for (std::size_t j = k + 2; j < n; ++j)
                {
                    const double w0 = at(j, k) * d0 + at(j, k + 1) * e;
                    const double w1 = at(j, k) * e + at(j, k + 1) * d1;
                    for (std::size_t i = j; i < n; ++i)
                    {
                        at(i, j) -= at(i, k) * w0;
                        at(i, j) -= at(i, k + 1) * w1;
                    }
                }
            }
            k += blockSize;
        }
        // Retry the delayed candidates after the pivots of this pass, as long as a pass eliminates something.
        if (end == candidates || k == passStart)
            break;
        passStart = k;
        end = candidates;
    }
    return k;
}

} // namespace oos::linalg::kernels::reference
