// The AVX2/FMA family, compiled with -mavx2 -mfma (cmake/OosKernels.cmake). Only this folder and avx512/ may use
// SIMD intrinsics. Every helper has internal linkage: nothing compiled here with AVX2 may be shared
// with the baseline code (see kernels/MatrixProduct.hpp).

#include "oos/linalg/kernels/MatrixProduct.hpp"

#include <immintrin.h>

namespace oos::linalg::kernels::avx2
{
namespace
{

constexpr std::size_t LANES = 4;   // doubles per 256-bit register
constexpr int MAX_ROW_VECTORS = 3; // rows of a register block: 3 registers of 4 = 12
constexpr std::size_t BLOCK_COLUMNS = 4;
constexpr std::size_t BLOCK_ROWS = LANES * MAX_ROW_VECTORS;

// The mask of the first `activeLanes` lanes (1 to 4) for the masked loads and stores of the last rows of C.
__m256i makeLaneMask(std::size_t activeLanes)
{
    const __m256i laneIndices = _mm256_set_epi64x(3, 2, 1, 0);
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x(static_cast<long long>(activeLanes)), laneIndices);
}

// PERFORMANCE: register blocking. A block of RowVectors·4 rows × Columns columns of C stays in registers while the
// whole inner dimension is swept (3 × 4 = 12 accumulators, 3 loads of A and 1 broadcast of B: the 16 YMM registers),
// so that each element of A loaded feeds Columns fused multiply-adds and each broadcast of B feeds RowVectors of them.
// The edge block (IsRowEdge) masks the lanes of its last register beyond the last row; the sequence of operations on
// every element is the same as in a full block, so the bits do not depend on the blocking.
// Measured (i5-12400F, one thread, square products): 30 GFLOP/s on 64 × 64 and 23 on 512 × 512, against 13
// and 12 for the reference loop (which GCC vectorises with SSE2).
template <int RowVectors, std::size_t Columns, bool IsRowEdge>
void multiplyBlock(const MatrixProductOperands& operands, std::size_t row, std::size_t column, __m256i lastMask)
{
    __m256d accumulators[RowVectors][Columns];
    for (int r = 0; r < RowVectors; ++r)
        for (std::size_t c = 0; c < Columns; ++c)
            accumulators[r][c] = _mm256_setzero_pd();

    const double* aBlock = operands.a + row;
    const double* bBlock = operands.b + column * operands.ldb;
    for (std::size_t p = 0; p < operands.innerCount; ++p)
    {
        const double* aColumn = aBlock + p * operands.lda;
        __m256d aValues[RowVectors];
        for (int r = 0; r < RowVectors; ++r)
        {
            if (IsRowEdge && r == RowVectors - 1)
                aValues[r] = _mm256_maskload_pd(aColumn + LANES * r, lastMask);
            else
                aValues[r] = _mm256_loadu_pd(aColumn + LANES * r);
        }
        for (std::size_t c = 0; c < Columns; ++c)
        {
            const __m256d bValue = _mm256_broadcast_sd(bBlock + p + c * operands.ldb);
            for (int r = 0; r < RowVectors; ++r)
                accumulators[r][c] = _mm256_fmadd_pd(aValues[r], bValue, accumulators[r][c]);
        }
    }

    for (std::size_t c = 0; c < Columns; ++c)
    {
        double* cColumn = operands.c + row + (column + c) * operands.ldc;
        for (int r = 0; r < RowVectors; ++r)
        {
            if (IsRowEdge && r == RowVectors - 1)
                _mm256_maskstore_pd(cColumn + LANES * r, lastMask, accumulators[r][c]);
            else
                _mm256_storeu_pd(cColumn + LANES * r, accumulators[r][c]);
        }
    }
}

// One horizontal panel of C (its rows `row` to `row + RowVectors·4`, the last register masked when IsRowEdge),
// column block by column block, the last 1 to 3 columns with a narrower block.
template <int RowVectors, bool IsRowEdge>
void multiplyRowPanel(const MatrixProductOperands& operands, std::size_t row, __m256i lastMask)
{
    std::size_t column = 0;
    for (; column + BLOCK_COLUMNS <= operands.columnCount; column += BLOCK_COLUMNS)
        multiplyBlock<RowVectors, BLOCK_COLUMNS, IsRowEdge>(operands, row, column, lastMask);
    switch (operands.columnCount - column)
    {
    case 3:
        multiplyBlock<RowVectors, 3, IsRowEdge>(operands, row, column, lastMask);
        break;
    case 2:
        multiplyBlock<RowVectors, 2, IsRowEdge>(operands, row, column, lastMask);
        break;
    case 1:
        multiplyBlock<RowVectors, 1, IsRowEdge>(operands, row, column, lastMask);
        break;
    default:
        break;
    }
}

void fillWithZeros(const MatrixProductOperands& operands)
{
    for (std::size_t j = 0; j < operands.columnCount; ++j)
        for (std::size_t i = 0; i < operands.rowCount; ++i)
            operands.c[i + j * operands.ldc] = 0.0;
}

} // namespace

void multiplyMatrices(const MatrixProductOperands& operands) noexcept
{
    if (operands.innerCount == 0)
    {
        fillWithZeros(operands);
        return;
    }
    const __m256i fullMask = _mm256_set1_epi64x(-1);
    std::size_t row = 0;
    for (; row + BLOCK_ROWS <= operands.rowCount; row += BLOCK_ROWS)
        multiplyRowPanel<MAX_ROW_VECTORS, false>(operands, row, fullMask);

    const std::size_t remainingRows = operands.rowCount - row;
    if (remainingRows == 0)
        return;
    const std::size_t rowVectors = (remainingRows + LANES - 1) / LANES;
    const __m256i lastMask = makeLaneMask(remainingRows - LANES * (rowVectors - 1));
    switch (rowVectors)
    {
    case 3:
        multiplyRowPanel<3, true>(operands, row, lastMask);
        break;
    case 2:
        multiplyRowPanel<2, true>(operands, row, lastMask);
        break;
    default:
        multiplyRowPanel<1, true>(operands, row, lastMask);
        break;
    }
}

} // namespace oos::linalg::kernels::avx2
