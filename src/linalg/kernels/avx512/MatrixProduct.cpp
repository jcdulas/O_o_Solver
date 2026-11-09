// The AVX-512 family, compiled with -mavx512f -mavx512dq -mavx512vl (cmake/OosKernels.cmake). Only this folder
// and avx2/ may use SIMD intrinsics. Every helper has internal linkage: nothing compiled here may be
// shared with the baseline code (see kernels/MatrixProduct.hpp).
//
// Not validated on the reference PC (no AVX-512): its correctness test runs on a machine that has AVX-512 and is
// skipped elsewhere.

#include "oos/linalg/kernels/MatrixProduct.hpp"

#include <immintrin.h>

namespace oos::linalg::kernels::avx512
{
namespace
{

constexpr std::size_t LANES = 8;   // doubles per 512-bit register
constexpr int MAX_ROW_VECTORS = 3; // rows of a register block: 3 registers of 8 = 24
constexpr std::size_t BLOCK_COLUMNS = 8;
constexpr std::size_t BLOCK_ROWS = LANES * MAX_ROW_VECTORS;
constexpr __mmask8 FULL_MASK = 0xFF;

// Register blocking, as the AVX2 family: 3 × 8 = 24 accumulators of the 32 ZMM registers, 3 loads of A and 1
// broadcast of B per step of the inner dimension. The lanes beyond the last row are masked in the edge block; every
// element follows the same sequence of fused multiply-adds, so the bits do not depend on the blocking.
// TODO(AVX-512 validation): measure this blocking on a processor with AVX-512 and mark it
// PERFORMANCE/Measured then; acceptable now because the family is not_validated and never chosen automatically here.
template <int RowVectors, std::size_t Columns>
void multiplyBlock(const MatrixProductOperands& operands, std::size_t row, std::size_t column, __mmask8 lastMask)
{
    __m512d accumulators[RowVectors][Columns];
    for (int r = 0; r < RowVectors; ++r)
        for (std::size_t c = 0; c < Columns; ++c)
            accumulators[r][c] = _mm512_setzero_pd();

    const double* aBlock = operands.a + row;
    const double* bBlock = operands.b + column * operands.ldb;
    for (std::size_t p = 0; p < operands.innerCount; ++p)
    {
        const double* aColumn = aBlock + p * operands.lda;
        __m512d aValues[RowVectors];
        for (int r = 0; r < RowVectors; ++r)
            aValues[r] = _mm512_maskz_loadu_pd(r == RowVectors - 1 ? lastMask : FULL_MASK, aColumn + LANES * r);
        for (std::size_t c = 0; c < Columns; ++c)
        {
            const __m512d bValue = _mm512_set1_pd(bBlock[p + c * operands.ldb]);
            for (int r = 0; r < RowVectors; ++r)
                accumulators[r][c] = _mm512_fmadd_pd(aValues[r], bValue, accumulators[r][c]);
        }
    }

    for (std::size_t c = 0; c < Columns; ++c)
    {
        double* cColumn = operands.c + row + (column + c) * operands.ldc;
        for (int r = 0; r < RowVectors; ++r)
            _mm512_mask_storeu_pd(cColumn + LANES * r, r == RowVectors - 1 ? lastMask : FULL_MASK, accumulators[r][c]);
    }
}

template <int RowVectors, std::size_t Columns>
void multiplyColumnRemainder(const MatrixProductOperands& operands, std::size_t row, std::size_t column,
                             std::size_t remainingColumns, __mmask8 lastMask)
{
    if constexpr (Columns > 0)
    {
        if (remainingColumns == Columns)
            multiplyBlock<RowVectors, Columns>(operands, row, column, lastMask);
        else
            multiplyColumnRemainder<RowVectors, Columns - 1>(operands, row, column, remainingColumns, lastMask);
    }
}

// One horizontal panel of C, column block by column block, the last 1 to 7 columns with a narrower block.
template <int RowVectors>
void multiplyRowPanel(const MatrixProductOperands& operands, std::size_t row, __mmask8 lastMask)
{
    std::size_t column = 0;
    for (; column + BLOCK_COLUMNS <= operands.columnCount; column += BLOCK_COLUMNS)
        multiplyBlock<RowVectors, BLOCK_COLUMNS>(operands, row, column, lastMask);
    multiplyColumnRemainder<RowVectors, BLOCK_COLUMNS - 1>(operands, row, column, operands.columnCount - column,
                                                           lastMask);
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
    std::size_t row = 0;
    for (; row + BLOCK_ROWS <= operands.rowCount; row += BLOCK_ROWS)
        multiplyRowPanel<MAX_ROW_VECTORS>(operands, row, FULL_MASK);

    const std::size_t remainingRows = operands.rowCount - row;
    if (remainingRows == 0)
        return;
    const std::size_t rowVectors = (remainingRows + LANES - 1) / LANES;
    const std::size_t lastLanes = remainingRows - LANES * (rowVectors - 1);
    const auto lastMask = static_cast<__mmask8>((1U << lastLanes) - 1U);
    switch (rowVectors)
    {
    case 3:
        multiplyRowPanel<3>(operands, row, lastMask);
        break;
    case 2:
        multiplyRowPanel<2>(operands, row, lastMask);
        break;
    default:
        multiplyRowPanel<1>(operands, row, lastMask);
        break;
    }
}

} // namespace oos::linalg::kernels::avx512
