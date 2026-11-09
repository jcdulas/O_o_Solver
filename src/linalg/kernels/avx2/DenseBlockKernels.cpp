// The dense kernels of the AVX2/FMA family, compiled with -mavx2 -mfma.
// The algorithms are those of kernels/SimdDenseKernels.ipp, included here with the vector operations of this family;
// every helper has internal linkage, and this unit defines nothing with external linkage but the entry points of
// kernels/DenseKernelEntryPoints.hpp (see kernels/MatrixProduct.hpp for why nothing may be shared with baseline code).

#include "oos/linalg/kernels/DenseKernelEntryPoints.hpp"

#include <immintrin.h>

namespace oos::linalg::kernels::avx2
{
namespace
{

// The vector operations of the family (12 × 4 micro-tile: 3 YMM registers of rows, 4 columns; the row products
// of the solves 6 target rows × 2 YMM registers of columns).
struct Simd
{
    using Vector = __m256d;
    static constexpr std::size_t LANES = 4;
    static constexpr std::size_t ROW_VECTORS = 3;
    static constexpr std::size_t COLUMNS = 4;
    static constexpr std::size_t ROW_TARGETS = 6;
    static constexpr std::size_t ROW_STRIP_VECTORS = 2;

    static Vector load(const double* source) { return _mm256_loadu_pd(source); }
    static void store(double* target, Vector value) { _mm256_storeu_pd(target, value); }
    static Vector broadcast(double value) { return _mm256_set1_pd(value); }
    // c − a·b with one rounding: vfnmadd computes −(a·b) + c exactly, then rounds.
    static Vector subtractProduct(Vector a, Vector b, Vector c) { return _mm256_fnmadd_pd(a, b, c); }
    static Vector multiply(Vector a, Vector b) { return _mm256_mul_pd(a, b); }
    static Vector subtract(Vector a, Vector b) { return _mm256_sub_pd(a, b); }
    static Vector divide(Vector a, Vector b) { return _mm256_div_pd(a, b); }
    // The scalar form of subtractProduct, the same bits as one lane of it: fma(−a, b, c) = −(a·b) + c rounded once.
    static double subtractProduct(double a, double b, double c) { return __builtin_fma(-a, b, c); }
    // A hint: the cache line holding `address` brought to L1 ahead of its use (no fault, no effect on any value).
    static void prefetch(const double* address) { _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0); }
};

#include "../SimdDenseKernels.ipp"

} // namespace

void subtractProductTile(const ProductTileOperands& operands, const PackingWorkspace& workspace) noexcept
{
    subtractTile(operands, workspace);
}

void packProductColumns(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                        double* packedW) noexcept
{
    packW(operands, firstTerm, termCount, packedW);
}

void subtractPackedProduct(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                           const double* packedW, double* packedA) noexcept
{
    subtractPackedProductRows(operands, firstTerm, termCount, packedW, packedA);
}

void packProductRows(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                     double* packedA) noexcept
{
    packA(operands, firstTerm, termCount, packedA);
}

void subtractPrepackedProduct(const ProductTileOperands& operands, std::size_t termCount, const double* packedA,
                              const double* packedW) noexcept
{
    subtractPackedMicroTiles(operands, termCount, packedA, packedW);
}

void subtractColumnProducts(const ColumnProductOperands& operands) noexcept
{
    subtractColumnProductsAll(operands);
}

void solvePanel(const PanelSolveOperands& operands) noexcept
{
    solvePanelAll(operands);
}

void solveUnitLowerBlock(const TriangleSolveOperands& operands) noexcept
{
    solveUnitLowerBlockAll(operands);
}

void solveUnitLowerTransposedBlock(const TriangleSolveOperands& operands) noexcept
{
    solveUnitLowerTransposedBlockAll(operands);
}

void subtractRowProducts(const RowProductOperands& operands) noexcept
{
    subtractRowProductsAll(operands);
}

void solveRowTriangle(const RowTriangleOperands& operands) noexcept
{
    solveRowTriangleAll(operands);
}

} // namespace oos::linalg::kernels::avx2
