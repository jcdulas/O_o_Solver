// The dense kernels of the WebAssembly family, compiled with -msimd128 (the 128-bit vector instructions of
// WebAssembly, two doubles per register).
// The algorithms are those of kernels/SimdDenseKernels.ipp, included here with the vector operations of this family;
// every helper has internal linkage, and this unit defines nothing with external linkage but the entry points of
// kernels/DenseKernelEntryPoints.hpp.
//
// WebAssembly has no fused multiply-add among these instructions: a term is a product rounded, then subtracted, as in
// the reference family. Every element still receives its terms one by one in the order the operands fix, so these
// kernels give the bits of the reference family; what they add is the packing and the register tiles of the
// template. The factorisation with pivoting of this family is the reference's (KernelFamily::pivotsByBlocks): the
// blocked one of the template orders its operations otherwise, and its results differ in their last digits.
// Measured on models of shells in both formulations: the result tables of the static and of the modal analyses are
// those of the reference family, bit for bit.

#include "oos/linalg/kernels/DenseKernelEntryPoints.hpp"

#include <wasm_simd128.h>

namespace oos::linalg::kernels::wasm128
{
namespace
{

// The vector operations of the family (8 × 4 micro-tile: 4 registers of rows, 4 columns, 16 accumulators; the row
// products of the solves 4 target rows × 2 registers of columns).
// PERFORMANCE: the height of the micro-tile. Measured on a static analysis of 2 304 curved shells of the mixed hybrid
// formulation with their stresses (one thread, i5-12400F, Node.js 24): the whole run 20.6 to 21.4 s with 2 registers of
// rows, 21.0 to 21.6 s with 3, 20.0 to 20.1 s with 4; 22.5 s with the reference family.
struct Simd
{
    using Vector = v128_t;
    static constexpr std::size_t LANES = 2;
    static constexpr std::size_t ROW_VECTORS = 4;
    static constexpr std::size_t COLUMNS = 4;
    static constexpr std::size_t ROW_TARGETS = 4;
    static constexpr std::size_t ROW_STRIP_VECTORS = 2;

    static Vector load(const double* source) { return wasm_v128_load(source); }
    static void store(double* target, Vector value) { wasm_v128_store(target, value); }
    static Vector broadcast(double value) { return wasm_f64x2_splat(value); }
    // c − a·b: the product rounded, then the subtraction rounded (no fused operation in this family).
    static Vector subtractProduct(Vector a, Vector b, Vector c) { return wasm_f64x2_sub(c, wasm_f64x2_mul(a, b)); }
    static Vector multiply(Vector a, Vector b) { return wasm_f64x2_mul(a, b); }
    static Vector subtract(Vector a, Vector b) { return wasm_f64x2_sub(a, b); }
    static Vector divide(Vector a, Vector b) { return wasm_f64x2_div(a, b); }
    // The scalar form of subtractProduct, the same bits as one lane of it (-ffp-contract=off: two roundings).
    static double subtractProduct(double a, double b, double c) { return c - a * b; }
    // WebAssembly has no prefetch instruction: nothing to do.
    static void prefetch(const double* address) { static_cast<void>(address); }
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

} // namespace oos::linalg::kernels::wasm128
