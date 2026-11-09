#pragma once

#include <cstddef>
#include <cstdint>

// The entry points of the dense kernel families and their operands. Like kernels/MatrixProduct.hpp,
// this header is included by translation units compiled with SIMD target flags (-mavx2 -mfma, -mavx512f ...): it must
// hold no inline function and include no header that defines one (an inline function compiled in a SIMD unit and in a
// baseline unit is one symbol for the linker, which may keep the SIMD copy and run it on a processor without those
// instructions). Only declarations, plain structs and constants live here. The public face of these kernels is
// linalg/dense/DenseKernels.hpp; nothing else should call the entry points.
//
// Every matrix is column-major with a leading dimension: element (i, j) of a matrix x with leading dimension ldx is
// x[i + j·ldx]. Sizes and indices are 64-bit (std::size_t).
//
// The arithmetic contract: every result element is computed as one
// chain of terms in a fixed order, the same for every blocking, tiling and thread count:
// - reference family: each term is a product rounded, then subtracted (no fused multiply-add; compiled with
//   -ffp-contract=off): the same bits on every machine;
// - SIMD families: each term is one fused multiply-subtract, acc = fma(-x, y, acc): AVX2 and AVX-512 give the same
// bits. Everything else (the products W = L·D, divisions by pivots, the 2 × 2 pivot formula) uses separately rounded
// operations in every family, so that it has the same bits everywhere.

namespace oos::linalg
{

// The block-diagonal D of an LDLᵀ factor: 1 × 1 and symmetric 2 × 2 blocks along the diagonal (Bunch-Kaufman,
// Golub & Van Loan §4.4.4). An empty view (size 0, diagonal null) stands for the identity where an operation allows it.
//
// d_p = diagonal[p·diagonalStride]; e_p = D(p+1, p) = subdiagonal[p]. subdiagonal is null when D is diagonal; otherwise
// e_p != 0 marks a 2 × 2 block on rows p and p+1, and then e_{p+1} = 0 (blocks do not chain) and p+1 < size. With a
// factor stored in place, diagonal points at a(0, 0) and diagonalStride = lda + 1.
struct BlockDiagonalView
{
    const double* diagonal = nullptr;
    std::size_t diagonalStride = 1;
    const double* subdiagonal = nullptr;
    std::size_t size = 0;
};

// The tiles of the SIMD families. A tile size is a multiple of KERNEL_TILE_GRANULE, which is a multiple of every
// micro-tile height and width (AVX2 12 × 4, AVX-512 24 × 8), and at most KERNEL_MAX_TILE. The inner dimension is swept
// in blocks of KERNEL_INNER_BLOCK terms, the accumulators being stored to C and reloaded between blocks so that the
// chain of every element is unchanged.
inline constexpr std::size_t KERNEL_TILE_GRANULE = 24;
inline constexpr std::size_t KERNEL_MAX_TILE = 480;
inline constexpr std::size_t KERNEL_INNER_BLOCK = 256;

// One tile of C ← C − Â·Ŵᵀ, where the terms of element (i, j) are taken over p = 0 .. innerCount-1 (or in decreasing
// p when isInnerReversed):
// - Â(i, p) = a[i + p·lda], or a[p + i·lda] when isATransposed;
// - B̂(j, p) = b[j + p·ldb], or b[p + j·ldb] when isBTransposed;
// - Ŵ = B̂ when d is empty, Ŵ = B̂·D otherwise (D indexed by p, d.size == innerCount, no 2 × 2 block cut at either end):
//   Ŵ(j, p) = B̂(j, p)·d_p for a 1 × 1 block; for a 2 × 2 block on (p, p+1):
//   Ŵ(j, p) = B̂(j, p)·d_p + B̂(j, p+1)·e_p and Ŵ(j, p+1) = B̂(j, p)·e_p + B̂(j, p+1)·d_{p+1}, each product rounded.
// With isLowerOnly, element (i, j) of the tile is updated only when i + diagonalOffset >= j (the tile is a piece of the
// lower triangle of a larger C, diagonalOffset = its first row - its first column); the others are not touched.
struct ProductTileOperands
{
    double* c = nullptr;
    std::size_t ldc = 1;
    std::size_t rowCount = 0;
    std::size_t columnCount = 0;
    const double* a = nullptr;
    std::size_t lda = 1;
    bool isATransposed = false;
    const double* b = nullptr;
    std::size_t ldb = 1;
    bool isBTransposed = false;
    BlockDiagonalView d;
    std::size_t innerCount = 0;
    bool isInnerReversed = false;
    bool isLowerOnly = false;
    std::ptrdiff_t diagonalOffset = 0;
};

// The packing buffers of one SIMD tile, owned by the caller: packedA holds at least rowCount rounded up to
// KERNEL_TILE_GRANULE times KERNEL_INNER_BLOCK values, packedW columnCount rounded up likewise times
// KERNEL_INNER_BLOCK. One workspace per thread; the reference family ignores it.
//
// The packed panels of a product shared by the tasks (the SIMD families' packProductColumns and
// subtractPackedProduct): for one block of terms [firstTerm, firstTerm + termCount) (termCount at most
// KERNEL_INNER_BLOCK, the terms in the order of the operands), Ŵ of a range of columns is packed once into micro-panels
// of the family's micro-tile width, term after term (the micro-panel of the columns from q, a multiple of
// KERNEL_TILE_GRANULE after the range's first, starts at q·termCount values); each task then packs Â of its rows and
// computes those terms of its part of C against the shared Ŵ. The chain of every element continues from what C
// holds, so taking the blocks of terms in increasing order gives the bits of subtractProductTile.
struct PackingWorkspace
{
    double* packedA = nullptr;
    double* packedW = nullptr;
};

// y(i) ← y(i) − Σ_p a(i, p)·v(p) for i < rowCount, p = 0 .. innerCount-1 in increasing order.
struct ColumnProductOperands
{
    double* y = nullptr;
    const double* a = nullptr;
    std::size_t lda = 1;
    const double* v = nullptr;
    std::size_t rowCount = 0;
    std::size_t innerCount = 0;
};

// The panel solve of a block LDLᵀ: P ← P·L⁻ᵀ·D⁻¹ for a panel P (rowCount × columnCount) below a factored diagonal block
// L·D·Lᵀ (columnCount × columnCount). The kernel receives W = L·D of the block, computed by the caller with the
// formula of ProductTileOperands (w(j, q) is read for q < j only, outside the 2 × 2 blocks), and D. For each column j:
// r(i, j) = p(i, j) − Σ_{q < j} p(i, q)·w(j, q) in increasing q (p(i, q) already solved; for a 2 × 2 block on (j, j+1)
// both columns sum over q < j), then p(i, j) = r(i, j) / d_j, or for a 2 × 2 block the formula of LAPACK dsytf2:
// d11 = d_{j+1}/e_j, d22 = d_j/e_j, t = 1/(d11·d22 − 1), s = t/e_j, p(i, j) = s·(d11·r_j − r_{j+1}),
// p(i, j+1) = s·(d22·r_{j+1} − r_j).
struct PanelSolveOperands
{
    double* p = nullptr;
    std::size_t ldp = 1;
    std::size_t rowCount = 0;
    std::size_t columnCount = 0;
    const double* w = nullptr;
    std::size_t ldw = 1;
    BlockDiagonalView d;
};

// A unit lower triangular solve with many right-hand sides on one diagonal block (size × size of l, size × rhsCount
// of b, overwritten by the solution):
// - forward, B ← L⁻¹·B: x(i) = b(i) − Σ_{q < i} l(i, q)·x(q), q increasing;
// - backward, B ← L⁻ᵀ·B: x(i) = b(i) − Σ_{q > i} l(q, i)·x(q), q decreasing.
// The diagonal of l is not read (unit diagonal).
struct TriangleSolveOperands
{
    const double* l = nullptr;
    std::size_t ldl = 1;
    double* b = nullptr;
    std::size_t ldb = 1;
    std::size_t size = 0;
    std::size_t rhsCount = 0;
};

// The kernels of the supernodal solves (linalg/supernodal/FactorSolve) work on a block of right-hand sides stored by
// rows: row e of the block holds the `width` values of equation e for the right-hand sides of the block, contiguous,
// rows `ldx` doubles apart (the transposed of the caller's column-major right-hand sides, so that the values a term
// multiplies are one contiguous vector and the rows of a panel are reached through a list of row indices, without a
// gather into a dense block). `width` is a multiple of the family's vector length (FactorSolve pads the block with
// zero columns); every column is computed alone, so a column has the same bits whatever the width of the block.

// x(target_i) ← x(target_i) − Σ_p l(i, p)·x(source_p) for i < targetCount, the terms in increasing p (one chain per
// element, p = 0 .. termCount-1), where x(row) is the row of `width` values at x + row·ldx, target_i =
// targets[i·targetStep] and source_p = sources[p·sourceStep] (a step of −1 walks a list backward), and
// l(i, p) = l[i·targetStride + p·termStride] (strides in doubles, possibly negative: a panel column-major with leading
// dimension ld gives l(i, p) = panel(first + i, p) by targetStride 1 and termStride ld, and its transposed by
// targetStride ld and termStride 1). No target is a source.
struct RowProductOperands
{
    double* x = nullptr;
    std::size_t ldx = 0;
    std::size_t width = 0;
    const std::int32_t* targets = nullptr;
    std::ptrdiff_t targetStep = 1;
    std::size_t targetCount = 0;
    const std::int32_t* sources = nullptr;
    std::ptrdiff_t sourceStep = 1;
    std::size_t termCount = 0;
    const double* l = nullptr;
    std::ptrdiff_t targetStride = 1;
    std::ptrdiff_t termStride = 1;
};

// The unit lower triangular solves of a supernode's eliminated rows, on rows of right-hand sides (as above): for the
// rows row_i = rows[i] (i < size) and the unit lower triangle l(i, q) = l[i + q·ldl] (its diagonal and upper part not
// read):
// - forward, x(row_i) ← x(row_i) − Σ_{q < i} l(i, q)·x(row_q), q increasing (L⁻¹);
// - transposed, x(row_i) ← x(row_i) − Σ_{q > i} l(q, i)·x(row_q), q decreasing (L⁻ᵀ);
// the chains of TriangleSolveOperands, row by row.
struct RowTriangleOperands
{
    double* x = nullptr;
    std::size_t ldx = 0;
    std::size_t width = 0;
    const std::int32_t* rows = nullptr;
    std::size_t size = 0;
    const double* l = nullptr;
    std::size_t ldl = 1;
    bool isTransposed = false;
};

// The in-place LDLᵀ of the reference family (whole block, unblocked). Columns 0 .. pivotColumnCount-1 are eliminated:
// on return they hold L below the diagonal and D on it, and the trailing block holds the Schur complement (lower
// triangle). Stops at the first pivot that is zero or not finite.
struct LdltOperands
{
    double* a = nullptr;
    std::size_t lda = 1;
    std::size_t size = 0;
    std::size_t pivotColumnCount = 0;
};

// The in-place LDLᵀ of the reference family on a front stored in two parts: the c = pivotColumnCount columns to
// eliminate over all the front's rows in `panel` (c + rowCount rows, leading dimension ldp), the trailing block of its
// last rowCount rows in `contribution` (leading dimension ldc). Element (i, j) of the front is panel[i + j·ldp] for
// j < c, contribution[(i − c) + (j − c)·ldc] otherwise; each receives the operations LdltOperands' factorisation
// gives it, in the same order (the same bits).
struct SplitLdltOperands
{
    double* panel = nullptr;
    std::size_t ldp = 1;
    double* contribution = nullptr;
    std::size_t ldc = 1;
    std::size_t pivotColumnCount = 0;
    std::size_t rowCount = 0;
};

// The in-place LDLᵀ with Bunch-Kaufman pivoting of the reference family (unblocked): P·A·Pᵀ = L·D·Lᵀ.
// permutation[k] = the original index at position k; subdiagonal[k] = e_k (0 outside the 2 × 2 blocks); the strictly
// lower part of a holds L (L(k+1, k) = 0 for a 2 × 2 block), its diagonal holds d.
//
// Only the first candidateCount columns may be pivots (the fully summed columns of a front of the multifrontal
// factorisation); the other rows are never interchanged. A candidate whose elimination would need a
// pivot outside the candidates is delayed: moved behind the remaining candidates, left uneliminated, and retried in a
// further pass after the candidates eliminated meanwhile; the entry point returns the number of columns eliminated,
// the delayed ones following them (positions [eliminated, candidateCount)), the trailing block holding the Schur
// complement. With candidateCount = size nothing is ever delayed: the textbook Bunch-Kaufman factorisation.
struct PivotedLdltOperands
{
    double* a = nullptr;
    std::size_t lda = 1;
    std::size_t size = 0;
    std::size_t* permutation = nullptr; // candidateCount values
    double* subdiagonal = nullptr;      // candidateCount values
    std::size_t candidateCount = 0;
};

// The Bunch-Kaufman constant α = (1 + √17)/8, which minimises the bound on element growth (Bunch & Kaufman 1977;
// Golub & Van Loan §4.4.4; LAPACK dsytf2).
inline constexpr double BUNCH_KAUFMAN_ALPHA = 0.6403882032022076;

// The threshold u of the pivot test of a candidate whose largest off-diagonal term lies in a row that cannot be a
// pivot: a 1 × 1 pivot is accepted when |a_kk| >= u·max_i |a_ik|, so that the entries of L stay below 1/u (threshold
// partial pivoting, Duff & Reid, "The multifrontal solution of indefinite sparse symmetric linear equations", ACM TOMS
// 9, 1983; 0.01 is the default of MA27 and MA57). Otherwise the candidate is delayed. Bunch-Kaufman's own test (α) is
// kept whenever the pivot partner it needs is a candidate.
inline constexpr double DELAYED_PIVOT_THRESHOLD = 0.01;

namespace kernels::reference
{
void subtractProductTile(const ProductTileOperands& operands, const PackingWorkspace& workspace) noexcept;
void subtractColumnProducts(const ColumnProductOperands& operands) noexcept;
void solvePanel(const PanelSolveOperands& operands) noexcept;
void solveUnitLowerBlock(const TriangleSolveOperands& operands) noexcept;
void solveUnitLowerTransposedBlock(const TriangleSolveOperands& operands) noexcept;
void subtractRowProducts(const RowProductOperands& operands) noexcept;
void solveRowTriangle(const RowTriangleOperands& operands) noexcept;
// Returns the number of columns eliminated (pivotColumnCount, or the index of the zero or non-finite pivot met).
[[nodiscard]] std::size_t factorWithoutPivoting(const LdltOperands& operands) noexcept;
// Returns the number of columns eliminated, as factorWithoutPivoting; the contribution is updated only when all are,
// and not at all when it is null (the panel alone: DenseKernels::factorPanelWithoutPivoting).
[[nodiscard]] std::size_t factorFrontWithoutPivoting(const SplitLdltOperands& operands) noexcept;
// Returns the number of columns eliminated (candidateCount when none is delayed).
[[nodiscard]] std::size_t factorWithPivoting(const PivotedLdltOperands& operands) noexcept;
} // namespace kernels::reference

namespace kernels::avx2
{
void subtractProductTile(const ProductTileOperands& operands, const PackingWorkspace& workspace) noexcept;
// Ŵ of the operands' columnCount columns for the terms [firstTerm, firstTerm + termCount) into packedW (columnCount
// rounded up to KERNEL_TILE_GRANULE times termCount values).
void packProductColumns(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                        double* packedW) noexcept;
// Those terms of C ← C − Â·Ŵᵀ for the operands' rows and columns, against packedW (the panels of the operands' first
// column); Â packed into packedA (rowCount rounded up to KERNEL_TILE_GRANULE times termCount values).
void subtractPackedProduct(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                           const double* packedW, double* packedA) noexcept;
// Â of the operands' rowCount rows for the terms [firstTerm, firstTerm + termCount) into packedA (rowCount rounded up
// to KERNEL_TILE_GRANULE times termCount values: the micro-panel of the rows from q, a multiple of the family's
// micro-tile height after the first, starts at q·termCount values): a front's tasks pack each row block once.
void packProductRows(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                     double* packedA) noexcept;
// termCount terms of C ← C − Â·Ŵᵀ for the operands' rows and columns from panels packed beforehand (packProductRows,
// packProductColumns), the micro-tiles alone; each element's chain continues from what C holds (only c, ldc, the
// counts, isLowerOnly and diagonalOffset of the operands are read).
void subtractPrepackedProduct(const ProductTileOperands& operands, std::size_t termCount, const double* packedA,
                              const double* packedW) noexcept;
void subtractColumnProducts(const ColumnProductOperands& operands) noexcept;
void solvePanel(const PanelSolveOperands& operands) noexcept;
void solveUnitLowerBlock(const TriangleSolveOperands& operands) noexcept;
void solveUnitLowerTransposedBlock(const TriangleSolveOperands& operands) noexcept;
void subtractRowProducts(const RowProductOperands& operands) noexcept;
void solveRowTriangle(const RowTriangleOperands& operands) noexcept;
} // namespace kernels::avx2

namespace kernels::avx512
{
void subtractProductTile(const ProductTileOperands& operands, const PackingWorkspace& workspace) noexcept;
void packProductColumns(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                        double* packedW) noexcept;
void subtractPackedProduct(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                           const double* packedW, double* packedA) noexcept;
void packProductRows(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                     double* packedA) noexcept;
void subtractPrepackedProduct(const ProductTileOperands& operands, std::size_t termCount, const double* packedA,
                              const double* packedW) noexcept;
void subtractColumnProducts(const ColumnProductOperands& operands) noexcept;
void solvePanel(const PanelSolveOperands& operands) noexcept;
void solveUnitLowerBlock(const TriangleSolveOperands& operands) noexcept;
void solveUnitLowerTransposedBlock(const TriangleSolveOperands& operands) noexcept;
void subtractRowProducts(const RowProductOperands& operands) noexcept;
void solveRowTriangle(const RowTriangleOperands& operands) noexcept;
} // namespace kernels::avx512

} // namespace oos::linalg
