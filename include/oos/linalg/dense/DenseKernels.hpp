#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/dense/Inertia.hpp"
#include "oos/linalg/kernels/DenseKernelEntryPoints.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::base
{
class CancellationToken;
class MemoryBudget;
class ThreadPool;
} // namespace oos::base

namespace oos::linalg
{

class KernelSelection;
class KernelWorkspace;

// How a DenseKernels object runs its work.
struct DenseKernelOptions
{
    // Below this many multiply-adds an operation runs on the calling thread: handing out a few microseconds of work
    // to a pool costs more than it gains (a batch costs a few microseconds to start and join).
    static constexpr std::uint64_t DEFAULT_PARALLEL_MINIMUM_MULTIPLY_ADDS = std::uint64_t{1} << 22;
    // The tile of C one task computes (rows and columns). 192 rows of packed A for 256 terms take 384 kB, which stays
    // in the 1.25 MB L2 of a Golden Cove core with room for the packed W and the tile of C; a multiple of
    // KERNEL_TILE_GRANULE.
    static constexpr std::size_t DEFAULT_TILE_SIZE = 192;

    // The pool that runs the tiles; null runs everything on the calling thread. Not owned; must outlive the kernels.
    base::ThreadPool* pool = nullptr;
    // The threshold above which an operation is shared between the threads of the pool. Tests lower it to 0 so that
    // the threaded code runs on small blocks; it never changes a result (the tiles do not depend on it).
    std::uint64_t parallelMinimumMultiplyAdds = DEFAULT_PARALLEL_MINIMUM_MULTIPLY_ADDS;
    // The tile size: a multiple of KERNEL_TILE_GRANULE between KERNEL_TILE_GRANULE and KERNEL_MAX_TILE. It decides the
    // pieces of the parallel work and the cache behaviour, never a result.
    std::size_t tileSize = DEFAULT_TILE_SIZE;
    // Checked between the tiles of a threaded operation and between the blocks of a blocked factorisation: a
    // cancelled operation throws OperationCancelledError, leaving its output partly updated. Not owned; may be null.
    const base::CancellationToken* cancellation = nullptr;
    // The packing buffers kept from one call to the next (KernelWorkspace), for a caller that makes many calls (the
    // multifrontal factorisation); null: each call allocates its own. Never changes a result. Not owned; may be null.
    KernelWorkspace* workspace = nullptr;
    // The LDLᵀ without pivoting of a front of at least this many multiply-adds runs as tasks (TaskedFront)
    // when the pool has more than one thread and the family is blocked. Tests lower it to 0 so that the tasks run on
    // small fronts; it never changes a result (the tasks keep every chain).
    // (About 0.4 ms on one core: measured on six threads, the tasks are no slower than the recursion's threaded calls
    // from a front of 600 rows and 150 or 252 columns, 2.4e7 multiply-adds.)
    static constexpr double DEFAULT_TASKED_FRONT_MINIMUM_MULTIPLY_ADDS = 2.0e7;
    double taskedFrontMinimumMultiplyAdds = DEFAULT_TASKED_FRONT_MINIMUM_MULTIPLY_ADDS;
    // The budget the tasks' packed panels are reserved in when the caller gives no buffer for them (a front whose
    // packed panels do not fit runs on the recursion, the same bits); null: not reserved. Not owned; may be null.
    base::MemoryBudget* budget = nullptr;
};

// What an LDLᵀ without pivoting did.
struct LdltOutcome
{
    // The columns eliminated: the requested count, or the index of the first pivot that was zero or not finite (the
    // columns from there on are left as they were after the updates of the eliminated ones).
    std::size_t eliminatedCount = 0;
    bool isComplete = false;
    // The signs of the pivots eliminated (no zero: elimination stops at a zero pivot).
    Inertia inertia;
};

// The pivoting of an LDLᵀ with Bunch-Kaufman pivoting: P·A·Pᵀ = L·D·Lᵀ.
struct PivotedLdlt
{
    // permutation[k] = the index in the original matrix of the row and column now at position k (one value per
    // candidate column; the rows beyond the candidates are never interchanged).
    std::vector<std::size_t> permutation;
    // e_k = D(k+1, k): non-zero at the first column of each 2 × 2 pivot block, 0 elsewhere (one value per candidate).
    std::vector<double> subdiagonal;
    // The inertia of D (of the eliminated columns), hence of A for a complete factorisation (Sylvester).
    Inertia inertia;
    std::size_t twoByTwoBlockCount = 0;
    // The columns eliminated; the delayed candidates are the positions [eliminatedCount, candidate count).
    std::size_t eliminatedCount = 0;
};

// The dense kernels of the factorisation for one kernel family: the block LDLᵀ without pivoting (with
// the factor diagonal ratios of the MAXRATIO check) and with Bunch-Kaufman pivoting, the triangular solves of a panel
// and of many right-hand sides, and the update C ← C − A·D·Aᵀ.
//
// The reference family runs the textbook algorithms of kernels/reference, the oracle; the SIMD families run blocked
// algorithms on register-blocked micro-kernels with packed panels. Every result element is one chain of terms in a
// fixed order (kernels/DenseKernelEntryPoints.hpp), so a result has the same bits whatever the tile size, the blocking
// and the number of threads; AVX2 and AVX-512 give the same bits; the reference and a SIMD family agree to rounding
// The threads share the tiles of a result, never the terms of one sum.
//
// All matrices are column-major views (DenseMatrixView); a symmetric matrix is read and written in its lower triangle
// only. Dimension mismatches throw std::invalid_argument naming the operation and both sizes.
//
// Thread-safety: an immutable value; several threads may call it at once on different data (each call uses the pool
// of the options, which supports nested batches). Ownership: does not own the pool.
class DenseKernels
{
public:
    // Throws std::invalid_argument when the tile size is not a multiple of KERNEL_TILE_GRANULE in
    // [KERNEL_TILE_GRANULE, KERNEL_MAX_TILE]. The caller has checked that the processor supports `kind`
    // (KernelSelection); a SIMD family on a processor without its instructions faults.
    explicit DenseKernels(KernelKind kind, DenseKernelOptions options = {});
    DenseKernels(const KernelSelection& selection, DenseKernelOptions options);

    [[nodiscard]] KernelKind kind() const noexcept { return m_kind; }
    [[nodiscard]] const DenseKernelOptions& options() const noexcept { return m_options; }

    // C ← C − A·D·Aᵀ on the lower triangle of the square C (n × n); A is n × k, D is k × k (an empty view: identity).
    // The upper triangle of C is not touched. Element (i, j) is c(i, j) − Σ_p a(i, p)·w(j, p) with W = A·D, p
    // increasing.
    void updateSymmetric(DenseMatrixView c, ConstDenseMatrixView a, const BlockDiagonalView& d) const;

    // C ← C − A·D·Bᵀ on the whole of C (m × n); A is m × k, B is n × k, D is k × k (an empty view: identity).
    void updateRectangular(DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b,
                           const BlockDiagonalView& d) const;

    // The in-place LDLᵀ of the symmetric `a` (n × n, lower triangle) without pivoting, eliminating its first
    // pivotColumnCount columns (all of them for a full factorisation; fewer for a partial one, the trailing block then
    // holds the Schur complement). On return the eliminated columns hold L below the diagonal (unit
    // diagonal implied) and D on it.
    // pivotRatios, when not empty (at least pivotColumnCount values), receives for each eliminated column the ratio of
    // the matrix diagonal to the factor diagonal, a_jj / d_j: the quantity compared with PARAM MAXRATIO; a large or
    // negative ratio flags a near-singularity.
    [[nodiscard]] LdltOutcome factorWithoutPivoting(DenseMatrixView a, std::size_t pivotColumnCount,
                                                    std::span<double> pivotRatios = {}) const;

    // The same factorisation on a front stored in two parts (the multifrontal factorisation): `panel`
    // holds the c columns to eliminate over all the n = c + r rows of the front (n × c, their diagonal block on top),
    // `contribution` the trailing block of its last r rows (r × r, lower triangle). Eliminates the c columns in the
    // panel and leaves in `contribution` the Schur complement F_rr − L_r·D·L_rᵀ: the bits factorWithoutPivoting(a, c)
    // gives on the front a = [panel | [0; contribution]] (every element receives the same chain; the trailing block
    // receives the terms of the c columns in one update), so a front need not be one matrix. The upper
    // triangle of the panel's diagonal block is not touched. At a zero or non-finite pivot the elimination stops there
    // (eliminatedCount) and `contribution` is left as it was. Throws std::invalid_argument when the shapes do not
    // match.
    [[nodiscard]] LdltOutcome factorFrontWithoutPivoting(DenseMatrixView panel, DenseMatrixView contribution) const;
    // The same with the trailing block stored by blocks of contributionBlockColumns columns (UpdateLayout, the update
    // matrices of the full path streamed: `contribution` holds its data, r × r; 0 or at least r: square of
    // its leading dimension), and the tasks' packed panels in `packing` when it holds TaskedFront::packingValues()
    // values (empty: the kernels allocate them). Every element the chain of the square form.
    [[nodiscard]] LdltOutcome factorFrontWithoutPivoting(DenseMatrixView panel, DenseMatrixView contribution,
                                                         std::int64_t contributionBlockColumns,
                                                         std::span<double> packing = {}) const;
    // The panel of factorFrontWithoutPivoting alone, its trailing update left to the caller. Throws
    // std::invalid_argument for a panel of fewer rows than columns.
    [[nodiscard]] LdltOutcome factorPanelWithoutPivoting(DenseMatrixView panel) const;

    // The in-place LDLᵀ of the symmetric `a` (n × n, lower triangle) with Bunch-Kaufman pivoting (1 × 1 and 2 × 2
    // pivots, Bunch & Kaufman 1977, the partial pivoting strategy of LAPACK dsytrf): P·A·Pᵀ = L·D·Lᵀ. On return the
    // strictly lower part holds L (L(k+1, k) = 0 at a 2 × 2 block) and the diagonal holds the d of D; the result
    // holds P, the subdiagonal of D and the inertia. A zero column gives a zero 1 × 1 pivot (counted in the inertia).
    [[nodiscard]] PivotedLdlt factorWithPivoting(DenseMatrixView a) const;

    // The partial form for a front of the multifrontal factorisation (delayed pivots): only the first
    // candidateCount columns (the fully summed ones) may be pivots, the other rows are never interchanged. A
    // candidate that needs a pivot outside the candidates and fails the threshold test (DELAYED_PIVOT_THRESHOLD) is
    // delayed: on return the eliminated columns come first (L and D as above), the delayed candidates follow them, and
    // the trailing block from eliminatedCount on holds the Schur complement, delayed candidates included. With
    // candidateCount = n this is factorWithPivoting(a). Throws std::invalid_argument when candidateCount > n.
    [[nodiscard]] PivotedLdlt factorWithPivoting(DenseMatrixView a, std::size_t candidateCount) const;

    // The panel solve of a block LDLᵀ: P ← P·L⁻ᵀ·D⁻¹, P being m × b and `factored` the b × b diagonal block factored in
    // place (L strictly below its diagonal; D given by d, 1 × 1 or 2 × 2 blocks). The rows of P are independent.
    // b is at most KERNEL_MAX_TILE (a block of a blocked factorisation; std::invalid_argument beyond).
    void solvePanel(DenseMatrixView panel, ConstDenseMatrixView factored, const BlockDiagonalView& d) const;

    // B ← L⁻¹·B for the unit lower triangular L (n × n, its diagonal not read) and B n × r (forward substitution).
    void solveUnitLower(ConstDenseMatrixView l, DenseMatrixView b) const;
    // B ← L⁻ᵀ·B (backward substitution).
    void solveUnitLowerTransposed(ConstDenseMatrixView l, DenseMatrixView b) const;
    // B ← D⁻¹·B for the block-diagonal D (n × n) and B n × r.
    void solveBlockDiagonal(const BlockDiagonalView& d, DenseMatrixView b) const;

    // C ← C − A·B for A m × k, B k × n, C m × n: element (i, j) is c(i, j) − Σ_p a(i, p)·b(p, j), p increasing (the
    // update of the rows below a supernode in a forward substitution).
    void subtractProduct(DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b) const;
    // C ← C − Aᵀ·B for A k × m, B k × n, C m × n: element (i, j) is c(i, j) − Σ_p a(p, i)·b(p, j), p increasing (the
    // backward substitution).
    void subtractTransposedProduct(DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b) const;

    // The kernels of the supernodal solves on rows of right-hand sides (kernels/DenseKernelEntryPoints.hpp,
    // RowProductOperands and RowTriangleOperands), on the calling thread: their caller (FactorSolve) shares the rows
    // and the columns of its blocks between threads. The width must be a multiple of rowLanes() and at most ldx
    // (std::invalid_argument otherwise); the rows are not checked (the caller's index lists).
    void subtractRowProducts(const RowProductOperands& operands) const;
    void solveRowTriangle(const RowTriangleOperands& operands) const;
    // The vector length of the family, of which the width of the rows is a multiple: 1 for the reference family, 4
    // for AVX2, 8 for AVX-512.
    [[nodiscard]] std::size_t rowLanes() const noexcept;

private:
    void subtractProductTiles(DenseMatrixView c, ConstDenseMatrixView a, bool isATransposed,
                              ConstDenseMatrixView b) const;

    KernelKind m_kind;
    DenseKernelOptions m_options;
};

// D of a factor stored in place: d on the diagonal of `factored`, e in `subdiagonal` (null for a diagonal D).
[[nodiscard]] BlockDiagonalView blockDiagonalOf(ConstDenseMatrixView factored,
                                                const double* subdiagonal = nullptr) noexcept;

// The multiply-adds of each operation (the flop count is twice as many), for the benchmarks and the cost estimates.
[[nodiscard]] std::uint64_t symmetricUpdateMultiplyAdds(std::size_t order, std::size_t innerCount) noexcept;
[[nodiscard]] std::uint64_t ldltMultiplyAdds(std::size_t order) noexcept;

} // namespace oos::linalg
