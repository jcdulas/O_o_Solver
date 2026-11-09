#pragma once

#include "EigenvectorStore.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::linalg
{

// The rows of one piece of the inner products of the Lanczos vectors, by default: the piece of a
// block of eight columns (64 kB) stays in the L2 cache while the columns of the basis stream past it.
inline constexpr std::size_t INNER_PRODUCT_PIECE_ROWS = 1024;

// The Gram-Schmidt steps of the block Lanczos solver in the M-inner product ⟨x, y⟩ = xᵀ·M·y (M positive
// semi-definite: a true inner product on the range of (K − σM)⁻¹·M, where every Lanczos vector lies). For a buckling
// pencil the inner product is the K-inner product (ShiftedPencil::multiplyInnerProduct; K positive definite) and "M"
// below reads K.
//
// projectOut is one pass of block classical Gram-Schmidt against two sets of M-orthonormal columns (the locked
// eigenvectors and the Lanczos basis), with one product M·W shared by both: W ← W − L·(Lᵀ·M·W) − Q·(Qᵀ·M·W). The solver
// applies it twice ("twice is enough": Giraud, Langou & Rozložník, Comput. Math. Appl. 50 (2005); a column that loses
// nearly all its norm is rank-deficient and is replaced, never kept). coefficients and subtractColumns are the same
// steps for one column against the columns before it in its block.
//
// Every inner product over the rows of the pencil (the coefficients of a projection, the norms, the normalisations) is
// summed by pieces of `pieceRows` rows: a piece's sum is one chain per coefficient in increasing
// rows (several coefficients side by side), or, for a single dot product, four interleaved chains (its rows by their
// remainder modulo 4, added pairwise at the end of the piece); then the pieces' sums are added in increasing order.
// The pieces depend on the dimension and `pieceRows` only, never on the threads, which share them: the bits do not
// depend on the threads. The updates (W − L·C_L − Q·C_Q, w − Σ c_i·q_i) give each value of W one chain over the
// columns in increasing order (the kernels' chains), whatever the pieces.
//
// Memory: its partial sums and tiles (estimateBytes, about 4.5 MB at most) in the budget, at construction.
//
// Thread-safety: not thread-safe (it writes its scratch). Ownership: references the pencil, the kernels, the budget and
// the scratch block (dimension × at least the widest W), which must outlive it.
class MassOrthogonalizer
{
public:
    MassOrthogonalizer(const ShiftedPencil& pencil, const DenseKernels& kernels, base::MemoryBudget& budget,
                       DenseMatrixView scratch, std::size_t pieceRows = INNER_PRODUCT_PIECE_ROWS);

    // One pass: W ← W − L·(Lᵀ·M·W) − Q·(Qᵀ·M·W). When `coefficients` is not null it receives Qᵀ·M·W (Q.columns ×
    // W.columns, column-major). When normsBefore is not empty it receives the M-norms of the columns of W before the
    // pass. Either set may be empty (0 columns). The locked set's spooled columns are read piece by piece
    // where the resident ones are read in place: the same values in the same chains, the same bits. Throws
    // MemoryBudgetError for its partial sums, FileError for a spooled piece.
    void projectOut(DenseMatrixView w, const LockedVectors& locked, ConstDenseMatrixView basis,
                    std::vector<double>* coefficients, std::span<double> normsBefore);

    // The bytes it reserves at most for a pencil of `dimension` (the estimate of the solver's memory).
    [[nodiscard]] static std::uint64_t estimateBytes(std::size_t dimension, std::size_t pieceRows) noexcept;

    // y ← M·x for one column (x and y of dimension rows).
    void multiplyColumn(const double* x, double* y) const;

    // out[i] = Σ_r y(r, i)·w_r for the columns of y (y.columns values), in one sweep of the rows.
    void coefficients(ConstDenseMatrixView y, const double* w, std::span<double> out);
    // w ← w − Σ_i c_i·q_i over the columns of q (each value one chain in increasing i); then, when y has columns,
    // out[i] = Σ_r y(r, i)·w_r of the updated w, in the same sweep (each row of w updated before it enters the sums).
    void subtractColumns(double* w, ConstDenseMatrixView q, std::span<const double> c, ConstDenseMatrixView y,
                         std::span<double> out);
    // Σ x_r·y_r over `count` rows.
    [[nodiscard]] double dot(const double* x, const double* y, std::size_t count);
    // x ← s·x, and y ← s·y when y is not null (each value alone).
    void scale(double* x, double* y, double s, std::size_t count) const;

private:
    // Σ over the pieces of `count` rows of what `piece(first, end, sums, task)` writes in `sums` (`width` values),
    // added in increasing order of the pieces into `totals`; the pieces run on the pool by batches whose sums fit the
    // partial sums (PARTIAL_SUM_VALUES, or one piece's width when it is wider), `task` the index of the task that runs
    // the piece (its tile).
    template <typename Piece>
    void sumByPieces(std::size_t count, std::size_t width, double* totals, Piece&& piece);
    // body(first, end) for the rows [0, count) by whole pieces, on the pool.
    template <typename Body>
    void forPieces(std::size_t count, Body&& body) const;

    const ShiftedPencil& m_pencil;
    const DenseKernels& m_kernels;
    base::MemoryBudget& m_budget;
    // The kernels of a piece of the update: the same family on the thread that runs the piece (its chains).
    DenseKernels m_pieceKernels;
    DenseMatrixView m_scratch;
    std::size_t m_pieceRows;
    // The tasks of a sweep (TASKS_PER_THREAD a thread, at most MAXIMUM_TASKS), each with its tile.
    std::size_t m_taskCount;
    base::BudgetedBuffer<double> m_tiles;
    base::BudgetedBuffer<double> m_partialSums;
    // The pieces of the locked set's spooled columns, one a task: grown with the spooled columns.
    base::BudgetedBuffer<double> m_spooledPieces;
};

} // namespace oos::linalg
