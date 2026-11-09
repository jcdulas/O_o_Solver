#pragma once

#include "KernelFamily.hpp"

#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace oos::linalg
{

// The blocked LDLᵀ factorisations of the SIMD families (internal to DenseKernels).
//
// Both work by blocks of columns: the columns of a block are computed left-looking against the earlier columns of the
// same block (column products), the rows below the block are solved by the panel kernel (without pivoting), and the
// columns after the block receive its terms by the tiled updates (C ← C − L·D·Lᵀ, the bulk of the work, on the
// micro-kernels). Without pivoting the blocks are taken recursively (factorColumns): the columns of a panel are split
// in two, the left part is eliminated, the right part's columns receive the terms of all the left part's columns in
// one update, then the right part is eliminated; the trailing block (the Schur complement, a front's update matrix)
// receives the terms of every eliminated column in one update at the end. Every element thus receives the terms
// L(i, p)·W(j, p) of all earlier columns p in increasing order, the chain the reference family follows column by
// column: the result does not depend on the block size, the recursion, the tile size or the thread count.
//
// Thread-safety: one object per call. Ownership: references the kernels and the family, which outlive it.
class BlockedLdlt
{
public:
    // PERFORMANCE: the block width trades the share of the work done by the trailing update on the tiled
    // micro-kernels (the larger the block, the more) against the cost of the in-block work, which runs at a fraction
    // of that rate (the smaller, the less). Without pivoting the diagonal block of LDLT_BLOCK_COLUMNS is itself
    // factorised by blocks of INNER_BLOCK_COLUMNS (its own panel solve and update); with pivoting the columns of a
    // block are computed by column products over the whole height, memory-bound, hence the narrower block.
    // Measured (i5-12400F, one thread, machine shared): ldlt at n = 1 000 and 2 000 fastest with
    // 96 (55 % of the peak against 40-48 % with 64 and 128); bunch-kaufman at n = 2 000 18-19 GFLOP/s with 24 to 64
    // against 11.6 with 128.
    static constexpr std::size_t LDLT_BLOCK_COLUMNS = 96;
    static constexpr std::size_t INNER_BLOCK_COLUMNS = 16;
    static constexpr std::size_t PIVOTED_BLOCK_COLUMNS = 64;

    BlockedLdlt(const DenseKernels& kernels, const KernelFamily& family) : m_kernels(kernels), m_family(family) {}

    // Eliminates the first pivotColumnCount columns of `a` without pivoting; returns the number eliminated (the index
    // of the first zero or non-finite pivot when there is one).
    [[nodiscard]] std::size_t factorWithoutPivoting(DenseMatrixView a, std::size_t pivotColumnCount) const;

    // The same on a front in two parts (DenseKernels::factorFrontWithoutPivoting): the blocks of the panel's columns
    // update the panel's later columns only; the trailing block then receives the terms of all the columns in one
    // update. Its elements get the chain they get in factorWithoutPivoting on the whole front (the blocking and the
    // tiles decide only when a chain is stored and reloaded). Returns the columns eliminated; the contribution
    // is updated only when all are.
    [[nodiscard]] std::size_t factorFront(DenseMatrixView panel, DenseMatrixView contribution) const;
    // factorFront without the trailing update (DenseKernels::factorPanelWithoutPivoting).
    [[nodiscard]] std::size_t factorPanel(DenseMatrixView panel) const;

    // Factorises `a` with Bunch-Kaufman pivoting among its first candidateCount columns (permutation and subdiagonal:
    // one value per candidate, filled here); returns the number of columns eliminated, the delayed candidates
    // following them (PivotedLdltOperands).
    [[nodiscard]] std::size_t factorWithPivoting(DenseMatrixView a, std::size_t candidateCount,
                                                 std::span<std::size_t> permutation,
                                                 std::span<double> subdiagonal) const;

private:
    [[nodiscard]] std::size_t factorBlocked(DenseMatrixView a, std::size_t pivotColumnCount,
                                            std::size_t blockColumns) const;
    // Eliminates the columns [first, end) of `a` over all their rows below the diagonal (down to a.rows), given that
    // they have received the terms of every column before `first`; the columns from `end` on receive nothing. A part
    // wider than LDLT_BLOCK_COLUMNS is split at a multiple of LDLT_BLOCK_COLUMNS near its middle (k-blocking: see the
    // PERFORMANCE note in the .cpp). Returns the columns eliminated (fewer at a zero or non-finite pivot).
    [[nodiscard]] std::size_t factorColumns(DenseMatrixView a, std::size_t first, std::size_t end,
                                            std::vector<double>& productRow) const;
    // The columns [first, end) of a block: the diagonal block factorised (by inner blocks when wider than
    // INNER_BLOCK_COLUMNS), then the rows below it solved; returns the columns eliminated in the block.
    [[nodiscard]] std::size_t factorBlockColumns(DenseMatrixView a, std::size_t first, std::size_t end,
                                                 std::vector<double>& productRow) const;

    const DenseKernels& m_kernels;
    const KernelFamily& m_family;
};

} // namespace oos::linalg
