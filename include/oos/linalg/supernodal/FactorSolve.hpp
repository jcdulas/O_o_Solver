#pragma once

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace oos::linalg
{

// Solves with a supernodal factor: A·X = B for many right-hand sides at once.
//
// X = L⁻ᵀ·D⁻¹·L⁻¹·B in three sweeps over the panels: forward (supernodes in increasing order: X_F ← L_FF⁻¹·X_F, then
// X_R ← X_R − L_RF·X_F on the rows below), diagonal (X_F ← D⁻¹·X_F, 1 × 1 and 2 × 2 pivots), backward (supernodes in
// decreasing order: X_F ← X_F − L_RFᵀ·X_R, then X_F ← L_FF⁻ᵀ·X_F). Every value is one chain of terms in a fixed order
// (the triangles' terms in increasing, then decreasing q; the rows below in panel order), whatever the number of
// columns, so a column has the same bits alone, in a block of 64 or among 6 000 load cases.
//
// How the sweeps run:
// - tiles: the right-hand sides are taken RHS_TILE_COLUMNS columns at a time (fewer when the workspace would exceed
//   SOLVE_WORKSPACE_BYTES), each tile copied unit by unit into blocks stored by rows (the values of an equation for the
//   tile's columns contiguous), on which the row kernels of DenseKernels (subtractRowProducts, solveRowTriangle) work
//   in place; the columns are padded with zeros to a multiple of the kernels' vector length;
// - units (SolvePlan): consecutive supernodes whose rows are gathered once into a block, swept there, and written back
//   once, instead of a gather and a scatter per supernode;
// - subtrees: with the independent subtrees of the factorisation (SupernodalFactor::subtrees, a cut that depends on
//   the tree only) and no delayed pivot, the forward sweep runs the subtrees in parallel, each accumulating its
//   contributions to the rows of its ancestors in an accumulator of its own (the update rows of its root), added to B
//   subtree by subtree in their order; then the top supernodes in order. The backward sweep runs the top supernodes in
//   decreasing order, then the subtrees in parallel. The sums of the rows of the top are thus grouped by subtree
//   (b_r + Σ_subtrees (−Σ_its supernodes L·x)), which fixes their rounding whatever the threads;
// - large supernodes (the top ones): their rows are shared between the threads (pieces of rows for the products,
//   pieces of columns for the diagonal blocks of the triangles), which never splits the terms of one value;
// - sparse right-hand sides: the forward sweep skips every supernode whose eliminated rows are zero in
//   every column of the tile and whose descendants all are (the reach of the nonzero rows in the elimination tree,
//   Gilbert 1994): a local load or the adjoint of a local response costs the path of its supernodes to the root, and
//   whole units are skipped. The values written are never −0 (x + 0 when a row is written back, and on the trailing
//   rows of a partial factor before the forward sweep), so the skipped supernodes, which only add ±0 to their rows,
//   change no bit: a sweep with skipped supernodes gives the bits of the full sweep, and a column the bits it has
//   alone.
// - the task flow: with the sweeps by subtrees on a pool of more than one thread, the
//   tiles of a call, their subtrees and their top parts run as one base::TaskFlow (sweeps::SolveFlow): several tiles
//   in flight within SOLVE_WORKSPACE_BYTES, each tile's top part gathered once into a block stored by rows and swept
//   there by tasks on blocks of rows, ordered by the rows each reads and writes (siblings together, a link's first
//   blocks as soon as their rows are final); one thread runs the sweeps above.
// The bits do not depend on the number of threads, on the tile width or on the units: each value keeps its chain.
//
// The right-hand sides are in the equation numbering of the factorised matrix. The three sweeps are also available
// apart, for the partial factorisation (condensed loads: the forward sweep of the interior; recovery: forward,
// diagonal and backward sweeps of the interior around given trailing values).
//
// Thread-safety: const; several solves may run at once. Ownership: references the factor and the kernels, which must
// outlive it. Memory: a solve's scratch (the tile's accumulators and a block per running task) is bounded by
// SOLVE_WORKSPACE_BYTES and allocated outside the memory budget, as the caller's right-hand sides are.
class FactorSolve
{
public:
    // The columns of a tile: a unit's block of SolvePlan::UNIT_ROWS rows of 128 values takes 1 MB, the L2 of a
    // Golden Cove core (1.25 MB); the panels are read once per tile and sweep.
    // PERFORMANCE: tiles of 64, 96, 128, 192 or 256 columns with blocks of about 1 MB solve within 5 % of each other;
    // blocks beyond the L2 (256 columns of 2 048 rows) lose 15 %.
    // Measured: the 1 M DOF shell and the 200 000 DOF 3D grid, 64 and 256 columns, six threads.
    static constexpr std::size_t RHS_TILE_COLUMNS = 128;
    // With the sweeps by subtrees, this many tiles run at once (each on every thread), so that the joins of one tile's
    // top supernodes are filled with the other's work; their accumulators are apart.
    // PERFORMANCE: two tiles in flight solve 256 and 512 columns 7 to 10 % faster than one after the other (1 M DOF
    // shell, 200 000 DOF 3D grid, six threads); four gain nothing more.
    static constexpr std::size_t TILES_IN_FLIGHT = 2;
    // The scratch of the tiles at most: the tile is narrowed (by halves) until the accumulators and the blocks of the
    // running tasks fit, so that a factor with large subtree boundaries does not take gigabytes of scratch.
    static constexpr std::uint64_t SOLVE_WORKSPACE_BYTES = std::uint64_t{512} << 20;

    FactorSolve(const SupernodalFactor& factor, const DenseKernels& kernels) : m_factor(factor), m_kernels(kernels) {}

    // B ← A⁻¹·B, B of dimension() rows. Throws std::invalid_argument when B has another row count or the factor is
    // partial; SingularityError when the factor has a zero pivot; OperationCancelledError between units.
    void solve(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation = nullptr) const;

    // The sweeps of solve(), over the eliminated columns of the factor (complete or partial).
    void eliminateForward(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation = nullptr) const;
    void solveDiagonal(DenseMatrixView rightHandSides, const base::CancellationToken* cancellation = nullptr) const;
    void substituteBackward(DenseMatrixView rightHandSides,
                            const base::CancellationToken* cancellation = nullptr) const;

    // The tile width the sweeps take for `columnCount` right-hand sides on this factor (for the tests and reports).
    [[nodiscard]] std::size_t tileColumns(std::size_t columnCount) const noexcept;

private:
    enum class Sweep
    {
        Forward,
        Diagonal,
        Backward
    };

    void runSweeps(DenseMatrixView rightHandSides, std::initializer_list<Sweep> sweeps,
                   const base::CancellationToken* cancellation) const;
    void requireRows(const char* operation, const DenseMatrixView& rightHandSides) const;

    const SupernodalFactor& m_factor;
    const DenseKernels& m_kernels;
};

} // namespace oos::linalg
