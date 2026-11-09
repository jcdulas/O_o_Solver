#pragma once

// Internal to linalg/supernodal: the sweeps of FactorSolve by subtrees as one task flow.

#include "SolvePlan.hpp"
#include "SolveSweeps.hpp"

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/TaskFlow.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::linalg::sweeps
{

// The rows of a top block: a top supernode's tasks work by blocks of this many rows of its panel (its diagonal blocks,
// the rows below it), each task on every column of a tile.
// PERFORMANCE: a task's operands (128 source rows and 128 target rows of a tile of 128 columns, 128 KB each) stay in
// the L2 of a core (1.25 MB) and match the row kernels' term block (ROW_TERM_BLOCK); a task is some 2 M multiply-adds,
// against the few microseconds a task of the flow costs.
// Measured on the design gradients of a large model, 50 cases.
inline constexpr std::size_t TOP_ROW_BLOCK = 128;
// A top supernode with less work than this for a tile (multiply-adds over its padded columns) is one task per sweep.
inline constexpr std::uint64_t TOP_TASK_MULTIPLY_ADDS = std::uint64_t{1} << 21;
// The rows of a piece of the gather and write-back of a top block.
inline constexpr std::size_t TOP_COPY_ROWS = 2048;

// The sweeps by subtrees of FactorSolve on one call's right-hand sides as one base::TaskFlow: the tiles of columns
// in flight together, each tile's subtrees as tasks (forward into their accumulators,
// backward from the rows of their ancestors), and its top part (the supernodes above the subtrees' cut) gathered once
// into a top block stored by rows, swept there by tasks on blocks of TOP_ROW_BLOCK rows and written back once.
//
// The tasks of a top supernode s (k eliminated rows in diagonal blocks D_I, the rows below in blocks B_c):
// - forward: T(s, I) the triangle of D_I; U(s, J → I) the terms of D_J on D_I (J < I); V(s, J → c) the terms of D_J on
//   B_c; P(s, I) the pivots of D_I (solve(): D⁻¹ fused) once every product has read D_I;
// - backward: W(s, I) the terms of the rows below on D_I; U'(s, J → I) the terms of D_J on D_I (J > I); T'(s, I) the
//   transposed triangle of D_I;
// - a supernode of little work (TOP_TASK_MULTIPLY_ADDS) one task per sweep.
// Each task depends on the last task that wrote each row it reads or writes (kept while the tasks are made in the
// order of the sweeps) and on the tasks of its chain: a row receives its terms in the order of the sweep in supernode
// order, each element the chain of FactorSolve's class comment, so the bits are those of the sweeps on one
// thread, whatever the threads and the timing. The top block keeps a value from one supernode to the next instead of
// writing it back between units: a zero may carry a sign a write-back would have cleared, which changes no nonzero
// result; every value written back is without −0.
//
// Thread-safety: one run at a time per object. Memory: the accumulators and top blocks of the tiles in flight and a
// block per running subtree task, outside the memory budget as FactorSolve's scratch is (SOLVE_WORKSPACE_BYTES bounds
// them through tilesInFlight).
class SolveFlow
{
public:
    SolveFlow(const SupernodalFactor& factor, const SolvePlan& plan, const DenseKernels& kernels,
              base::ThreadPool& pool, const base::CancellationToken* cancellation);

    // The rows of the top block (the panels' rows of the top supernodes and the subtrees' outside rows).
    [[nodiscard]] std::size_t topRowCount() const noexcept { return m_topRows.size(); }

    // The sweeps `isForward` (with D⁻¹ when isDiagonalFused) and `isBackward` of every tile of `width` columns of
    // `rightHandSides`, `inFlight` tiles at once.
    void run(DenseMatrixView rightHandSides, std::size_t width, std::size_t inFlight, bool isForward, bool isBackward,
             bool isDiagonalFused);

    // After run(): the tasks' wall time summed and the flow's wall time (the threads' occupancy of the flow).
    [[nodiscard]] double workSeconds() const noexcept { return m_workSeconds; }
    [[nodiscard]] double wallSeconds() const noexcept { return m_wallSeconds; }

private:
    struct Slot;
    class TileBuilder;

    const SupernodalFactor& m_factor;
    const SolvePlan& m_plan;
    const DenseKernels& m_kernels;
    base::ThreadPool& m_pool;
    const base::CancellationToken* m_cancellation;

    std::vector<std::int32_t> m_topSupernodes;    // increasing
    std::vector<std::int32_t> m_topRows;          // equations, increasing
    std::size_t m_topEliminatedRowCount = 0;      // the first top rows, eliminated by the top supernodes
    std::vector<std::int64_t> m_positionOffset;   // per top supernode + 1
    std::vector<std::int32_t> m_positions;        // per top supernode, its panel rows in the top block
    std::vector<std::int32_t> m_outsidePositions; // per subtree, its outside rows in the top block (plan's order)
    double m_workSeconds = 0.0;
    double m_wallSeconds = 0.0;
};

} // namespace oos::linalg::sweeps
