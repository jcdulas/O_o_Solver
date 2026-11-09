#pragma once

#include "oos/base/threading/TaskFlow.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::linalg
{

class KernelWorkspace;

// The LDLᵀ without pivoting of a front as tasks of a base::TaskFlow: the front's c panel columns over its n = c + r
// rows (`panel`, n × c) and its trailing block (`contribution`, r × r: square of its leading dimension, or by blocks of
// columns as UpdateLayout says), eliminated right-looking by panel blocks of at most KERNEL_INNER_BLOCK columns (one
// inner block of terms of the micro-kernel), the rows cut on the same grid:
//   F(k)       the diagonal block of panel block k eliminated (BlockedLdlt's recursion, on one thread);
//   S(k, m)    the rows of row block m below it solved against it (the panel kernel);
//   U(k, m, j) the tile (row block m, panel block j > k) receives block k's terms from the packed panels;
//   P(k, m)    once the panel is eliminated, the rows of trailing row block m packed for block k's terms (Â and Ŵ);
//   V(k, m, j) the tile (trailing row block m, trailing block j <= m) receives block k's terms, after V(k − 1, m, j).
// Lookahead: the tiles of panel block k + 1 come before the others of block k (priority), so F(k + 1) runs while they
// are computed. The trailing block receives its terms only once every panel column is eliminated (a zero or non-finite
// pivot leaves it untouched); at most two blocks' packed panels are alive.
//
// Bits: every element receives the chain of terms of BlockedLdlt (increasing p, the blocks of terms one after the
// other, the in-block terms by the diagonal block's recursion or the panel solve, then the division by its pivot):
// the result has the bits of DenseKernels::factorFrontWithoutPivoting on one thread, whatever the threads. At a
// failed diagonal block (a zero or non-finite pivot) every task that depends on it does nothing (each reads a flag its
// dependencies wrote) and the others finish; every task succeeds for the flow, so a supernode's hand-over after the
// front still runs: what a failed front holds does not depend on the threads either.
//
// Memory: the packed panels live in `packing` (packingValues() values, the caller's: budgeted by it); per task, the
// kernels' packing buffers of the diagonal block's recursion (`workspace`) and the L·D of a diagonal block (B² values).
//
// Thread-safety: the tasks write what they own (a diagonal block, the rows of a row block, a tile) and read what their
// dependencies wrote. Ownership: references the views, the packing and the workspace, which outlive the flow; the
// object outlives the flow's run.
class TaskedFront
{
public:
    using TaskId = base::TaskFlow::TaskId;
    // The columns of a block of a square trailing block (a multiple of every micro-tile height: no partial micro-tile
    // inside a tile).
    static constexpr std::size_t SQUARE_BLOCK_COLUMNS = 240;
    // The column blocks of the trailing block one task updates in a row block: one (measured with 2 and 4, the row
    // block's packed rows then read once for the group: the same rate within the run-to-run spread of ±15 % on six
    // threads).
    static constexpr std::size_t TRAILING_GROUP_BLOCKS = 1;

    // The packed panels' values for a front of `columns` panel columns and `rowsBelow` rows below them, its trailing
    // block by blocks of contributionBlockColumns (0: square).
    [[nodiscard]] static std::size_t packingValues(std::size_t columns, std::size_t rowsBelow,
                                                   std::int64_t contributionBlockColumns) noexcept;
    // The multiply-adds of the front's elimination (its panel and its trailing block).
    [[nodiscard]] static double multiplyAdds(std::size_t columns, std::size_t rowsBelow) noexcept;
    // The largest order of a front of more than one block of terms the tasks take. Beyond it the recursion's updates of
    // up to half the panel's columns at once read the trailing block fewer times than the tasks' block after block of
    // 256 terms. Measured: on six threads the LDLᵀ of 12 000 61 %
    // of the peak with the tasks against 59 % without, of 16 000 and 20 000 60 against 62 and 63 %, a front of 20 000
    // rows and 5 000 columns 62 against 65 %; a front of one block of terms (the streamed links) faster with the tasks
    // at every order measured (60 against 54 % at 20 000).
    static constexpr std::size_t MAXIMUM_MULTIBLOCK_ORDER = 12000;
    // Whether the tasks are taken for a front: its multiply-adds at least `minimumMultiplyAdds`, and one block of
    // terms or an order of at most MAXIMUM_MULTIBLOCK_ORDER.
    [[nodiscard]] static bool isWorthTasks(std::size_t columns, std::size_t rowsBelow,
                                           double minimumMultiplyAdds) noexcept;
    // The trailing block's blocks of columns, their first columns and rowsBelow: those of its storage when it is stored
    // by blocks (UpdateLayout), else of SQUARE_BLOCK_COLUMNS.
    [[nodiscard]] static std::vector<std::size_t> trailingBlockFirsts(std::size_t rowsBelow,
                                                                      std::int64_t contributionBlockColumns);

    // The kind must be a blocked family (avx2, avx512). packing holds at least packingValues() values.
    TaskedFront(KernelKind kind, DenseMatrixView panel, DenseMatrixView contribution,
                std::int64_t contributionBlockColumns, std::span<double> packing, KernelWorkspace* workspace);
    TaskedFront(const TaskedFront&) = delete;
    TaskedFront& operator=(const TaskedFront&) = delete;
    ~TaskedFront();

    // Adds the front's tasks to the flow: its panel's after every task of panelAfter (its panel assembled); the tiles
    // of trailing block j after trailingAfter[j] when given (its columns assembled; empty: none). Priorities from
    // priorityBase: the panel's tasks PANEL_PRIORITY above it (by block), the trailing tiles below it (by block of
    // terms, then of columns).
    void addTasks(base::TaskFlow& flow, std::span<const TaskId> panelAfter, std::span<const TaskId> trailingAfter,
                  std::int64_t priorityBase);
    static constexpr std::int64_t PANEL_PRIORITY = std::int64_t{1} << 40;

    // After addTasks: the task after which every panel column is eliminated (or the front has failed: eliminatedCount);
    // the task after which trailing block j holds its final values; the task after which the whole front is done.
    [[nodiscard]] TaskId panelDone() const noexcept { return m_panelDone; }
    [[nodiscard]] TaskId trailingBlockDone(std::size_t j) const noexcept { return m_trailingDone[j]; }
    [[nodiscard]] TaskId done() const noexcept { return m_done; }
    // The trailing block's column blocks: their count and the first column of each (relative to the trailing block).
    [[nodiscard]] std::size_t trailingBlockCount() const noexcept { return m_trailingFirst.size() - 1; }
    [[nodiscard]] std::size_t trailingBlockFirst(std::size_t j) const noexcept { return m_trailingFirst[j]; }

    // After the flow ran: the columns eliminated (all of them, or the index of the first zero or non-finite pivot).
    [[nodiscard]] std::size_t eliminatedCount() const noexcept;

    // Adds the tasks to a flow of its own and runs it on the pool: the eliminated columns.
    [[nodiscard]] std::size_t run(base::ThreadPool* pool, const base::CancellationToken* cancellation);

private:
    struct State;
    std::unique_ptr<State> m_state;
    std::vector<std::size_t> m_trailingFirst; // trailing blocks' first columns, and r
    std::vector<TaskId> m_trailingDone;
    TaskId m_panelDone = 0;
    TaskId m_done = 0;
};

} // namespace oos::linalg
