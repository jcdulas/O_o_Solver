#pragma once

// Internal to linalg/supernodal: the steps of the sweeps of FactorSolve (forward, diagonal and backward steps on the
// blocks of the solve plan's units), shared with the streamed factorisation (StreamedFactorisation), whose
// forward sweep runs while the panels are made and whose backward sweep runs by regions of the tree. Every step keeps
// the chains of terms of FactorSolve's class comment: the same bits wherever it is called from.

#include "SolvePlan.hpp"

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace oos::linalg::sweeps
{

// The work of a unit of one supernode (multiply-adds over the padded tile) from which its rows are shared between the
// threads: below it, the joins of the threads cost more than they save.
inline constexpr std::uint64_t LARGE_SUPERNODE_MULTIPLY_ADDS = std::uint64_t{1} << 21;
// The target rows of one piece of a product of a large supernode.
inline constexpr std::size_t PIECE_TARGET_ROWS = 16;
// The rows of a diagonal block of the triangles of a large supernode: the threads join after each block.
inline constexpr std::size_t TRIANGLE_BLOCK_ROWS = 96;
// The columns of one piece of a diagonal block or of the pivots of a large supernode: a multiple of the vector length
// of every family.
inline constexpr std::size_t PIECE_COLUMNS = 32;
// The rows of one piece of a gather or a scatter of a large supernode.
inline constexpr std::size_t PIECE_COPY_ROWS = 256;
// The supernodes of one piece of the scan of the nonzero rows and of the diagonal sweep.
inline constexpr std::size_t PIECE_SUPERNODES = 256;
// The columns of one piece of the addition of the subtrees' accumulators.
inline constexpr std::size_t PIECE_ADD_COLUMNS = 8;
// Columns copied together between the right-hand sides and a block (one row of a block takes them contiguous).
inline constexpr std::size_t COPY_COLUMNS = 4;

// x + 0 is x but for −0, which becomes +0 (the FactorSolve class comment: no −0 is written). Not folded by the
// compiler: the project is built without -ffast-math (and so with signed zeros).
[[nodiscard]] inline double canonical(double value)
{
    return value + 0.0;
}

[[nodiscard]] inline std::size_t roundUp(std::size_t value, std::size_t multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

// The columns of the right-hand sides a tile solves, and the distance between two rows of its blocks (the width
// rounded up to the kernels' vector length; the columns beyond the width are zeros).
struct Tile
{
    DenseMatrixView b;
    std::size_t width = 0;
    std::size_t ldx = 0;
};

// Column j of the tile's right-hand sides.
inline double* columnOf(const Tile& tile, std::size_t j)
{
    return tile.b.data + j * tile.b.leadingDimension;
}

// What every step of a sweep reads.
struct SweepContext
{
    const SupernodalFactor& factor;
    const SolvePlan& plan;
    const DenseKernels& kernels;
    base::ThreadPool* pool; // null: everything on the calling thread
    const base::CancellationToken* cancellation;
    bool isDiagonalFused; // the forward sweep applies D⁻¹ to the rows it eliminates (solve())
};

// Blocks of rows for the tasks of a solve, reused across units and tiles; a block serves one task at a time.
// Thread-safe.
class BlockPool
{
public:
    explicit BlockPool(std::size_t values) : m_values(std::max<std::size_t>(values, 1)) {}

    [[nodiscard]] std::unique_ptr<double[]> acquire()
    {
        {
            const std::scoped_lock lock(m_mutex);
            if (!m_free.empty())
            {
                std::unique_ptr<double[]> block = std::move(m_free.back());
                m_free.pop_back();
                return block;
            }
        }
        // Uninitialised: a unit writes every value of its rows before reading it.
        return std::make_unique_for_overwrite<double[]>(m_values);
    }

    void release(std::unique_ptr<double[]> block)
    {
        const std::scoped_lock lock(m_mutex);
        m_free.push_back(std::move(block));
    }

private:
    std::size_t m_values;
    std::mutex m_mutex;
    std::vector<std::unique_ptr<double[]>> m_free; // at most one per thread
};

// A block taken from the pool for the life of a task.
class BlockLease
{
public:
    explicit BlockLease(BlockPool& pool) : m_pool(pool), m_block(pool.acquire()) {}
    BlockLease(const BlockLease&) = delete;
    BlockLease& operator=(const BlockLease&) = delete;
    ~BlockLease() { m_pool.release(std::move(m_block)); }

    [[nodiscard]] double* data() const noexcept { return m_block.get(); }

private:
    BlockPool& m_pool;
    std::unique_ptr<double[]> m_block;
};

void checkCancellation(const base::CancellationToken* cancellation);

// The rows of a unit read from its accumulator: the last ones (the rows beyond a subtree are its largest equations,
// and they come last in a unit's block), from the returned index on.
[[nodiscard]] std::size_t firstAccumulatorRow(std::span<const std::int32_t> accumulatorRows);

// Rows [first, end) of a unit's block from the right-hand sides (and from the accumulator from `accumulatorFrom` on),
// the padding columns zero.
void loadRows(const Tile& tile, std::span<const std::int32_t> equations, std::span<const std::int32_t> accumulatorRows,
              const double* accumulator, std::size_t accumulatorFrom, double* block, std::size_t first,
              std::size_t end);
// Rows [first, end) of a unit's block back to the right-hand sides and the accumulator, without −0.
void storeRows(const Tile& tile, std::span<const std::int32_t> equations, std::span<const std::int32_t> accumulatorRows,
               double* accumulator, std::size_t accumulatorFrom, const double* block, std::size_t first,
               std::size_t end);

// y ← D⁻¹·y on the eliminated rows of a panel in a block, columns [firstColumn, endColumn).
void applyDiagonal(const FactorPanel& panel, std::span<const std::int32_t> localRows, double* block, std::size_t ldx,
                   std::size_t firstColumn, std::size_t endColumn);
// The same on the pivots [firstPivot, endPivot) alone, which no 2 × 2 pivot straddles.
void applyDiagonal(const FactorPanel& panel, std::span<const std::int32_t> localRows, double* block, std::size_t ldx,
                   std::size_t firstColumn, std::size_t endColumn, std::size_t firstPivot, std::size_t endPivot);

// The forward and backward steps of a supernode on its unit's block.
void forwardSupernode(const SweepContext& context, const Tile& tile, double* block, std::int32_t s);
void backwardSupernode(const SweepContext& context, const Tile& tile, double* block, std::int32_t s);

// Whether a unit is a large supernode alone, its rows shared between the threads.
[[nodiscard]] bool isLargeSingleton(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit);

// The forward sweep of a unit on one thread (its active supernodes; its rows beyond its subtree in `accumulator`), and
// its backward sweep; the same of a large supernode alone on the threads of the context's pool.
void forwardUnit(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* accumulator,
                 std::span<const std::uint8_t> isActive, double* block);
void backwardUnit(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* block);
void forwardLarge(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* block);
void backwardLarge(const SweepContext& context, const Tile& tile, const SolvePlan::Unit& unit, double* block);

// The supernodes the forward sweep of a tile visits (its nonzero rows and their ancestors).
[[nodiscard]] std::vector<std::uint8_t> activeSupernodes(const SweepContext& context, const Tile& tile);

// The first step of a forward sweep: the trailing rows of a partial factor written without −0 (they are only ever added
// to, so that skipping a supernode changes no bit).
void canonicalizeTrailingRows(const SweepContext& context, const Tile& tile);
// The step of a forward sweep by subtrees between the subtrees and the top: each subtree's accumulator (the update rows
// of its root, plan.outsideRowCount() × tile.ldx in all) added to the right-hand sides in subtree order, the inactive
// subtrees skipped (isActive: activeSupernodes of the tile).
void addAccumulators(const SweepContext& context, const Tile& tile, const double* accumulators,
                     std::span<const std::uint8_t> isActive);
// The backward step of supernode s alone (the regions of a recovery, StreamedFactorisation): its panel's rows gathered
// into `block` (block row k is panel row k), swept as backwardUnit or backwardLarge sweep it (its rows shared between
// the threads when large), its eliminated rows written back: the same bits as the sweep by units.
void backwardAlone(const SweepContext& context, const Tile& tile, std::int32_t s, double* block);

} // namespace oos::linalg::sweeps
