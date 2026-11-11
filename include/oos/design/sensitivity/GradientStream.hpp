#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/design/sensitivity/SensitivitySystem.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace oos::design
{

// The columns (or rows) of G formed at once by default: a multiple of the
// columns of FactorSolve's tiles, as SENSITIVITY_BLOCK_COLUMNS.
inline constexpr std::size_t GRADIENT_STREAM_BLOCK = 256;

// How the streamed G is formed (the full mode is computed in blocks and streamed):
// - ByColumns: the direct method by blocks of variables, du/dx_j = K⁻¹·b_j for the block, every row of the block's
//   columns formed from them (G[:, block]); one solve per variable with a pseudo-load, per subcase;
// - ByRows: the adjoint method by blocks of items, λ_i = K⁻¹·c_iᵀ for the block (an equation item with its chained
//   weights, ChainedAdjoint), then its rows column block by column block (G[block, columns]); one solve per item that
//   reads the displacements.
enum class StreamOrientation : std::uint8_t
{
    ByColumns,
    ByRows
};

[[nodiscard]] std::string_view streamOrientationName(StreamOrientation orientation) noexcept;

// One block of G handed to the sink: rows [rowBegin, rowBegin + rowCount) (the items, in the row order of the stream)
// and columns [columnBegin, columnBegin + columnCount) (the independent variables), row-major. The values are valid
// during GradientSink::consume only.
struct GradientBlock
{
    std::size_t rowBegin = 0;
    std::size_t rowCount = 0;
    std::size_t columnBegin = 0;
    std::size_t columnCount = 0;
    std::span<const double> values;
};

// Where the blocks of G go (a plug-in point, called once per block): the caller's callable, a file, a matrix.
// consume() is called on the thread that runs the stream, in a fixed order (ByColumns: column blocks in increasing
// order; ByRows: row blocks in increasing order, each with its column blocks in increasing order); an exception it
// throws stops the stream and goes to the caller.
class GradientSink
{
public:
    GradientSink() = default;
    GradientSink(const GradientSink&) = delete;
    GradientSink& operator=(const GradientSink&) = delete;
    virtual ~GradientSink() = default;

    virtual void consume(const GradientBlock& block) = 0;
};

// What a stream is asked: the block (the columns or rows formed at once) and the method (Automatic: the orientation
// with fewer solves, ByColumns on a tie, as MethodChoice takes the direct order on a tie; Direct forces ByColumns,
// Adjoint and Chained force ByRows).
struct GradientStreamOptions
{
    std::size_t block = GRADIENT_STREAM_BLOCK;
    SensitivityMethod method = SensitivityMethod::Automatic;
    // Empty: every variable. A zero skips its field and shape linearisation but retains its zero output column.
    std::vector<std::uint8_t> isVariableWanted;
};

// The size of G and what its destination must hold, known from the counts alone (before any linearisation or solve).
struct GradientSize
{
    std::size_t rowCount = 0;    // every item of every evaluated response
    std::size_t columnCount = 0; // the independent variables
    std::uint64_t bytes = 0;     // rowCount × columnCount doubles
};

// The orientation and the costs of a stream, known before its first solve.
struct GradientStreamPlan
{
    GradientSize size;
    StreamOrientation orientation = StreamOrientation::ByColumns;
    std::int64_t solvesByColumns = 0; // Σ over the subcases of the variables with a pseudo-load there
    std::int64_t solvesByRows = 0;    // the items that read the displacements
    std::uint64_t blockPeakBytes = 0; // the largest memory the stream's own buffers hold at once (estimated)
};

// What a stream did.
struct GradientStreamReport
{
    GradientStreamPlan plan;
    std::int64_t solves = 0;
    std::int64_t blocks = 0;             // the blocks handed to the sink
    std::uint64_t measuredPeakBytes = 0; // the largest memory the stream's own buffers held at once
    SensitivityReport linearisation;     // the factorisations, field perturbations and differences made before
};

// One subcase of a stream: the factorised system of its constraint set and the index of the subcase in the
// pseudo-loads.
struct StreamSubcase
{
    std::int32_t label = 0;
    const SensitivitySystem* system = nullptr;
    std::size_t loadIndex = 0;
};

// What a stream reads: the linearised items (every item of every evaluated response, no item mask), the rows of G in
// their order, the subcases with their factors, the pseudo-loads, the number of independent variables.
struct GradientStreamInputs
{
    const LinearisedItems* linearised = nullptr;
    std::span<const ItemKey> rows;
    std::span<const StreamSubcase> subcases;
    const PseudoLoads* loads = nullptr;
    std::size_t columnCount = 0;
    // Shares the columns of a block (their pseudo-loads and reductions, ByColumns) between threads; null: one thread.
    base::ThreadPool* pool = nullptr;
};

// The full gradient mode streamed by blocks: K⁻¹·B (or K⁻¹·Cᵀ) is computed by blocks of columns, each block reduced
// at once to its rows or columns of G, then discarded, so that the derivatives of the whole displacement field are
// never stored.
//
// With the linearisation of StaticSensitivity (each item r: its displacement weights c_r and its explicit part
// ∂r/∂x as a sparse row), G[r, j] = ∂r/∂x_j + c_r·du/dx_j = ∂r/∂x_j + λ_rᵀ·b_j, formed:
// - ByColumns, per block of variables J: per subcase, K·du_j = b_j for the j of J with a pseudo-load there (one block
//   solve), then G[r, J] = ∂r/∂x_J + c_r·du_J for every row of that subcase; the block G[:, J] goes to the sink;
// - ByRows, per block of items I: per subcase, K·λ_i = c_iᵀ for the i of I of that subcase that read the displacements
//   (one block solve), then for every column block J, G[I, J] = ∂r/∂x_J + λ_Iᵀ·b_J (PseudoLoads::dotRange); each tile
//   goes to the sink.
// The global items (no displacement weights: weight, volume) are their explicit parts. Both orientations give the
// gradients of StaticSensitivity to round-off (the same weights, pseudo-loads and factor; only the grouping of the
// right-hand sides of a block solve differs).
//
// Memory: a block holds ByColumns rows × B + equations × B doubles, ByRows (gSize + equations) × B + B × B, plus a
// g-set vector and a column of equations per piece of the block's columns (at most eight pieces): the peak grows with
// the block and with the rows (ByColumns) or the model (ByRows), never with the number of columns of G; G itself is
// never held. The buffers are reserved in the budget before they are made.
//
// Determinism: the blocks, subcases and rows run in a fixed order; the columns of a block are prepared and reduced in
// parallel (ByColumns), each alone; the block solves are deterministic for any number of threads.
//
// Thread-safety: runs on the calling thread (the sink is called from it). Ownership: none kept.
class GradientStream
{
public:
    // The size of G for `rowCount` items and `columnCount` variables.
    [[nodiscard]] static GradientSize size(std::size_t rowCount, std::size_t columnCount) noexcept;

    // Refuses, before anything is computed, a destination of `capacityBytes` (memory or disk it can still take; a
    // callable sink takes the blocks one by one and holds no G: base::MemoryBudget::UNLIMITED) that cannot hold G:
    // OutputVolumeError naming the destination, the size of G and the remedies.
    static void requireDestination(const GradientSize& size, std::string_view destination, std::uint64_t capacityBytes);

    // The orientation and the costs. std::invalid_argument for a block of 0.
    [[nodiscard]] static GradientStreamPlan plan(const GradientStreamInputs& inputs,
                                                 const GradientStreamOptions& options);

    // Forms G block by block and hands each block to `sink`. Throws MemoryBudgetError (a block that does not fit),
    // what the sink throws, std::invalid_argument (a block of 0, a row whose subcase has no system).
    [[nodiscard]] static GradientStreamReport run(const GradientStreamInputs& inputs,
                                                  const GradientStreamOptions& options, GradientSink& sink,
                                                  base::MemoryBudget& budget);
};

} // namespace oos::design
