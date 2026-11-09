#include "oos/linalg/dense/DenseKernels.hpp"

#include "BlockedLdlt.hpp"
#include "KernelFamily.hpp"
#include "LdltArithmetic.hpp"
#include "TileRunner.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/memory/AlignedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"
#include "oos/linalg/dense/TaskedFront.hpp"
#include "oos/linalg/supernodal/UpdateLayout.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::linalg
{
namespace
{

// At the start of a factorisation, for every family. The blocked (SIMD) factorisations check again between their
// blocks and tiles; the unblocked reference family (the oracle, and the fallback without AVX2) runs a front in one
// call.
// TODO: checks inside the reference factorisations if a front on the fallback family must be interruptible;
// acceptable now because a front of the reference family on a machine without AVX2 lasts seconds at most at the sizes
// of the models met so far.
void throwIfCancelled(const DenseKernelOptions& options)
{
    if (options.cancellation != nullptr)
        options.cancellation->throwIfCancelled();
}

std::string describeShape(std::size_t rows, std::size_t columns)
{
    return std::to_string(rows).append(" × ").append(std::to_string(columns));
}

[[noreturn]] void throwShapeError(const char* operation, const char* operand, const std::string& expected,
                                  std::size_t rows, std::size_t columns)
{
    throw std::invalid_argument(base::concatenate({"DenseKernels::", operation, ": the matrix ", operand, " must be ",
                                                   expected, ", found ", describeShape(rows, columns)}));
}

// A view is usable when its leading dimension covers its rows and it has data when it is not empty.
void requireValidView(const char* operation, const char* operand, const ConstDenseMatrixView& view)
{
    if (view.leadingDimension < std::max<std::size_t>(view.rows, 1))
        throw std::invalid_argument(
            base::concatenate({"DenseKernels::", operation, ": the matrix ", operand, " has a leading dimension of ",
                               std::to_string(view.leadingDimension), ", below its ", std::to_string(view.rows),
                               " rows; pass the distance between two columns, at least the row count"}));
    if (view.data == nullptr && view.rows != 0 && view.columns != 0)
        throw std::invalid_argument(base::concatenate({"DenseKernels::", operation, ": the matrix ", operand, " (",
                                                       describeShape(view.rows, view.columns),
                                                       ") has no data; give a matrix with its values"}));
}

void requireSquare(const char* operation, const char* operand, const ConstDenseMatrixView& view)
{
    requireValidView(operation, operand, view);
    if (view.rows != view.columns)
        throwShapeError(operation, operand, "square", view.rows, view.columns);
}

void requireRows(const char* operation, const char* operand, const ConstDenseMatrixView& view, std::size_t rows)
{
    requireValidView(operation, operand, view);
    if (view.rows != rows)
        throwShapeError(operation, operand, std::string("of ").append(std::to_string(rows)).append(" rows"), view.rows,
                        view.columns);
}

// D must be the identity (no diagonal) or have `size` pivots, without a 2 × 2 block cut at its end.
void requireBlockDiagonal(const char* operation, const BlockDiagonalView& d, std::size_t size, bool isIdentityAllowed)
{
    if (d.diagonal == nullptr)
    {
        if (isIdentityAllowed || size == 0)
            return;
        throw std::invalid_argument(base::concatenate(
            {"DenseKernels::", operation, ": D has no diagonal, expected ", std::to_string(size), " pivots"}));
    }
    if (d.size != size)
        throw std::invalid_argument(
            base::concatenate({"DenseKernels::", operation, ": D has ", std::to_string(d.size), " pivots, expected ",
                               std::to_string(size), " (the inner dimension)"}));
    if (d.subdiagonal != nullptr && size > 0 && d.subdiagonal[size - 1] != 0.0)
        throw std::invalid_argument(
            base::concatenate({"DenseKernels::", operation, ": the last pivot of D starts a 2 × 2 block (e_",
                               std::to_string(size - 1), " != 0) that the operation cuts; pass the whole block"}));
}

std::size_t pieceCountOf(std::size_t count, std::size_t pieceSize)
{
    return (count + pieceSize - 1) / pieceSize;
}

// The tiles of a product a small result of which takes a long inner dimension are made smaller, down to
// KERNEL_TILE_GRANULE, until there are at least this many.
constexpr std::size_t LONG_PRODUCT_TILES = 16;

// PERFORMANCE: a small result over a long inner dimension (the projections Qᵀ·(M·W) of a block of seven Lanczos
// columns against its basis: a few hundred × 7 over 100 000 terms) was one tile of the default size, so one thread
// did it all; its tiles are made smaller until they are LONG_PRODUCT_TILES, from the shape alone (never from the
// threads, so that the bits do not depend on them). Every element of a result is one chain of terms whatever its tile
// (the arithmetic contract of the kernels), so the bits do not change.
// Measured: eigenvalue solve of 99 846 DOFs and 100 modes, six threads.
std::size_t tileSizeOf(std::size_t m, std::size_t n, std::size_t k, std::size_t tileSize)
{
    if (k <= tileSize)
        return tileSize;
    std::size_t tile = tileSize;
    while (tile > KERNEL_TILE_GRANULE && pieceCountOf(m, tile) * pieceCountOf(n, tile) < LONG_PRODUCT_TILES)
        tile -= KERNEL_TILE_GRANULE;
    return tile;
}

// The products on shared panels (subtractOnPanels): the columns of a block of Ŵ packed once and shared by the row
// pieces, in tiles of the options (10 tiles of 192: 1 920 columns of 256 terms, 3.9 MB, stay in the 18 MB L3 of the
// reference PC beside the threads' packed A of 384 kB in their L2).
constexpr std::size_t PANEL_TILES = 10;
// The columns of one piece of the shared panels' product, in tiles: the pieces are its rows by these groups, so that a
// large product has pieces for every thread between the synchronisations of its blocks of terms while each piece packs
// its rows of A for 384 columns at the default. Measured (six threads, against groups of four tiles):
// the update of 3 000 218-224 → 242-281 GFLOP/s, a front of 4 000 rows and 1 000 columns 198-206 → 218-232, the update
// of 1 000 165-170 → 178-211; one thread within 2 %.
constexpr std::size_t PANEL_GROUP_TILES = 2;
// A product whose result holds fewer values runs tile by tile: there the shared panels save little packing and cost a
// parallel phase per block of terms (the long products of few columns of the Lanczos projections).
constexpr std::size_t PANEL_MINIMUM_RESULT_VALUES =
    2 * DenseKernelOptions::DEFAULT_TILE_SIZE * DenseKernelOptions::DEFAULT_TILE_SIZE;

// The rows of one piece of the shared panels' product: half a tile (96 rows at the default), at least a granule
// (measured with groups of four tiles: 216 to 233 GFLOP/s on six threads at n = 2 000 against 191 to 209 with pieces of
// a whole tile, the same on one thread within 2 %).
std::size_t panelPieceRows(std::size_t tileSize)
{
    return std::max(KERNEL_TILE_GRANULE, tileSize / 2 / KERNEL_TILE_GRANULE * KERNEL_TILE_GRANULE);
}

// The terms of a block of the shared panels' product: ⌈k / KERNEL_INNER_BLOCK⌉ blocks of equal length, the rule of the
// SIMD families' tiles (kernels/SimdDenseKernels.ipp, innerBlockTerms; written twice: no function is shared with them).
std::size_t innerBlockTerms(std::size_t innerCount)
{
    const std::size_t blocks = (innerCount + KERNEL_INNER_BLOCK - 1) / KERNEL_INNER_BLOCK;
    return blocks == 0 ? 0 : (innerCount + blocks - 1) / blocks;
}

std::size_t roundedToGranule(std::size_t values)
{
    return (values + KERNEL_TILE_GRANULE - 1) / KERNEL_TILE_GRANULE * KERNEL_TILE_GRANULE;
}

// Whether a product of an m × n result runs on shared panels: a SIMD family, more than one tile, enough values.
bool isOnPanels(const KernelFamily& family, std::size_t m, std::size_t n, std::size_t tileSize)
{
    return family.isBlocked && (m > tileSize || n > tileSize) && m * n >= PANEL_MINIMUM_RESULT_VALUES;
}

// The part of the product `whole` at rows [firstRow, firstRow + rows) and columns [firstColumn, firstColumn + columns):
// the same operands seen from that corner (D and the terms unchanged).
ProductTileOperands blockOf(const ProductTileOperands& whole, std::size_t firstRow, std::size_t rows,
                            std::size_t firstColumn, std::size_t columns)
{
    ProductTileOperands block = whole;
    block.c = whole.c + firstRow + firstColumn * whole.ldc;
    block.rowCount = rows;
    block.columnCount = columns;
    block.a = whole.isATransposed ? whole.a + firstRow * whole.lda : whole.a + firstRow;
    block.b = whole.isBTransposed ? whole.b + firstColumn * whole.ldb : whole.b + firstColumn;
    block.diagonalOffset =
        whole.diagonalOffset + static_cast<std::ptrdiff_t>(firstRow) - static_cast<std::ptrdiff_t>(firstColumn);
    return block;
}

// PERFORMANCE: packed panels shared by the tasks (Goto & van de Geijn, "Anatomy of high-performance matrix
// multiplication", ACM TOMS 34, 2008, §6: the packed B in L3 for all the threads, each thread's packed A in its L2).
// Tile by tile, every tile packed its own rows of A and columns of W for each block of terms: an n × n update packed
// each value of its operands n/192 times, about a fifth of its time at n = 2 000 (the packing reads the operands
// column by column, strided, from L3 or memory). Here each block of terms packs the columns of W once, in pieces on
// the threads, and each piece of rows packs its A once and sweeps all those columns. The chain of every element is
// unchanged: its blocks of terms in increasing order, each the micro-kernel's sequence of terms.
// Measured (interleaved runs with the code before, one thread): the update 69-72 → 76-81 % of
// the peak at n = 1 000 and 2 000, a front of 4 000 rows and 1 000 columns 73 → 81 %.
void subtractOnPanels(const DenseKernelOptions& options, const KernelFamily& family, const ProductTileOperands& whole,
                      std::uint64_t multiplyAdds)
{
    const std::size_t m = whole.rowCount;
    const std::size_t n = whole.columnCount;
    const std::size_t k = whole.innerCount;
    const std::size_t tile = options.tileSize;
    const std::size_t blockTerms = innerBlockTerms(k);
    const std::size_t panelColumns = PANEL_TILES * tile;
    ScratchBuffers packedColumns(roundedToGranule(std::min(n, panelColumns)) * blockTerms, options.workspace);
    ScratchBuffer packedW = packedColumns.acquire();
    ScratchBuffers packedRows(roundedToGranule(std::min(m, panelPieceRows(tile))) * blockTerms, options.workspace);
    base::ThreadPool* pool = multiplyAdds >= options.parallelMinimumMultiplyAdds ? options.pool : nullptr;
    for (std::size_t firstTerm = 0; firstTerm < k; firstTerm += blockTerms)
    {
        const std::size_t termCount = std::min(blockTerms, k - firstTerm);
        for (std::size_t firstColumn = 0; firstColumn < n; firstColumn += panelColumns)
        {
            const std::size_t columns = std::min(panelColumns, n - firstColumn);
            // The rows that meet these columns: with isLowerOnly, row i reaches column i + diagonalOffset.
            const std::ptrdiff_t reach = static_cast<std::ptrdiff_t>(firstColumn) - whole.diagonalOffset;
            const std::size_t firstRow =
                whole.isLowerOnly ? std::min(m, static_cast<std::size_t>(std::max<std::ptrdiff_t>(reach, 0))) : 0;
            if (firstRow == m)
                continue;
            base::parallelFor(
                pool, pieceCountOf(columns, tile), 1,
                [&](std::size_t piece, std::size_t)
                {
                    const std::size_t first = piece * tile;
                    family.packProductColumns(
                        blockOf(whole, 0, 0, firstColumn + first, std::min(tile, columns - first)), firstTerm,
                        termCount, packedW.values.get() + first * termCount);
                },
                options.cancellation);
            // The pieces: tile rows by groups of PANEL_GROUP_TILES tiles of columns (each packs its rows of A once).
            const std::size_t pieceRows = panelPieceRows(tile);
            const std::size_t rowPieces = pieceCountOf(m - firstRow, pieceRows);
            const std::size_t groupColumns = PANEL_GROUP_TILES * tile;
            const std::size_t groups = pieceCountOf(columns, groupColumns);
            base::parallelFor(
                pool, rowPieces * groups, 1,
                [&](std::size_t piece, std::size_t)
                {
                    // The last rows first: below a diagonal they are the widest pieces.
                    const std::size_t row = firstRow + (rowPieces - 1 - piece / groups) * pieceRows;
                    const std::size_t rows = std::min(pieceRows, m - row);
                    const std::size_t first = piece % groups * groupColumns;
                    std::size_t reached = std::min(groupColumns, columns - first);
                    if (whole.isLowerOnly)
                    {
                        const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(row + rows) + whole.diagonalOffset -
                                                    static_cast<std::ptrdiff_t>(firstColumn + first);
                        reached = std::min(reached, static_cast<std::size_t>(std::max<std::ptrdiff_t>(last, 0)));
                    }
                    if (reached == 0)
                        return;
                    ScratchBuffer packedA = packedRows.acquire();
                    family.subtractPackedProduct(blockOf(whole, row, rows, firstColumn + first, reached), firstTerm,
                                                 termCount, packedW.values.get() + first * termCount,
                                                 packedA.values.get());
                    packedRows.release(std::move(packedA));
                },
                options.cancellation);
        }
    }
    packedColumns.release(std::move(packedW));
}

// PERFORMANCE: the LDLᵀ without pivoting of a large front as tasks (TaskedFront): the diagonal blocks,
// the row blocks of the panel solved and packed once, the tiles from the packed panels, the next panel block first
// (lookahead), the trailing block's tiles without a phase per block of columns or of terms. Every element keeps its
// chain: the bits of the recursion.
// Measured on six threads.
//
// The front's tasks when they are taken (a blocked family, a pool of more than one thread, a front of at least
// taskedFrontMinimumMultiplyAdds), the packed panels in `packing` or allocated here (reserved in options.budget when it
// is set); nothing when not taken or when the packed panels do not fit: the caller runs the recursion.
std::optional<std::size_t> factorByTasks(const DenseKernelOptions& options, KernelKind kind, DenseMatrixView panel,
                                         DenseMatrixView contribution, std::int64_t blockColumns,
                                         std::span<double> packing)
{
    const KernelFamily& family = kernelFamilyOf(kind);
    if (!family.isBlocked || family.packProductRows == nullptr || options.pool == nullptr ||
        options.pool->threadCount() < 2 || panel.columns == 0)
        return std::nullopt;
    const std::size_t columns = panel.columns;
    const std::size_t rowsBelow = panel.rows - columns;
    if (!TaskedFront::isWorthTasks(columns, rowsBelow, options.taskedFrontMinimumMultiplyAdds))
        return std::nullopt;
    const std::size_t values = TaskedFront::packingValues(columns, rowsBelow, blockColumns);
    base::MemoryReservation reservation;
    base::AlignedBuffer<double> own;
    if (packing.size() < values)
    {
        if (!packing.empty())
            return std::nullopt;
        try
        {
            if (options.budget != nullptr)
                reservation = options.budget->reserve(static_cast<std::uint64_t>(values) * sizeof(double),
                                                      "dense kernels: the packed panels of a front's tasks");
            own = base::AlignedBuffer<double>(values, base::FOR_OVERWRITE);
        }
        catch (const base::MemoryBudgetError&)
        {
            return std::nullopt;
        }
        packing = std::span<double>(own.data(), own.size());
    }
    TaskedFront front(kind, panel, contribution, blockColumns, packing, options.workspace);
    return front.run(options.pool, options.cancellation);
}

} // namespace

DenseKernels::DenseKernels(KernelKind kind, DenseKernelOptions options) : m_kind(kind), m_options(options)
{
    const std::size_t tile = m_options.tileSize;
    if (tile < KERNEL_TILE_GRANULE || tile > KERNEL_MAX_TILE || tile % KERNEL_TILE_GRANULE != 0)
        throw std::invalid_argument(
            base::concatenate({"DenseKernels: the tile size must be a multiple of ",
                               std::to_string(KERNEL_TILE_GRANULE), " between ", std::to_string(KERNEL_TILE_GRANULE),
                               " and ", std::to_string(KERNEL_MAX_TILE), ", found ", std::to_string(tile)}));
}

DenseKernels::DenseKernels(const KernelSelection& selection, DenseKernelOptions options)
    : DenseKernels(selection.kind(), options)
{
}

void DenseKernels::updateSymmetric(DenseMatrixView c, ConstDenseMatrixView a, const BlockDiagonalView& d) const
{
    requireSquare("updateSymmetric", "C", c);
    requireRows("updateSymmetric", "A", a, c.rows);
    requireBlockDiagonal("updateSymmetric", d, a.columns, true);
    const std::size_t n = c.rows;
    const std::size_t k = a.columns;
    if (n == 0 || k == 0)
        return;

    const KernelFamily& family = kernelFamilyOf(m_kind);
    if (isOnPanels(family, n, n, m_options.tileSize))
    {
        ProductTileOperands whole;
        whole.c = c.data;
        whole.ldc = c.leadingDimension;
        whole.rowCount = n;
        whole.columnCount = n;
        whole.a = a.data;
        whole.lda = a.leadingDimension;
        whole.b = a.data;
        whole.ldb = a.leadingDimension;
        whole.d = d;
        whole.innerCount = k;
        whole.isLowerOnly = true;
        subtractOnPanels(m_options, family, whole, symmetricUpdateMultiplyAdds(n, k));
        return;
    }
    const std::vector<Tile> tiles = makeTiles(n, n, m_options.tileSize, true);
    TileRunner runner(m_options, family, k, n);
    runner.run(tiles.size(), symmetricUpdateMultiplyAdds(n, k),
               [&](std::size_t piece, const PackingWorkspace& workspace)
               {
                   const Tile& tile = tiles[piece];
                   ProductTileOperands operands;
                   operands.c = &c(tile.firstRow, tile.firstColumn);
                   operands.ldc = c.leadingDimension;
                   operands.rowCount = tile.rowCount;
                   operands.columnCount = tile.columnCount;
                   operands.a = &a.data[tile.firstRow];
                   operands.lda = a.leadingDimension;
                   operands.b = &a.data[tile.firstColumn];
                   operands.ldb = a.leadingDimension;
                   operands.d = d;
                   operands.innerCount = k;
                   // Only a tile that reaches above the diagonal needs the element test.
                   operands.isLowerOnly = tile.firstRow < tile.firstColumn + tile.columnCount - 1;
                   operands.diagonalOffset =
                       static_cast<std::ptrdiff_t>(tile.firstRow) - static_cast<std::ptrdiff_t>(tile.firstColumn);
                   family.subtractProductTile(operands, workspace);
               });
}

void DenseKernels::updateRectangular(DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b,
                                     const BlockDiagonalView& d) const
{
    requireValidView("updateRectangular", "C", c);
    requireRows("updateRectangular", "A", a, c.rows);
    requireRows("updateRectangular", "B", b, c.columns);
    if (b.columns != a.columns)
        throwShapeError("updateRectangular", "B",
                        std::string("of ").append(std::to_string(a.columns)).append(" columns like A"), b.rows,
                        b.columns);
    requireBlockDiagonal("updateRectangular", d, a.columns, true);
    const std::size_t m = c.rows;
    const std::size_t n = c.columns;
    const std::size_t k = a.columns;
    if (m == 0 || n == 0 || k == 0)
        return;

    const KernelFamily& family = kernelFamilyOf(m_kind);
    if (isOnPanels(family, m, n, m_options.tileSize))
    {
        ProductTileOperands whole;
        whole.c = c.data;
        whole.ldc = c.leadingDimension;
        whole.rowCount = m;
        whole.columnCount = n;
        whole.a = a.data;
        whole.lda = a.leadingDimension;
        whole.b = b.data;
        whole.ldb = b.leadingDimension;
        whole.d = d;
        whole.innerCount = k;
        subtractOnPanels(m_options, family, whole, static_cast<std::uint64_t>(m) * n * k);
        return;
    }
    const std::vector<Tile> tiles = makeTiles(m, n, tileSizeOf(m, n, k, m_options.tileSize), false);
    TileRunner runner(m_options, family, k, std::max(m, n));
    runner.run(tiles.size(), static_cast<std::uint64_t>(m) * n * k,
               [&](std::size_t piece, const PackingWorkspace& workspace)
               {
                   const Tile& tile = tiles[piece];
                   ProductTileOperands operands;
                   operands.c = &c(tile.firstRow, tile.firstColumn);
                   operands.ldc = c.leadingDimension;
                   operands.rowCount = tile.rowCount;
                   operands.columnCount = tile.columnCount;
                   operands.a = &a.data[tile.firstRow];
                   operands.lda = a.leadingDimension;
                   operands.b = &b.data[tile.firstColumn];
                   operands.ldb = b.leadingDimension;
                   operands.d = d;
                   operands.innerCount = k;
                   family.subtractProductTile(operands, workspace);
               });
}

LdltOutcome DenseKernels::factorWithoutPivoting(DenseMatrixView a, std::size_t pivotColumnCount,
                                                std::span<double> pivotRatios) const
{
    requireSquare("factorWithoutPivoting", "A", a);
    if (pivotColumnCount > a.rows)
        throw std::invalid_argument(
            base::concatenate({"DenseKernels::factorWithoutPivoting: ", std::to_string(pivotColumnCount),
                               " columns to eliminate in a matrix of order ", std::to_string(a.rows),
                               "; at most the order (the full factorisation)"}));
    if (!pivotRatios.empty() && pivotRatios.size() < pivotColumnCount)
        throw std::invalid_argument(base::concatenate(
            {"DenseKernels::factorWithoutPivoting: room for ", std::to_string(pivotRatios.size()),
             " pivot ratios, expected one per eliminated column (", std::to_string(pivotColumnCount), ")"}));

    throwIfCancelled(m_options);
    std::vector<double> originalDiagonal(pivotRatios.empty() ? 0 : pivotColumnCount);
    for (std::size_t j = 0; j < originalDiagonal.size(); ++j)
        originalDiagonal[j] = a(j, j);

    const KernelFamily& family = kernelFamilyOf(m_kind);
    std::size_t eliminated = 0;
    const std::size_t n = a.rows;
    const std::optional<std::size_t> tasked =
        factorByTasks(m_options, m_kind, a.block(0, 0, n, pivotColumnCount),
                      a.block(pivotColumnCount, pivotColumnCount, n - pivotColumnCount, n - pivotColumnCount), 0, {});
    if (tasked)
        eliminated = *tasked;
    else if (family.isBlocked)
        eliminated = BlockedLdlt(*this, family).factorWithoutPivoting(a, pivotColumnCount);
    else
        eliminated = kernels::reference::factorWithoutPivoting(
            LdltOperands{a.data, a.leadingDimension, a.rows, pivotColumnCount});

    LdltOutcome outcome;
    outcome.eliminatedCount = eliminated;
    outcome.isComplete = eliminated == pivotColumnCount;
    outcome.inertia = inertiaOf(BlockDiagonalView{a.data, a.leadingDimension + 1, nullptr, eliminated});
    for (std::size_t j = 0; j < originalDiagonal.size() && j < eliminated; ++j)
        pivotRatios[j] = originalDiagonal[j] / a(j, j);
    return outcome;
}

LdltOutcome DenseKernels::factorFrontWithoutPivoting(DenseMatrixView panel, DenseMatrixView contribution) const
{
    requireValidView("factorFrontWithoutPivoting", "of the panel", panel);
    requireSquare("factorFrontWithoutPivoting", "of the trailing block", contribution);
    if (panel.rows < panel.columns || contribution.rows != panel.rows - panel.columns)
        throwShapeError("factorFrontWithoutPivoting", "of the trailing block",
                        panel.rows < panel.columns
                            ? std::string("below a panel of at least as many rows as columns")
                            : describeShape(panel.rows - panel.columns, panel.rows - panel.columns)
                                  .append(" (the panel's rows below its diagonal block)"),
                        contribution.rows, contribution.columns);
    throwIfCancelled(m_options);

    const KernelFamily& family = kernelFamilyOf(m_kind);
    const std::size_t columns = panel.columns;
    std::size_t eliminated = 0;
    const std::optional<std::size_t> tasked = factorByTasks(m_options, m_kind, panel, contribution, 0, {});
    if (tasked)
        eliminated = *tasked;
    else if (family.isBlocked)
        eliminated = BlockedLdlt(*this, family).factorFront(panel, contribution);
    else
        eliminated = kernels::reference::factorFrontWithoutPivoting(
            SplitLdltOperands{panel.data, panel.leadingDimension, contribution.data,
                              std::max<std::size_t>(contribution.leadingDimension, 1), columns, contribution.rows});

    LdltOutcome outcome;
    outcome.eliminatedCount = eliminated;
    outcome.isComplete = eliminated == columns;
    outcome.inertia = inertiaOf(BlockDiagonalView{panel.data, panel.leadingDimension + 1, nullptr, eliminated});
    return outcome;
}

LdltOutcome DenseKernels::factorFrontWithoutPivoting(DenseMatrixView panel, DenseMatrixView contribution,
                                                     std::int64_t contributionBlockColumns,
                                                     std::span<double> packing) const
{
    const std::size_t rowsBelow = panel.rows >= panel.columns ? panel.rows - panel.columns : 0;
    const UpdateLayout layout{static_cast<std::int64_t>(rowsBelow), contributionBlockColumns};
    if (!layout.isBlocked())
        return factorFrontWithoutPivoting(panel, contribution);
    requireValidView("factorFrontWithoutPivoting", "of the panel", panel);
    if (panel.rows < panel.columns || contribution.rows != rowsBelow || contribution.columns != rowsBelow)
        throwShapeError("factorFrontWithoutPivoting", "of the trailing block (by blocks)",
                        describeShape(rowsBelow, rowsBelow).append(" (the panel's rows below its diagonal block)"),
                        contribution.rows, contribution.columns);
    throwIfCancelled(m_options);
    const std::size_t columns = panel.columns;
    std::size_t eliminated = 0;
    const std::optional<std::size_t> tasked =
        factorByTasks(m_options, m_kind, panel, contribution, contributionBlockColumns, packing);
    if (tasked)
        eliminated = *tasked;
    else
    {
        eliminated = factorPanelWithoutPivoting(panel).eliminatedCount;
        if (eliminated == columns)
        {
            // F_rr − L_r·D·L_rᵀ block by block: the diagonal block of each (its lower triangle), then the rows below
            // it.
            const ConstDenseMatrixView rowsBelowPanel = panel.block(columns, 0, rowsBelow, columns);
            const BlockDiagonalView pivots = blockDiagonalOf(panel.block(0, 0, columns, columns));
            const auto width = static_cast<std::size_t>(contributionBlockColumns);
            for (std::size_t first = 0; first < rowsBelow; first += width)
            {
                const std::size_t columnsOfBlock = std::min(width, rowsBelow - first);
                const std::size_t rowsOfBlock = rowsBelow - first;
                const DenseMatrixView block{
                    contribution.data + layout.blockOffset(static_cast<std::int64_t>(first / width)), rowsOfBlock,
                    columnsOfBlock, rowsOfBlock};
                updateSymmetric(block.block(0, 0, columnsOfBlock, columnsOfBlock),
                                rowsBelowPanel.block(first, 0, columnsOfBlock, columns), pivots);
                if (rowsOfBlock > columnsOfBlock)
                    updateRectangular(
                        block.block(columnsOfBlock, 0, rowsOfBlock - columnsOfBlock, columnsOfBlock),
                        rowsBelowPanel.block(first + columnsOfBlock, 0, rowsOfBlock - columnsOfBlock, columns),
                        rowsBelowPanel.block(first, 0, columnsOfBlock, columns), pivots);
            }
        }
    }
    LdltOutcome outcome;
    outcome.eliminatedCount = eliminated;
    outcome.isComplete = eliminated == columns;
    outcome.inertia = inertiaOf(BlockDiagonalView{panel.data, panel.leadingDimension + 1, nullptr, eliminated});
    return outcome;
}

LdltOutcome DenseKernels::factorPanelWithoutPivoting(DenseMatrixView panel) const
{
    requireValidView("factorPanelWithoutPivoting", "of the panel", panel);
    if (panel.rows < panel.columns)
        throwShapeError("factorPanelWithoutPivoting", "of the panel",
                        std::string("of at least as many rows as columns"), panel.rows, panel.columns);
    throwIfCancelled(m_options);
    const KernelFamily& family = kernelFamilyOf(m_kind);
    const std::size_t columns = panel.columns;
    std::size_t eliminated = 0;
    if (family.isBlocked)
        eliminated = BlockedLdlt(*this, family).factorPanel(panel);
    else
        eliminated = kernels::reference::factorFrontWithoutPivoting(
            SplitLdltOperands{panel.data, panel.leadingDimension, nullptr, 1, columns, panel.rows - columns});
    LdltOutcome outcome;
    outcome.eliminatedCount = eliminated;
    outcome.isComplete = eliminated == columns;
    outcome.inertia = inertiaOf(BlockDiagonalView{panel.data, panel.leadingDimension + 1, nullptr, eliminated});
    return outcome;
}

PivotedLdlt DenseKernels::factorWithPivoting(DenseMatrixView a) const
{
    requireSquare("factorWithPivoting", "A", a);
    return factorWithPivoting(a, a.rows);
}

PivotedLdlt DenseKernels::factorWithPivoting(DenseMatrixView a, std::size_t candidateCount) const
{
    requireSquare("factorWithPivoting", "A", a);
    const std::size_t n = a.rows;
    if (candidateCount > n)
        throw std::invalid_argument(base::concatenate(
            {"DenseKernels::factorWithPivoting: ", std::to_string(candidateCount),
             " candidate pivot columns in a matrix of order ", std::to_string(n), "; at most the order"}));
    throwIfCancelled(m_options);
    PivotedLdlt result;
    result.permutation.resize(candidateCount);
    result.subdiagonal.assign(candidateCount, 0.0);

    const KernelFamily& family = kernelFamilyOf(m_kind);
    if (family.isBlocked)
        result.eliminatedCount =
            BlockedLdlt(*this, family).factorWithPivoting(a, candidateCount, result.permutation, result.subdiagonal);
    else
        result.eliminatedCount = kernels::reference::factorWithPivoting(PivotedLdltOperands{
            a.data, a.leadingDimension, n, result.permutation.data(), result.subdiagonal.data(), candidateCount});

    BlockDiagonalView d = blockDiagonalOf(a, result.subdiagonal.data());
    d.size = result.eliminatedCount;
    result.inertia = inertiaOf(d);
    for (std::size_t p = 0; p < d.size; ++p)
        if (startsTwoByTwoBlock(d, p))
            ++result.twoByTwoBlockCount;
    return result;
}

void DenseKernels::solvePanel(DenseMatrixView panel, ConstDenseMatrixView factored, const BlockDiagonalView& d) const
{
    requireSquare("solvePanel", "of the factored block", factored);
    requireValidView("solvePanel", "P", panel);
    if (panel.columns != factored.rows)
        throwShapeError("solvePanel", "P", std::string("of ").append(std::to_string(factored.rows)).append(" columns"),
                        panel.rows, panel.columns);
    requireBlockDiagonal("solvePanel", d, factored.rows, false);
    const std::size_t m = panel.rows;
    const std::size_t b = panel.columns;
    // The W block below is b² values on the heap per call, outside the memory budget: the width of a panel is a block
    // of a blocked factorisation (at most a tile), never a size that grows with the model.
    if (b > KERNEL_MAX_TILE)
        throw std::invalid_argument(base::concatenate(
            {"DenseKernels::solvePanel: the block has ", std::to_string(b), " columns, at most KERNEL_MAX_TILE = ",
             std::to_string(KERNEL_MAX_TILE), " are accepted; solve a wider block in pieces of at most that width"}));
    if (m == 0 || b == 0)
        return;

    // W = L·D of the block (row j, columns q < j), with the expressions of every family.
    std::vector<double> w(b * b, 0.0);
    std::vector<double> row(b);
    for (std::size_t j = 1; j < b; ++j)
    {
        computeProductRow(factored, j, 0, d, std::span<double>(row.data(), j));
        for (std::size_t q = 0; q < j; ++q)
            w[j + q * b] = row[q];
    }

    const KernelFamily& family = kernelFamilyOf(m_kind);
    const std::size_t pieceRows = m_options.tileSize;
    TileRunner runner(m_options, family, 0, 0);
    runner.run(pieceCountOf(m, pieceRows), static_cast<std::uint64_t>(m) * b * b / 2,
               [&](std::size_t piece, const PackingWorkspace&)
               {
                   const std::size_t firstRow = piece * pieceRows;
                   PanelSolveOperands operands;
                   operands.p = &panel(firstRow, 0);
                   operands.ldp = panel.leadingDimension;
                   operands.rowCount = std::min(pieceRows, m - firstRow);
                   operands.columnCount = b;
                   operands.w = w.data();
                   operands.ldw = b;
                   operands.d = d;
                   family.solvePanel(operands);
               });
}

void DenseKernels::solveUnitLower(ConstDenseMatrixView l, DenseMatrixView b) const
{
    requireSquare("solveUnitLower", "L", l);
    requireRows("solveUnitLower", "B", b, l.rows);
    const std::size_t n = l.rows;
    const std::size_t rhsCount = b.columns;
    if (n == 0 || rhsCount == 0)
        return;

    const KernelFamily& family = kernelFamilyOf(m_kind);
    const std::size_t tile = m_options.tileSize;
    // The triangle of rows [first, first + size), all right-hand sides, shared between tasks by columns of B.
    const auto solveTriangle = [&](std::size_t first, std::size_t size)
    {
        TileRunner runner(m_options, family, 0, 0);
        runner.run(pieceCountOf(rhsCount, tile), static_cast<std::uint64_t>(size) * size / 2 * rhsCount,
                   [&](std::size_t piece, const PackingWorkspace&)
                   {
                       const std::size_t firstColumn = piece * tile;
                       family.solveUnitLowerBlock(TriangleSolveOperands{l.pointer(first, first), l.leadingDimension,
                                                                        &b(first, firstColumn), b.leadingDimension,
                                                                        size, std::min(tile, rhsCount - firstColumn)});
                   });
    };
    if (!family.isBlocked)
    {
        solveTriangle(0, n);
        return;
    }
    // Blocked forward substitution: block of rows I = [first, first + size) receives at once the terms of all the
    // solved rows above it (B_I ← B_I − L_I,<I · X_<I, a tiled product with p increasing), then its own triangle:
    // each x(i) gets its terms in increasing q, as in the unblocked reference.
    for (std::size_t first = 0; first < n; first += tile)
    {
        const std::size_t size = std::min(tile, n - first);
        if (first > 0)
        {
            const std::vector<Tile> tiles = makeTiles(size, rhsCount, tile, false);
            TileRunner runner(m_options, family, first, std::max(size, rhsCount));
            runner.run(tiles.size(), static_cast<std::uint64_t>(size) * rhsCount * first,
                       [&](std::size_t piece, const PackingWorkspace& workspace)
                       {
                           const Tile& t = tiles[piece];
                           ProductTileOperands operands;
                           operands.c = &b(first + t.firstRow, t.firstColumn);
                           operands.ldc = b.leadingDimension;
                           operands.rowCount = t.rowCount;
                           operands.columnCount = t.columnCount;
                           operands.a = l.pointer(first + t.firstRow, 0);
                           operands.lda = l.leadingDimension;
                           // B̂(j, p) = x(p, j): the solved rows, read transposed.
                           operands.b = &b(0, t.firstColumn);
                           operands.ldb = b.leadingDimension;
                           operands.isBTransposed = true;
                           operands.innerCount = first;
                           family.subtractProductTile(operands, workspace);
                       });
        }
        solveTriangle(first, size);
    }
}

void DenseKernels::solveUnitLowerTransposed(ConstDenseMatrixView l, DenseMatrixView b) const
{
    requireSquare("solveUnitLowerTransposed", "L", l);
    requireRows("solveUnitLowerTransposed", "B", b, l.rows);
    const std::size_t n = l.rows;
    const std::size_t rhsCount = b.columns;
    if (n == 0 || rhsCount == 0)
        return;

    const KernelFamily& family = kernelFamilyOf(m_kind);
    const std::size_t tile = m_options.tileSize;
    const auto solveTriangle = [&](std::size_t first, std::size_t size)
    {
        TileRunner runner(m_options, family, 0, 0);
        runner.run(pieceCountOf(rhsCount, tile), static_cast<std::uint64_t>(size) * size / 2 * rhsCount,
                   [&](std::size_t piece, const PackingWorkspace&)
                   {
                       const std::size_t firstColumn = piece * tile;
                       family.solveUnitLowerTransposedBlock(
                           TriangleSolveOperands{l.pointer(first, first), l.leadingDimension, &b(first, firstColumn),
                                                 b.leadingDimension, size, std::min(tile, rhsCount - firstColumn)});
                   });
    };
    if (!family.isBlocked)
    {
        solveTriangle(0, n);
        return;
    }
    // Blocked backward substitution, from the last block of rows up: block I = [first, end) receives the terms of all
    // the solved rows below it in decreasing p (B_I ← B_I − (L_>I,I)ᵀ · X_>I, a tiled product with the inner index
    // reversed), then its own triangle, in decreasing q: the chain of the unblocked reference.
    std::size_t end = n;
    while (end > 0)
    {
        const std::size_t first = (end - 1) / tile * tile;
        const std::size_t size = end - first;
        if (end < n)
        {
            const std::vector<Tile> tiles = makeTiles(size, rhsCount, tile, false);
            TileRunner runner(m_options, family, n - end, std::max(size, rhsCount));
            runner.run(tiles.size(), static_cast<std::uint64_t>(size) * rhsCount * (n - end),
                       [&](std::size_t piece, const PackingWorkspace& workspace)
                       {
                           const Tile& t = tiles[piece];
                           ProductTileOperands operands;
                           operands.c = &b(first + t.firstRow, t.firstColumn);
                           operands.ldc = b.leadingDimension;
                           operands.rowCount = t.rowCount;
                           operands.columnCount = t.columnCount;
                           // Â(i, p) = l(end + p, first + i): the columns of the block, below it, read transposed.
                           operands.a = l.pointer(end, first + t.firstRow);
                           operands.lda = l.leadingDimension;
                           operands.isATransposed = true;
                           operands.b = &b(end, t.firstColumn);
                           operands.ldb = b.leadingDimension;
                           operands.isBTransposed = true;
                           operands.innerCount = n - end;
                           operands.isInnerReversed = true;
                           family.subtractProductTile(operands, workspace);
                       });
        }
        solveTriangle(first, size);
        end = first;
    }
}

void DenseKernels::solveBlockDiagonal(const BlockDiagonalView& d, DenseMatrixView b) const
{
    requireValidView("solveBlockDiagonal", "B", b);
    requireBlockDiagonal("solveBlockDiagonal", d, b.rows, false);
    const std::size_t n = b.rows;
    const std::size_t rhsCount = b.columns;
    if (n == 0 || rhsCount == 0)
        return;
    const std::size_t tile = m_options.tileSize;
    TileRunner runner(m_options, kernelFamilyOf(KernelKind::Reference), 0, 0);
    runner.run(pieceCountOf(rhsCount, tile), static_cast<std::uint64_t>(n) * rhsCount,
               [&](std::size_t piece, const PackingWorkspace&)
               {
                   const std::size_t endColumn = std::min(rhsCount, (piece + 1) * tile);
                   for (std::size_t column = piece * tile; column < endColumn; ++column)
                   {
                       std::size_t p = 0;
                       while (p < n)
                       {
                           if (startsTwoByTwoBlock(d, p))
                           {
                               const TwoByTwoInverse inverse =
                                   invertTwoByTwo(d.diagonal[p * d.diagonalStride], d.subdiagonal[p],
                                                  d.diagonal[(p + 1) * d.diagonalStride]);
                               const double r0 = b(p, column);
                               const double r1 = b(p + 1, column);
                               b(p, column) = inverse.s * (inverse.d11 * r0 - r1);
                               b(p + 1, column) = inverse.s * (inverse.d22 * r1 - r0);
                               p += 2;
                           }
                           else
                           {
                               b(p, column) = b(p, column) / d.diagonal[p * d.diagonalStride];
                               p += 1;
                           }
                       }
                   }
               });
}

void DenseKernels::subtractProduct(DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b) const
{
    requireValidView("subtractProduct", "C", c);
    requireRows("subtractProduct", "A", a, c.rows);
    requireValidView("subtractProduct", "B", b);
    if (b.rows != a.columns || b.columns != c.columns)
        throwShapeError("subtractProduct", "B",
                        describeShape(a.columns, c.columns)
                            .append(" (A is ")
                            .append(describeShape(a.rows, a.columns))
                            .append(", C is ")
                            .append(describeShape(c.rows, c.columns))
                            .append(")"),
                        b.rows, b.columns);
    subtractProductTiles(c, a, false, b);
}

void DenseKernels::subtractTransposedProduct(DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b) const
{
    requireValidView("subtractTransposedProduct", "C", c);
    requireValidView("subtractTransposedProduct", "A", a);
    requireValidView("subtractTransposedProduct", "B", b);
    if (a.columns != c.rows)
        throwShapeError("subtractTransposedProduct", "A",
                        std::string("of ").append(std::to_string(c.rows)).append(" columns (the rows of C)"), a.rows,
                        a.columns);
    if (b.rows != a.rows || b.columns != c.columns)
        throwShapeError("subtractTransposedProduct", "B", describeShape(a.rows, c.columns), b.rows, b.columns);
    subtractProductTiles(c, a, true, b);
}

void DenseKernels::subtractProductTiles(DenseMatrixView c, ConstDenseMatrixView a, bool isATransposed,
                                        ConstDenseMatrixView b) const
{
    const std::size_t m = c.rows;
    const std::size_t n = c.columns;
    const std::size_t k = b.rows;
    if (m == 0 || n == 0 || k == 0)
        return;
    const KernelFamily& family = kernelFamilyOf(m_kind);
    if (isOnPanels(family, m, n, m_options.tileSize))
    {
        ProductTileOperands whole;
        whole.c = c.data;
        whole.ldc = c.leadingDimension;
        whole.rowCount = m;
        whole.columnCount = n;
        whole.a = a.data;
        whole.lda = a.leadingDimension;
        whole.isATransposed = isATransposed;
        whole.b = b.data;
        whole.ldb = b.leadingDimension;
        whole.isBTransposed = true;
        whole.innerCount = k;
        subtractOnPanels(m_options, family, whole, static_cast<std::uint64_t>(m) * n * k);
        return;
    }
    const std::vector<Tile> tiles = makeTiles(m, n, tileSizeOf(m, n, k, m_options.tileSize), false);
    TileRunner runner(m_options, family, k, std::max(m, n));
    runner.run(tiles.size(), static_cast<std::uint64_t>(m) * n * k,
               [&](std::size_t piece, const PackingWorkspace& workspace)
               {
                   const Tile& tile = tiles[piece];
                   ProductTileOperands operands;
                   operands.c = &c(tile.firstRow, tile.firstColumn);
                   operands.ldc = c.leadingDimension;
                   operands.rowCount = tile.rowCount;
                   operands.columnCount = tile.columnCount;
                   // Â(i, p) = a(i, p), or a(p, i) for the transposed product.
                   operands.a = isATransposed ? a.pointer(0, tile.firstRow) : a.pointer(tile.firstRow, 0);
                   operands.lda = a.leadingDimension;
                   operands.isATransposed = isATransposed;
                   // B̂(j, p) = b(p, j): the columns of B read as rows.
                   operands.b = b.pointer(0, tile.firstColumn);
                   operands.ldb = b.leadingDimension;
                   operands.isBTransposed = true;
                   operands.innerCount = k;
                   family.subtractProductTile(operands, workspace);
               });
}

std::size_t DenseKernels::rowLanes() const noexcept
{
    switch (m_kind)
    {
    case KernelKind::Reference:
        return 1;
    case KernelKind::Avx2:
        return 4;
    case KernelKind::Avx512:
        return 8;
    }
    return 1;
}

namespace
{

void requireRowBlock(const char* operation, std::size_t width, std::size_t ldx, std::size_t lanes)
{
    if (width % lanes != 0 || width > ldx)
        throw std::invalid_argument(base::concatenate(
            {"DenseKernels::", operation, ": rows of ", std::to_string(width), " values ", std::to_string(ldx),
             " apart; give a width that is a multiple of ", std::to_string(lanes), " and at most the distance"}));
}

} // namespace

void DenseKernels::subtractRowProducts(const RowProductOperands& operands) const
{
    requireRowBlock("subtractRowProducts", operands.width, operands.ldx, rowLanes());
    if (operands.width == 0 || operands.targetCount == 0 || operands.termCount == 0)
        return;
    kernelFamilyOf(m_kind).subtractRowProducts(operands);
}

void DenseKernels::solveRowTriangle(const RowTriangleOperands& operands) const
{
    requireRowBlock("solveRowTriangle", operands.width, operands.ldx, rowLanes());
    if (operands.width == 0 || operands.size < 2)
        return;
    kernelFamilyOf(m_kind).solveRowTriangle(operands);
}

BlockDiagonalView blockDiagonalOf(ConstDenseMatrixView factored, const double* subdiagonal) noexcept
{
    return {factored.data, factored.leadingDimension + 1, subdiagonal, factored.rows};
}

std::uint64_t symmetricUpdateMultiplyAdds(std::size_t order, std::size_t innerCount) noexcept
{
    return static_cast<std::uint64_t>(order) * (order + 1) / 2 * innerCount;
}

std::uint64_t ldltMultiplyAdds(std::size_t order) noexcept
{
    // Σ_k (n − k)²/2 ≈ n³/6 multiply-adds (Golub & Van Loan §4.1.2: n³/3 flops).
    const auto n = static_cast<std::uint64_t>(order);
    return n * n * n / 6;
}

} // namespace oos::linalg
