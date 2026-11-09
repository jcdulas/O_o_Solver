#include "MassOrthogonalizer.hpp"

#include "oos/base/threading/ParallelFor.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace oos::linalg
{

namespace
{

// The partial sums one batch of pieces keeps (512 kB) before they are added, in order, to the totals.
constexpr std::size_t PARTIAL_SUM_VALUES = std::size_t{1} << 16;
// The tasks of a sweep per thread of the pool: enough for the threads to even out, few enough for each to take
// several pieces (a task costs a few microseconds to hand out).
constexpr std::size_t TASKS_PER_THREAD = 4;
constexpr std::size_t MAXIMUM_TASKS = 64;
// The columns of W one tile of a projection holds, row by row (the coefficients of a column of L or Q with the eight
// columns side by side: the SIMD lanes of the processor).
constexpr std::size_t TILE_COLUMNS = 8;
// The interleaved chains of a single dot product over a piece: independent additions the processor overlaps.
constexpr std::size_t DOT_CHAINS = 4;

// Σ x_r·y_r over [first, end) in DOT_CHAINS interleaved chains (rows by their remainder modulo DOT_CHAINS, counted from
// `first`), added pairwise.
double pieceDot(const double* x, const double* y, std::size_t first, std::size_t end) noexcept
{
    std::array<double, DOT_CHAINS> sums{};
    std::size_t r = first;
    for (; r + DOT_CHAINS <= end; r += DOT_CHAINS)
    {
        for (std::size_t k = 0; k < DOT_CHAINS; ++k)
            sums[k] += x[r + k] * y[r + k];
    }
    for (std::size_t k = 0; r < end; ++r, ++k)
        sums[k] += x[r] * y[r];
    return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

// out[j] = Σ_{r < rows} q_r·tile(r, j) for j < used, the tile row-major with TILE_COLUMNS values a row: one chain per
// j in increasing r.
void tileCoefficients(const double* q, const double* tile, std::size_t rows, double* out, std::size_t used) noexcept
{
    std::array<double, TILE_COLUMNS> sums{};
    for (std::size_t r = 0; r < rows; ++r)
    {
        const double value = q[r];
        const double* row = tile + r * TILE_COLUMNS;
        for (std::size_t j = 0; j < TILE_COLUMNS; ++j)
            sums[j] += value * row[j];
    }
    std::copy_n(sums.begin(), used, out);
}

// The same for two columns q0 and q1 at once (their chains those of tileCoefficients: independent, so that the
// processor overlaps them, the tile read once for both).
void tileCoefficientsPair(const double* q0, const double* q1, const double* tile, std::size_t rows, double* out0,
                          double* out1, std::size_t used) noexcept
{
    std::array<double, TILE_COLUMNS> sums0{};
    std::array<double, TILE_COLUMNS> sums1{};
    for (std::size_t r = 0; r < rows; ++r)
    {
        const double value0 = q0[r];
        const double value1 = q1[r];
        const double* row = tile + r * TILE_COLUMNS;
        for (std::size_t j = 0; j < TILE_COLUMNS; ++j)
        {
            sums0[j] += value0 * row[j];
            sums1[j] += value1 * row[j];
        }
    }
    std::copy_n(sums0.begin(), used, out0);
    std::copy_n(sums1.begin(), used, out1);
}

// w_r ← w_r − Σ_{i < qCount} c_i·q(r, i) (one chain in increasing i) for r in [first, end); then out[i] = Σ_r
// y(r, i)·w_r of the updated rows for i < N (one chain per i in increasing r). N = 0 only subtracts.
template <std::size_t N>
void subtractRows(double* w, ConstDenseMatrixView q, const double* c, ConstDenseMatrixView y, std::size_t first,
                  std::size_t end, double* out) noexcept
{
    std::array<double, N + 1> sums{};
    for (std::size_t r = first; r < end; ++r)
    {
        double value = w[r];
        for (std::size_t i = 0; i < q.columns; ++i)
            value -= c[i] * q.data[r + i * q.leadingDimension];
        w[r] = value;
        for (std::size_t i = 0; i < N; ++i)
            sums[i] += y.data[r + i * y.leadingDimension] * value;
    }
    std::copy_n(sums.begin(), N, out);
}

// The same for any count of columns of y (the chains of subtractRows<N>).
void subtractRowsAny(double* w, ConstDenseMatrixView q, const double* c, ConstDenseMatrixView y, std::size_t first,
                     std::size_t end, double* out) noexcept
{
    switch (y.columns)
    {
    case 0:
        subtractRows<0>(w, q, c, y, first, end, out);
        return;
    case 1:
        subtractRows<1>(w, q, c, y, first, end, out);
        return;
    case 2:
        subtractRows<2>(w, q, c, y, first, end, out);
        return;
    case 3:
        subtractRows<3>(w, q, c, y, first, end, out);
        return;
    case 4:
        subtractRows<4>(w, q, c, y, first, end, out);
        return;
    case 5:
        subtractRows<5>(w, q, c, y, first, end, out);
        return;
    case 6:
        subtractRows<6>(w, q, c, y, first, end, out);
        return;
    case 7:
        subtractRows<7>(w, q, c, y, first, end, out);
        return;
    case 8:
        subtractRows<8>(w, q, c, y, first, end, out);
        return;
    default:
        std::fill_n(out, y.columns, 0.0);
        for (std::size_t r = first; r < end; ++r)
        {
            double value = w[r];
            for (std::size_t i = 0; i < q.columns; ++i)
                value -= c[i] * q.data[r + i * q.leadingDimension];
            w[r] = value;
            for (std::size_t i = 0; i < y.columns; ++i)
                out[i] += y.data[r + i * y.leadingDimension] * value;
        }
    }
}

DenseKernels pieceKernelsOf(const DenseKernels& kernels)
{
    DenseKernelOptions options = kernels.options();
    options.pool = nullptr;
    options.cancellation = nullptr;
    // Its packing buffers are its own (the kernels' workspace serves one call at a time).
    options.workspace = nullptr;
    return DenseKernels(kernels.kind(), options);
}

std::size_t threadsOf(const DenseKernels& kernels) noexcept
{
    const base::ThreadPool* pool = kernels.options().pool;
    return pool != nullptr ? pool->threadCount() : 1;
}

} // namespace

MassOrthogonalizer::MassOrthogonalizer(const ShiftedPencil& pencil, const DenseKernels& kernels,
                                       base::MemoryBudget& budget, DenseMatrixView scratch, std::size_t pieceRows)
    : m_pencil(pencil),
      m_kernels(kernels),
      m_budget(budget),
      m_pieceKernels(pieceKernelsOf(kernels)),
      m_scratch(scratch),
      m_pieceRows(pieceRows),
      m_taskCount(std::min(TASKS_PER_THREAD * threadsOf(kernels), MAXIMUM_TASKS))
{
    if (pieceRows == 0)
        throw std::invalid_argument("MassOrthogonalizer: the rows of a piece of the inner products must be at least 1");
    const auto dimension = static_cast<std::size_t>(pencil.dimension());
    const std::size_t tileRows = std::min(pieceRows, std::max<std::size_t>(dimension, 1));
    m_tiles = base::BudgetedBuffer<double>(budget, m_taskCount * tileRows * TILE_COLUMNS,
                                           "Lanczos: inner products by pieces", base::FOR_OVERWRITE);
    m_partialSums = base::BudgetedBuffer<double>(budget, PARTIAL_SUM_VALUES, "Lanczos: inner products by pieces",
                                                 base::FOR_OVERWRITE);
}

std::uint64_t MassOrthogonalizer::estimateBytes(std::size_t dimension, std::size_t pieceRows) noexcept
{
    const std::size_t tileRows = std::min(pieceRows, std::max<std::size_t>(dimension, 1));
    return (static_cast<std::uint64_t>(MAXIMUM_TASKS) * tileRows * TILE_COLUMNS + PARTIAL_SUM_VALUES) * sizeof(double);
}

template <typename Piece>
void MassOrthogonalizer::sumByPieces(std::size_t count, std::size_t width, double* totals, Piece&& piece)
{
    std::fill(totals, totals + width, 0.0);
    const std::size_t pieces = base::pieceCount(count, m_pieceRows);
    if (pieces == 0 || width == 0)
        return;
    if (m_partialSums.size() < width)
        m_partialSums =
            base::BudgetedBuffer<double>(m_budget, width, "Lanczos: inner products by pieces", base::FOR_OVERWRITE);
    const std::size_t batch = std::min(pieces, m_partialSums.size() / width);
    for (std::size_t firstPiece = 0; firstPiece < pieces; firstPiece += batch)
    {
        const std::size_t batchPieces = std::min(batch, pieces - firstPiece);
        const std::size_t piecesPerTask = base::pieceCount(batchPieces, m_taskCount);
        base::parallelFor(m_kernels.options().pool, batchPieces, piecesPerTask,
                          [&](std::size_t begin, std::size_t end)
                          {
                              const std::size_t task = begin / piecesPerTask;
                              for (std::size_t k = begin; k < end; ++k)
                              {
                                  const std::size_t first = (firstPiece + k) * m_pieceRows;
                                  piece(first, std::min(first + m_pieceRows, count), m_partialSums.data() + k * width,
                                        task);
                              }
                          });
        // The pieces' sums in increasing order (each total one chain over the pieces).
        for (std::size_t k = 0; k < batchPieces; ++k)
        {
            for (std::size_t v = 0; v < width; ++v)
                totals[v] += m_partialSums[k * width + v];
        }
    }
}

template <typename Body>
void MassOrthogonalizer::forPieces(std::size_t count, Body&& body) const
{
    // The values are each made alone (or each its chain over columns): any grouping of the rows gives the same bits.
    const std::size_t pieces = base::pieceCount(count, m_pieceRows);
    const std::size_t rowsPerTask = base::pieceCount(pieces, m_taskCount) * m_pieceRows;
    base::parallelFor(m_kernels.options().pool, count, std::max<std::size_t>(rowsPerTask, 1), body);
}

double MassOrthogonalizer::dot(const double* x, const double* y, std::size_t count)
{
    double total = 0.0;
    sumByPieces(count, 1, &total, [&](std::size_t first, std::size_t end, double* sum, std::size_t)
                { *sum = pieceDot(x, y, first, end); });
    return total;
}

void MassOrthogonalizer::coefficients(ConstDenseMatrixView y, const double* w, std::span<double> out)
{
    sumByPieces(y.rows, y.columns, out.data(),
                [&](std::size_t first, std::size_t end, double* sums, std::size_t)
                {
                    // One chain per column of y in increasing rows, eight columns side by side.
                    for (std::size_t i0 = 0; i0 < y.columns; i0 += TILE_COLUMNS)
                    {
                        const std::size_t used = std::min(TILE_COLUMNS, y.columns - i0);
                        std::array<double, TILE_COLUMNS> part{};
                        for (std::size_t r = first; r < end; ++r)
                        {
                            const double value = w[r];
                            for (std::size_t i = 0; i < used; ++i)
                                part[i] += y.data[r + (i0 + i) * y.leadingDimension] * value;
                        }
                        std::copy_n(part.begin(), used, sums + i0);
                    }
                });
}

void MassOrthogonalizer::subtractColumns(double* w, ConstDenseMatrixView q, std::span<const double> c,
                                         ConstDenseMatrixView y, std::span<double> out)
{
    if (y.columns == 0)
    {
        forPieces(q.rows,
                  [&](std::size_t first, std::size_t end) { subtractRows<0>(w, q, c.data(), y, first, end, nullptr); });
        return;
    }
    sumByPieces(q.rows, y.columns, out.data(), [&](std::size_t first, std::size_t end, double* sums, std::size_t)
                { subtractRowsAny(w, q, c.data(), y, first, end, sums); });
}

void MassOrthogonalizer::scale(double* x, double* y, double s, std::size_t count) const
{
    forPieces(count,
              [&](std::size_t first, std::size_t end)
              {
                  for (std::size_t r = first; r < end; ++r)
                      x[r] *= s;
                  if (y != nullptr)
                  {
                      for (std::size_t r = first; r < end; ++r)
                          y[r] *= s;
                  }
              });
}

void MassOrthogonalizer::multiplyColumn(const double* x, double* y) const
{
    const auto n = static_cast<std::size_t>(m_pencil.dimension());
    m_pencil.multiplyInnerProduct({x, n, 1, std::max<std::size_t>(n, 1)}, {y, n, 1, std::max<std::size_t>(n, 1)});
}

void MassOrthogonalizer::projectOut(DenseMatrixView w, const LockedVectors& lockedSet, ConstDenseMatrixView basis,
                                    std::vector<double>* coefficients, std::span<double> normsBefore)
{
    const ConstDenseMatrixView locked = lockedSet.resident();
    const std::size_t spooledCount = lockedSet.spooledCount();
    if (w.columns > m_scratch.columns)
        throw std::logic_error("MassOrthogonalizer::projectOut: the block is wider than the scratch");
    const std::size_t rows = w.rows;
    const std::size_t width = w.columns;
    const DenseMatrixView mw = m_scratch.block(0, 0, rows, width);
    m_pencil.multiplyInnerProduct(w, mw);
    const std::size_t residentCount = locked.columns;
    const std::size_t lockedCount = residentCount + spooledCount;
    const std::size_t basisCount = basis.columns;
    const std::size_t setCount = lockedCount + basisCount;
    const std::size_t norms = std::min(normsBefore.size(), width);
    // PERFORMANCE: the coefficients of the locked set and of the basis, and the norms before the pass, in one sweep
    // of the rows by pieces, every thread on pieces of its own: a piece of M·W copied row by row into a tile (eight
    // columns side by side, 64 kB in the L2 cache) while the columns of L and Q stream past it once; then the update
    // W − L·C_L − Q·C_Q by the same rows, L and Q read once more. Before: the coefficients as one dense product per set
    // (tiles of 16 columns of L or Q shared by the threads: 7 tasks for a basis of 105 columns on six threads, each a
    // chain over every row), the norms on one thread.
    // Measured: eigenvalue solve of 99 846 DOFs, 100 modes.
    const std::size_t valueCount = setCount * width + norms;
    std::vector<double> totals(valueCount, 0.0); // bounded by the locked set and the basis times the block
    const std::size_t tileRows = std::min(m_pieceRows, std::max<std::size_t>(rows, 1));
    if (spooledCount > 0 && m_spooledPieces.size() < m_taskCount * tileRows * spooledCount)
        m_spooledPieces = base::BudgetedBuffer<double>(m_budget, m_taskCount * tileRows * spooledCount,
                                                       "Lanczos: spooled eigenvector pieces", base::FOR_OVERWRITE);
    sumByPieces(rows, valueCount, totals.data(),
                [&](std::size_t first, std::size_t end, double* sums, std::size_t task)
                {
                    const std::size_t pieceRows = end - first;
                    double* tile = m_tiles.data() + task * tileRows * TILE_COLUMNS;
                    // The spooled columns' rows of this piece, read once for every tile of W.
                    double* spooled =
                        spooledCount > 0 ? m_spooledPieces.data() + task * tileRows * spooledCount : nullptr;
                    if (spooled != nullptr)
                        lockedSet.readPiece(first, end, spooled);
                    for (std::size_t j0 = 0; j0 < width; j0 += TILE_COLUMNS)
                    {
                        const std::size_t used = std::min(TILE_COLUMNS, width - j0);
                        for (std::size_t r = 0; r < pieceRows; ++r)
                        {
                            for (std::size_t j = 0; j < TILE_COLUMNS; ++j)
                                tile[r * TILE_COLUMNS + j] = j < used ? mw(first + r, j0 + j) : 0.0;
                        }
                        // The columns of L then Q, by pairs.
                        const auto column = [&](std::size_t i) -> const double*
                        {
                            if (i < residentCount)
                                return locked.pointer(first, i);
                            if (i < lockedCount)
                                return spooled + (i - residentCount) * pieceRows;
                            return basis.pointer(first, i - lockedCount);
                        };
                        std::size_t i = 0;
                        for (; i + 1 < setCount; i += 2)
                            tileCoefficientsPair(column(i), column(i + 1), tile, pieceRows, sums + i * width + j0,
                                                 sums + (i + 1) * width + j0, used);
                        if (i < setCount)
                            tileCoefficients(column(i), tile, pieceRows, sums + i * width + j0, used);
                    }
                    for (std::size_t j = 0; j < norms; ++j)
                        sums[setCount * width + j] = pieceDot(&w(0, j), &mw(0, j), first, end);
                });
    for (std::size_t j = 0; j < norms; ++j)
        normsBefore[j] = std::sqrt(std::max(0.0, totals[setCount * width + j]));
    // C_L and C_Q column-major (rows: the columns of L or Q), as the dense kernels take them.
    std::vector<double> lockedCoefficients(lockedCount * width, 0.0); // bounded by the modes times the block
    std::vector<double> basisCoefficients(basisCount * width, 0.0);   // bounded by the basis times the block
    for (std::size_t i = 0; i < lockedCount; ++i)
    {
        for (std::size_t j = 0; j < width; ++j)
            lockedCoefficients[i + j * lockedCount] = totals[i * width + j];
    }
    for (std::size_t i = 0; i < basisCount; ++i)
    {
        for (std::size_t j = 0; j < width; ++j)
            basisCoefficients[i + j * basisCount] = totals[(lockedCount + i) * width + j];
    }
    // W ← W − L·C_L − Q·C_Q by rows: each element of W its chain over L's columns, then Q's.
    const ConstDenseMatrixView lockedC(lockedCoefficients.data(), lockedCount, width,
                                       std::max<std::size_t>(lockedCount, 1));
    const ConstDenseMatrixView basisC(basisCoefficients.data(), basisCount, width,
                                      std::max<std::size_t>(basisCount, 1));
    // The spooled columns' coefficients: rows [residentCount, lockedCount) of C_L.
    const ConstDenseMatrixView residentC(lockedCoefficients.data(), residentCount, width,
                                         std::max<std::size_t>(lockedCount, 1));
    const ConstDenseMatrixView spooledC(lockedCoefficients.data() + residentCount, spooledCount, width,
                                        std::max<std::size_t>(lockedCount, 1));
    forPieces(
        rows,
        [&](std::size_t first, std::size_t end)
        {
            const DenseMatrixView part = w.block(first, 0, end - first, width);
            if (spooledCount == 0)
            {
                if (lockedCount > 0)
                    m_pieceKernels.subtractProduct(part, locked.block(first, 0, end - first, lockedCount), lockedC);
            }
            else
            {
                // The resident columns' terms, then the spooled ones' piece by piece: each value's chain over
                // L's columns in increasing order, as one product makes it, then Q's.
                if (residentCount > 0)
                    m_pieceKernels.subtractProduct(part, locked.block(first, 0, end - first, residentCount), residentC);
                std::vector<double> piece; // bounded by a piece of the spooled columns
                for (std::size_t at = first; at < end; at += m_pieceRows)
                {
                    const std::size_t pieceEnd = std::min(at + m_pieceRows, end);
                    piece.resize((pieceEnd - at) * spooledCount);
                    lockedSet.readPiece(at, pieceEnd, piece.data());
                    m_pieceKernels.subtractProduct(w.block(at, 0, pieceEnd - at, width),
                                                   ConstDenseMatrixView(piece.data(), pieceEnd - at, spooledCount,
                                                                        std::max<std::size_t>(pieceEnd - at, 1)),
                                                   spooledC);
                }
            }
            if (basisCount > 0)
                m_pieceKernels.subtractProduct(part, basis.block(first, 0, end - first, basisCount), basisC);
        });
    if (coefficients != nullptr)
        *coefficients = std::move(basisCoefficients);
}

} // namespace oos::linalg
