#include "oos/design/sensitivity/PseudoLoadDots.hpp"

#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace oos::design
{

namespace
{

// The variables of a piece of the final sums of addProducts (every column for each of them).
constexpr std::size_t PRODUCT_VARIABLES_PER_PIECE = 64;
// The tiles the pieces walk together between two joins.
constexpr std::size_t CHUNK_TILES = 8;

} // namespace

void PseudoLoadDots::addProducts(std::span<const std::size_t> columnSubcases, std::span<double* const> destinations,
                                 const TileFill& fill)
{
    const std::size_t n = m_loads->variableCount();
    const std::size_t columns = columnSubcases.size();
    if (columns == 0 || n == 0)
        return;
    const std::size_t g = m_gSize;
    for (const std::size_t s : columnSubcases)
        if (m_segments.count(s) == 0)
            m_segments.emplace(s, m_loads->segments(s, TILE_DOFS, g, *m_budget));
    // The columns by subcase (stable), cut into one piece per thread: the pieces never change a bit, each column being
    // its own chains.
    std::vector<std::size_t> order(columns); // bounded by a solve's columns
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return columnSubcases[a] < columnSubcases[b]; });
    const std::size_t threads = m_pool != nullptr ? m_pool->threadCount() : 1;
    const std::size_t pieceColumns = base::pieceCount(columns, std::min(threads, columns));
    const std::size_t pieces = base::pieceCount(columns, pieceColumns);
    // The accumulators: per piece and run of one subcase in it (a group), [source][column of the group]; each column's
    // first accumulator and stride, for the final sums.
    const std::size_t sources = m_loads->fieldCount() + (m_loads->hasVariableLoads() ? n : 0);
    struct Group
    {
        std::size_t subcase;
        std::size_t first; // in the piece
        std::size_t end;
        std::size_t offset; // of its accumulators
    };
    std::vector<std::vector<Group>> groups(pieces); // bounded by the columns
    std::vector<std::size_t> accumulatorOf(columns);
    std::vector<std::size_t> strideOf(columns);
    std::size_t offset = 0;
    for (std::size_t p = 0; p < pieces; ++p)
    {
        const std::size_t begin = p * pieceColumns;
        const std::size_t end = std::min(columns, begin + pieceColumns);
        for (std::size_t k = begin; k < end;)
        {
            std::size_t last = k + 1;
            while (last < end && columnSubcases[order[last]] == columnSubcases[order[k]])
                ++last;
            const std::size_t width = last - k;
            groups[p].push_back({columnSubcases[order[k]], k - begin, last - begin, offset});
            for (std::size_t q = k; q < last; ++q)
            {
                accumulatorOf[order[q]] = offset + (q - k);
                strideOf[order[q]] = width;
            }
            offset += sources * width;
            k = last;
        }
    }
    // Kept from one call to the next (the first touch of its pages paid once), zeroed on the pool.
    if (m_accumulators.size() < offset)
    {
        m_accumulators = base::BudgetedBuffer<double>(); // the memory back first
        m_accumulators =
            base::BudgetedBuffer<double>(*m_budget, offset, "adjoint column products", base::FOR_OVERWRITE);
    }
    base::fillOnPool(m_pool, std::span<double>(m_accumulators.data(), offset), 0.0);
    double* const accumulators = m_accumulators.data();
    const std::span<const base::DofIndex> readDofs = m_readDofs;
    const std::size_t tiles = base::pieceCount(g, TILE_DOFS);
    // The first read DOF of each tile.
    std::vector<std::size_t> firstReadOf(tiles + 1, readDofs.size()); // bounded by the g set's tiles
    for (std::size_t t = 0, r = 0; t < tiles; ++t)
    {
        while (r < readDofs.size() && static_cast<std::size_t>(readDofs[r]) < t * TILE_DOFS)
            ++r;
        firstReadOf[t] = r;
    }
    std::vector<std::vector<double>> blocks(pieces); // one per piece, a tile × its columns
    for (std::size_t p = 0; p < pieces; ++p)
        blocks[p].resize(TILE_DOFS * (std::min(columns, (p + 1) * pieceColumns) - p * pieceColumns));
    // PERFORMANCE: the pieces walk the tiles CHUNK_TILES at a time together, so that the values the fill reads for a
    // chunk (the shared route's class vectors there: a few MB) come from the memory once and from the L3 for the other
    // pieces, instead of once per piece.
    // Measured: the gradients of 50 cases of mixed fidelity, on a model at 1/16 of its full size.
    for (std::size_t chunk = 0; chunk < tiles; chunk += CHUNK_TILES)
        base::parallelFor(
            m_pool, pieces, 1,
            [&](std::size_t firstPiece, std::size_t endPiece)
            {
                for (std::size_t p = firstPiece; p < endPiece; ++p)
                {
                    const std::size_t begin = p * pieceColumns;
                    const std::size_t width = std::min(columns, begin + pieceColumns) - begin;
                    const std::span<const std::size_t> pieceOrder(order.data() + begin, width);
                    double* const block = blocks[p].data();
                    for (std::size_t t = chunk; t < std::min(tiles, chunk + CHUNK_TILES); ++t)
                    {
                        const std::size_t firstRead = firstReadOf[t];
                        const std::size_t endRead = firstReadOf[t + 1];
                        // Every entry of a pseudo-load is at a read DOF (markReads): a tile without one has none.
                        if (endRead == firstRead)
                            continue;
                        const auto firstDof = static_cast<base::DofIndex>(t * TILE_DOFS);
                        fill(firstRead, endRead, firstDof, pieceOrder, block, width);
                        for (const Group& group : groups[p])
                        {
                            const PseudoLoads::LoadSegments& segments = m_segments.at(group.subcase);
                            const std::size_t groupWidth = group.end - group.first;
                            for (std::size_t q = segments.tileStart[t]; q < segments.tileStart[t + 1]; ++q)
                            {
                                const PseudoLoads::LoadSegment& segment = segments.segments[q];
                                const SparseVector& entries = m_loads->sourceOf(segment.source, group.subcase);
                                double* sums = accumulators + group.offset + segment.source * groupWidth;
                                for (std::uint32_t e = segment.begin; e < segment.end; ++e)
                                {
                                    const auto [dof, value] = entries[e];
                                    const double* x =
                                        block + static_cast<std::size_t>(dof - firstDof) * width + group.first;
                                    for (std::size_t k = 0; k < groupWidth; ++k)
                                        sums[k] += value * x[k];
                                }
                            }
                        }
                    }
                }
            });
    // The final sums of each variable, every column in column order (the pieces of variables never share a value).
    const std::size_t fieldCount = m_loads->fieldCount();
    const bool hasOwnLoads = sources > fieldCount;
    base::parallelFor(m_pool, n, PRODUCT_VARIABLES_PER_PIECE,
                      [&](std::size_t firstVariable, std::size_t endVariable)
                      {
                          for (std::size_t j = firstVariable; j < endVariable; ++j)
                              for (std::size_t c = 0; c < columns; ++c)
                              {
                                  const double* sums = accumulators + accumulatorOf[c];
                                  const std::size_t stride = strideOf[c];
                                  double sum = 0.0;
                                  for (const auto& [field, derivative] : m_loads->fieldsOf(j))
                                      sum += derivative * sums[field * stride];
                                  // dotRange adds the own load's product, +0 without one.
                                  sum += hasOwnLoads ? sums[(fieldCount + j) * stride] : 0.0;
                                  destinations[c][j] += sum;
                              }
                      });
}

} // namespace oos::design
