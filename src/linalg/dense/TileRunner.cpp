#include "TileRunner.hpp"

#include "oos/base/threading/ParallelFor.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <algorithm>
#include <utility>

namespace oos::linalg
{

std::vector<Tile> makeTiles(std::size_t rowCount, std::size_t columnCount, std::size_t tileSize, bool isLowerOnly)
{
    std::vector<Tile> tiles;
    for (std::size_t firstColumn = 0; firstColumn < columnCount; firstColumn += tileSize)
    {
        const std::size_t columns = std::min(tileSize, columnCount - firstColumn);
        // A tile meets the lower triangle when its last row is at or below its first column.
        const std::size_t firstRowOfColumn = isLowerOnly ? firstColumn / tileSize * tileSize : 0;
        for (std::size_t firstRow = firstRowOfColumn; firstRow < rowCount; firstRow += tileSize)
            tiles.push_back({firstRow, std::min(tileSize, rowCount - firstRow), firstColumn, columns});
    }
    return tiles;
}

TileRunner::TileRunner(const DenseKernelOptions& options, const KernelFamily& family, std::size_t innerCount,
                       std::size_t largestExtent)
    : m_options(options),
      m_needsBuffers(family.isBlocked),
      m_panelValues((std::min(options.tileSize, largestExtent) + KERNEL_TILE_GRANULE - 1) / KERNEL_TILE_GRANULE *
                    KERNEL_TILE_GRANULE * std::min(innerCount, KERNEL_INNER_BLOCK)),
      m_buffers(2 * m_panelValues, options.workspace)
{
}

void TileRunner::run(std::size_t pieceCount, std::uint64_t multiplyAdds,
                     const std::function<void(std::size_t, const PackingWorkspace&)>& body)
{
    base::ThreadPool* pool = multiplyAdds >= m_options.parallelMinimumMultiplyAdds ? m_options.pool : nullptr;
    // With a cancellation token, the pieces not started are skipped once it is cancelled, then parallelFor throws.
    base::parallelFor(
        pool, pieceCount, 1,
        [&](std::size_t begin, std::size_t end)
        {
            if (!m_needsBuffers || m_panelValues == 0)
            {
                for (std::size_t piece = begin; piece < end; ++piece)
                    body(piece, PackingWorkspace{});
                return;
            }
            ScratchBuffer buffer = m_buffers.acquire();
            const PackingWorkspace workspace{buffer.values.get(), buffer.values.get() + m_panelValues};
            for (std::size_t piece = begin; piece < end; ++piece)
                body(piece, workspace);
            m_buffers.release(std::move(buffer));
        },
        m_options.cancellation);
}

ScratchBuffers::~ScratchBuffers()
{
    if (m_workspace == nullptr)
        return;
    for (ScratchBuffer& buffer : m_free)
        m_workspace->release(std::move(buffer));
}

ScratchBuffer ScratchBuffers::acquire()
{
    {
        const std::scoped_lock lock(m_mutex);
        if (!m_free.empty())
        {
            ScratchBuffer buffer = std::move(m_free.back());
            m_free.pop_back();
            return buffer;
        }
    }
    if (m_workspace != nullptr)
        return m_workspace->acquire(m_values);
    return {std::make_unique_for_overwrite<double[]>(m_values), m_values};
}

void ScratchBuffers::release(ScratchBuffer buffer)
{
    const std::scoped_lock lock(m_mutex);
    m_free.push_back(std::move(buffer));
}

} // namespace oos::linalg
