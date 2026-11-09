#pragma once

#include "KernelFamily.hpp"

#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/KernelWorkspace.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace oos::linalg
{

// A rectangle of a result matrix computed by one task.
struct Tile
{
    std::size_t firstRow = 0;
    std::size_t rowCount = 0;
    std::size_t firstColumn = 0;
    std::size_t columnCount = 0;
};

// The tiles of an m × n result in squares of tileSize (the last ones narrower); with isLowerOnly, only the tiles that
// meet the lower triangle (a tile on the diagonal is kept whole: its kernel skips the upper elements).
[[nodiscard]] std::vector<Tile> makeTiles(std::size_t rowCount, std::size_t columnCount, std::size_t tileSize,
                                          bool isLowerOnly);

// Uninitialised buffers of at least a fixed number of doubles for the tasks of one operation (the packed panels of
// its tiles or pieces), from the kernels' workspace when the options name one (kept between calls), else allocated by
// the operation and kept between its pieces: a buffer carries no data from one piece to another.
//
// Thread-safety: acquire and release are thread-safe (a mutex, or the workspace's). Ownership: owns the buffers it
// keeps; gives them back to the workspace when it is destroyed.
class ScratchBuffers
{
public:
    ScratchBuffers(std::size_t values, KernelWorkspace* workspace) noexcept : m_values(values), m_workspace(workspace)
    {
    }
    ScratchBuffers(const ScratchBuffers&) = delete;
    ScratchBuffers& operator=(const ScratchBuffers&) = delete;
    ScratchBuffers(ScratchBuffers&&) = delete;
    ScratchBuffers& operator=(ScratchBuffers&&) = delete;
    ~ScratchBuffers();

    [[nodiscard]] ScratchBuffer acquire();
    void release(ScratchBuffer buffer);

private:
    std::size_t m_values;
    KernelWorkspace* m_workspace;
    std::mutex m_mutex;
    std::vector<ScratchBuffer> m_free;
};

// Runs the pieces of a result on the pool of the options, when the work is above the threading threshold, and hands
// each task the packing buffers of the SIMD kernels.
//
// The pieces are fixed by the caller from the work alone (never from the number of threads), and each
// piece writes only its own part of the result, so a result has the same bits whatever the thread count. The buffers
// are scratch: they are taken from a free list for the duration of a piece and never carry data from one piece to
// another, so which buffer a task gets does not matter.
//
// Thread-safety: one runner per call; its free list is guarded by a mutex.
class TileRunner
{
public:
    // innerCount: the length of the sums of the operation; largestExtent: the largest row or column count of a tile
    // (at most the tile size). Together they bound the packed panels, so that a small operation allocates little.
    TileRunner(const DenseKernelOptions& options, const KernelFamily& family, std::size_t innerCount,
               std::size_t largestExtent);

    // Calls body(piece, workspace) for every piece in [0, pieceCount), in parallel when multiplyAdds reaches the
    // threading threshold of the options.
    void run(std::size_t pieceCount, std::uint64_t multiplyAdds,
             const std::function<void(std::size_t, const PackingWorkspace&)>& body);

private:
    const DenseKernelOptions& m_options;
    bool m_needsBuffers;
    std::size_t m_panelValues;
    ScratchBuffers m_buffers; // packed A then packed W, m_panelValues each
};

} // namespace oos::linalg
