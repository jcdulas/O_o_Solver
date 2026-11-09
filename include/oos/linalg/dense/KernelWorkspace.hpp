#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace oos::linalg
{

// A scratch buffer: `size` uninitialised doubles (its user writes each value before reading it).
struct ScratchBuffer
{
    std::unique_ptr<double[]> values;
    std::size_t size = 0;
};

// The scratch buffers of the dense kernels kept from one call to the next (the packed panels of the tiles and of the
// products on shared panels), for a caller that makes many calls: the multifrontal factorisation. A
// buffer allocated and freed for each call came back from the system as fresh pages (the Windows heap hands the blocks
// above 512 kB to the system), each faulted in by its first write: a front of a few hundred columns paid more for its
// packing buffers' pages than for their packing.
//
// PERFORMANCE: a buffer is lent from the kept ones (the smallest that holds the request), else allocated in a size
// class (a power of two of at least SMALLEST_CLASS_VALUES values, so that the requests of the calls share buffers),
// and kept when given back, at most KEPT_BUFFER_COUNT of them (the smallest go first). The buffers are bounded by the
// tiles (DenseKernelOptions::tileSize), the blocks of terms and the groups of columns of the shared panels, never by
// a model: at most a few MB each, a few dozen MB in all with the threads of a pool; outside the memory budget, as the
// buffers each call allocated before.
// Measured (oos_supernodal_benchmark --factorisations 3, one thread): the page faults of a later factorisation
// of the 3D grid of 100 000 DOFs 364 000 → 285 000, of the shell of 100 000 DOFs 64 000 → 51 000, its time 0.42 →
// 0.39 s.
//
// Thread-safety: acquire and release are thread-safe (a mutex). Ownership: owns the buffers it keeps; a buffer lent is
// its borrower's until given back.
class KernelWorkspace
{
public:
    static constexpr std::size_t KEPT_BUFFER_COUNT = 32;
    static constexpr std::size_t SMALLEST_CLASS_VALUES = std::size_t{1} << 12;

    KernelWorkspace() = default;
    KernelWorkspace(const KernelWorkspace&) = delete;
    KernelWorkspace& operator=(const KernelWorkspace&) = delete;
    KernelWorkspace(KernelWorkspace&&) = delete;
    KernelWorkspace& operator=(KernelWorkspace&&) = delete;
    ~KernelWorkspace() = default;

    // A buffer of at least `values` doubles: a kept one, or a new one of its size class. Throws std::bad_alloc.
    [[nodiscard]] ScratchBuffer acquire(std::size_t values);
    // Takes a buffer back (kept, or freed beyond KEPT_BUFFER_COUNT).
    void release(ScratchBuffer buffer) noexcept;
    // The bytes of the buffers kept now (not lent).
    [[nodiscard]] std::uint64_t keptBytes() const;

private:
    mutable std::mutex m_mutex;
    std::vector<ScratchBuffer> m_kept; // guarded by m_mutex; at most KEPT_BUFFER_COUNT
};

} // namespace oos::linalg
