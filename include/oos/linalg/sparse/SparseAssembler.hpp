#pragma once

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"

#include <cstddef>
#include <functional>

namespace oos::linalg
{

// The sizes of the work of an assembly. Both are fixed by the caller from the work, never from the number of threads;
// neither changes the result (every term is summed in element order whatever they are), only the speed and the
// memory: a wave holds at most elementsPerPiece × piecesPerWave element matrices.
struct AssemblyOptions
{
    std::size_t elementsPerPiece = 128; // elements formed by one call of the source
    std::size_t piecesPerWave = 64;     // pieces formed in parallel before they are accumulated
};

// Adds the element matrices of a model into a BlockSparseMatrix, fast and deterministically: the
// same bits whatever the number of threads, and the same bits as a serial loop over the elements.
//
// The scheme, in waves of pieces of consecutive elements:
//   1. formation: the pieces of a wave are formed in parallel, each by one call of the source into its own
//      ContributionBlock (an element matrix is computed once, by one thread);
//   2. accumulation by ownership: the block columns of the pattern are dealt to the threads in small chunks of
//      consecutive nodes, round robin, and each thread walks every contribution of the wave **in element order**,
//      adding only the terms that fall in the columns it owns (in a contribution, row-major, as a serial loop).
// Every term of the matrix is therefore owned by exactly one thread, which adds its contributions in the order of the
// elements, the order a serial loop uses: the bits depend neither on the number of threads (which only changes who
// owns a column) nor on the piece and wave sizes. There is no list of triplets and no atomic operation.
//
// Thread-safety: assemble() may run concurrently on different targets. Ownership: holds only its options.
class SparseAssembler
{
public:
    // Writes the contributions of the elements [begin, end) into `block` (empty when called), in element order. It may
    // write any number of contributions per element (none for an element without stiffness). Called concurrently
    // for different ranges: it must not write shared state.
    using ContributionSource = std::function<void(std::size_t begin, std::size_t end, ContributionBlock& block)>;

    // Throws std::invalid_argument when a size of the options is 0.
    explicit SparseAssembler(AssemblyOptions options = {});

    // Adds the contributions of elements [0, elementCount) to `target` (which is not cleared first).
    // Throws std::invalid_argument when a contribution names a DOF outside the pattern or couples two nodes that the
    // pattern does not couple (the pattern was built from another connectivity); the exception of the source is
    // passed on; OperationCancelledError when `cancellation` is cancelled between pieces. After an exception the
    // target holds a partial sum.
    void assemble(std::size_t elementCount, const ContributionSource& source, BlockSparseMatrix& target,
                  base::ThreadPool* pool = nullptr, const base::CancellationToken* cancellation = nullptr) const;

    [[nodiscard]] const AssemblyOptions& options() const noexcept { return m_options; }

private:
    AssemblyOptions m_options;
};

} // namespace oos::linalg
