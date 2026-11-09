#include "oos/linalg/sparse/SparseAssembler.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::linalg
{
namespace
{

void requireValidDofs(const ContributionBlock& block, const SparsePattern& pattern, std::size_t firstElement)
{
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        for (const NodeDof& dof : block.dofs(i))
        {
            const bool isNodeValid = dof.node >= 0 && dof.node < pattern.nodeCount();
            if (!isNodeValid || dof.component < 0 || dof.component >= pattern.nodeDofCount(dof.node))
                throw std::invalid_argument(base::concatenate(
                    {"SparseAssembler: contribution ", std::to_string(i), " of the piece starting at element ",
                     std::to_string(firstElement), " names node ", std::to_string(dof.node), " component ",
                     std::to_string(dof.component), ", outside the pattern (", std::to_string(pattern.nodeCount()),
                     " nodes); the element DOFs must be mapped to the nodes of the pattern"}));
        }
    }
}

// Block columns are dealt to the owners in chunks of COLUMNS_PER_CHUNK consecutive nodes, round robin: owner of column
// j = (j / COLUMNS_PER_CHUNK) mod ownerCount. Consecutive elements touch nearby nodes (the numbering of a mesh is
// local), so contiguous ranges would give a whole wave to one owner; small chunks spread every wave over all of them.
// PERFORMANCE: round-robin chunks instead of one contiguous range per owner; a wave of a plate numbered row by row
// otherwise falls into one owner's range and runs serially.
// Measured: plate of 250 000 elements, 6 threads: assembly 0.30 s with contiguous ranges, 0.17 s
// with chunks of 64 columns.
constexpr std::int32_t COLUMNS_PER_CHUNK = 64;

[[nodiscard]] std::size_t ownerOf(std::int32_t column, std::size_t ownerCount) noexcept
{
    return static_cast<std::size_t>(column / COLUMNS_PER_CHUNK) % ownerCount;
}

// The scratch of one owner: the distinct nodes of a contribution and its DOFs grouped by node.
struct OwnerScratch
{
    std::vector<std::int32_t> nodes;
    std::vector<std::size_t> localNodeOfDof;
    std::vector<std::size_t> nodeDofStart; // the DOFs of local node p are dofsByNode[nodeDofStart[p], ...[p + 1])
    std::vector<std::size_t> dofsByNode;   // DOF indices of the contribution, grouped by node, increasing in a group
};

// Adds the terms of one contribution whose column node belongs to `owner`.
//
// Every term of the matrix receives the contribution's terms (r, c) in row-major order, as a serial loop over the
// element matrix would add them (it matters only when an element repeats a DOF): the DOFs are grouped by node keeping
// their order, and for each pair of nodes the rows run outside, the columns inside.
// PERFORMANCE: grouping the DOFs by node does the block lookup once per pair of nodes instead of a test per term.
// Measured: plate of 250 000 elements: 0.40 s -> 0.33 s on one thread, 0.17 s -> 0.14 s on six.
void accumulateContribution(std::span<const NodeDof> dofs, std::span<const double> matrix, std::size_t owner,
                            std::size_t ownerCount, const SparsePattern& pattern, double* values, OwnerScratch& scratch)
{
    const bool isOwned = std::any_of(dofs.begin(), dofs.end(),
                                     [&](const NodeDof& dof) { return ownerOf(dof.node, ownerCount) == owner; });
    if (!isOwned)
        return; // every term of the contribution has its column node in another owner's chunks
    const std::size_t n = dofs.size();
    scratch.nodes.clear();
    scratch.localNodeOfDof.resize(n);
    for (std::size_t r = 0; r < n; ++r)
    {
        const auto found = std::find(scratch.nodes.begin(), scratch.nodes.end(), dofs[r].node);
        scratch.localNodeOfDof[r] = static_cast<std::size_t>(found - scratch.nodes.begin());
        if (found == scratch.nodes.end())
            scratch.nodes.push_back(dofs[r].node);
    }
    const std::size_t k = scratch.nodes.size();
    // A stable counting sort of the DOFs by local node.
    scratch.nodeDofStart.assign(k + 1, 0);
    for (std::size_t r = 0; r < n; ++r)
        ++scratch.nodeDofStart[scratch.localNodeOfDof[r] + 1];
    for (std::size_t p = 0; p < k; ++p)
        scratch.nodeDofStart[p + 1] += scratch.nodeDofStart[p];
    scratch.dofsByNode.resize(n);
    for (std::size_t r = 0; r < n; ++r)
        scratch.dofsByNode[scratch.nodeDofStart[scratch.localNodeOfDof[r]]++] = r;
    for (std::size_t p = k; p > 0; --p)
        scratch.nodeDofStart[p] = scratch.nodeDofStart[p - 1];
    scratch.nodeDofStart[0] = 0;

    for (std::size_t q = 0; q < k; ++q)
    {
        const std::int32_t column = scratch.nodes[q];
        if (ownerOf(column, ownerCount) != owner)
            continue;
        for (std::size_t p = 0; p < k; ++p)
        {
            const std::int32_t row = scratch.nodes[p];
            if (row < column)
                continue; // the transposed terms, in the lower triangle, carry it
            const std::int64_t block = pattern.findBlock(row, column);
            if (block < 0)
                throw std::invalid_argument(
                    base::concatenate({"SparseAssembler: a contribution couples nodes ", std::to_string(row), " and ",
                                       std::to_string(column),
                                       ", which the pattern does not couple; build the pattern from the same "
                                       "connectivity and extra couplings as the elements"}));
            double* const blockValues = values + pattern.blockValueOffset(block);
            const std::size_t rowDofs = static_cast<std::size_t>(pattern.nodeDofCount(row));
            for (std::size_t i = scratch.nodeDofStart[p]; i < scratch.nodeDofStart[p + 1]; ++i)
            {
                const std::size_t r = scratch.dofsByNode[i];
                const double* const matrixRow = matrix.data() + r * n;
                const auto rowComponent = static_cast<std::size_t>(dofs[r].component);
                for (std::size_t j = scratch.nodeDofStart[q]; j < scratch.nodeDofStart[q + 1]; ++j)
                {
                    const std::size_t c = scratch.dofsByNode[j];
                    blockValues[static_cast<std::size_t>(dofs[c].component) * rowDofs + rowComponent] += matrixRow[c];
                }
            }
        }
    }
}

} // namespace

SparseAssembler::SparseAssembler(AssemblyOptions options) : m_options(options)
{
    if (options.elementsPerPiece == 0 || options.piecesPerWave == 0)
        throw std::invalid_argument(
            base::concatenate({"SparseAssembler: expected piece and wave sizes of at least 1, found ",
                               std::to_string(options.elementsPerPiece), " elements per piece and ",
                               std::to_string(options.piecesPerWave), " pieces per wave"}));
}

void SparseAssembler::assemble(std::size_t elementCount, const ContributionSource& source, BlockSparseMatrix& target,
                               base::ThreadPool* pool, const base::CancellationToken* cancellation) const
{
    const SparsePattern& pattern = target.pattern();
    // The owners split the terms, never the sum of one term: their number may follow the pool.
    const std::size_t ownerCount = pool == nullptr ? 1 : pool->threadCount();
    std::vector<OwnerScratch> scratch(ownerCount);
    std::vector<ContributionBlock> blocks(m_options.piecesPerWave);
    double* const values = target.values().data();

    const std::size_t pieces = base::pieceCount(elementCount, m_options.elementsPerPiece);
    for (std::size_t waveStart = 0; waveStart < pieces; waveStart += m_options.piecesPerWave)
    {
        const std::size_t waveCount = std::min(m_options.piecesPerWave, pieces - waveStart);
        base::parallelFor(
            pool, waveCount, 1,
            [&](std::size_t firstSlot, std::size_t endSlot)
            {
                for (std::size_t slot = firstSlot; slot < endSlot; ++slot)
                {
                    const std::size_t begin = (waveStart + slot) * m_options.elementsPerPiece;
                    const std::size_t end = std::min(begin + m_options.elementsPerPiece, elementCount);
                    blocks[slot].clear();
                    source(begin, end, blocks[slot]);
                    requireValidDofs(blocks[slot], pattern, begin);
                }
            },
            cancellation);
        base::parallelFor(pool, ownerCount, 1,
                          [&](std::size_t firstOwner, std::size_t endOwner)
                          {
                              for (std::size_t owner = firstOwner; owner < endOwner; ++owner)
                              {
                                  for (std::size_t slot = 0; slot < waveCount; ++slot)
                                  {
                                      const ContributionBlock& block = blocks[slot];
                                      for (std::size_t i = 0; i < block.size(); ++i)
                                          accumulateContribution(block.dofs(i), block.matrix(i), owner, ownerCount,
                                                                 pattern, values, scratch[owner]);
                                  }
                              }
                          });
    }
}

} // namespace oos::linalg
