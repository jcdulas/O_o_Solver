#pragma once

// The passes over the elements of the paths that assemble no matrix (internal to solutions): the full path streamed
// (sol101/StreamedFront) and the pencils from the elements (common/ElementPencil). An element's DOFs carried onto the
// equations (StreamedElements::reduce), its terms added to a front at their positions, and the element set walked in
// pieces on the pool, each piece's results merged in piece order (the same sums whatever the threads).

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::solutions::passes
{

// The g-set DOFs of an element's DOFs.
inline void gDofsOf(const assembly::DofMap& dofMap, std::span<const elements::ElementDof> dofs,
                    std::vector<base::DofIndex>& out)
{
    out.clear();
    for (const elements::ElementDof& dof : dofs)
        out.push_back(dofMap.dofOf(dof));
}

// The positions in the front of a reduction's equations (each must be there: the placement's invariant).
inline void positionsOf(const linalg::FrontTerms& front, const StreamedReduction& reduction,
                        std::vector<std::int32_t>& positions)
{
    positions.clear();
    for (const std::int32_t equation : reduction.equations)
    {
        const std::int32_t position = front.positionOf(equation);
        if (position < 0)
            throw base::InternalError(base::concatenate(
                {"the full path streamed: equation ", std::to_string(equation), " of an item placed at supernode ",
                 std::to_string(front.supernode), " is not in its front (a bug of O_o_Solver: report it)"}));
        positions.push_back(position);
    }
}

// A piece of the element set: a run of consecutive rows of one batch.
struct ElementPiece
{
    std::size_t batch = 0;
    elements::ElementRange range;
};

// The pieces of `rows` rows of the batches whose family has one of the facets.
inline std::vector<ElementPiece> elementPieces(const AnalysisModel& model, std::span<const elements::Facet> facets,
                                               std::int64_t rows)
{
    std::vector<ElementPiece> pieces; // bounded by the elements over `rows`
    const auto batches = model.elementSet().batches();
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const elements::FacetSet has = batches[b].family->facets();
        if (std::none_of(facets.begin(), facets.end(), [&](elements::Facet facet) { return has.has(facet); }))
            continue;
        for (std::int64_t first = 0; first < batches[b].view.rowCount(); first += rows)
            pieces.push_back({b, {first, std::min(first + rows, batches[b].view.rowCount())}});
    }
    return pieces;
}

// The values of an element's DOFs, u_e = T·u, per column (n × columns, column-major), from the equations' values.
inline void elementValues(const StreamedReduction& reduction, std::size_t n, linalg::ConstDenseMatrixView solution,
                          std::vector<double>& values)
{
    const std::size_t columns = solution.columns;
    values.assign(n * columns, 0.0);
    for (const StreamedReduction::Term& term : reduction.terms)
    {
        const auto equation = static_cast<std::size_t>(reduction.equations[static_cast<std::size_t>(term.equation)]);
        for (std::size_t b = 0; b < columns; ++b)
            values[static_cast<std::size_t>(term.dof) + b * n] += term.coefficient * solution(equation, b);
    }
}

// The pieces of one wave: four a thread.
inline std::size_t waveOf(const base::ThreadPool* pool) noexcept
{
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    return std::max<std::size_t>(4 * threads, 1);
}

// Runs `work(piece, slot)` over the pieces in waves on the pool (each piece's results its own, in slot `slot` of the
// wave), then `merge(piece, slot)` for each piece of the wave in order: the merged sums in piece order whatever the
// threads.
template <typename Work, typename Merge>
void inWaves(std::size_t pieceCount, base::ThreadPool* pool, const base::CancellationToken* cancellation, Work&& work,
             Merge&& merge)
{
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    const std::size_t wave = waveOf(pool);
    for (std::size_t first = 0; first < pieceCount; first += wave)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const std::size_t count = std::min(wave, pieceCount - first);
        if (pool != nullptr && threads > 1)
            pool->runBatch(count, [&](std::size_t i) { work(first + i, i); });
        else
            for (std::size_t i = 0; i < count; ++i)
                work(first + i, i);
        for (std::size_t i = 0; i < count; ++i)
            merge(first + i, i);
    }
}

} // namespace oos::solutions::passes
