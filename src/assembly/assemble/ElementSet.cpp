#include "oos/assembly/assemble/ElementSet.hpp"

#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>

namespace oos::assembly
{

ElementSet::ElementSet(const elements::SnapshotIndex& index, const elements::ElementRegistry& registry)
    : m_index(&index)
{
    for (const input::ElementBatch& batch : index.snapshot().elementBatches())
    {
        if (const elements::ElementFamily* family = registry.find(batch.card()))
            m_batches.push_back({family, elements::ElementBatchView(batch, index)});
    }
}

void ElementSet::check(base::DiagnosticSink& sink, base::ThreadPool* pool) const
{
    for (const AssemblyBatch& batch : m_batches)
    {
        const std::int64_t rows = batch.view.rowCount();
        if (pool == nullptr || !batch.family->isCheckedByRow() || rows < 2 * CHECK_RANGE_ROWS)
        {
            batch.family->check(batch.view, sink);
            continue;
        }
        // PERFORMANCE: the rows of a long batch checked by ranges on the pool (a ShellQuad4 resolves its geometry,
        // section and frames: some 1.5 us an element, 0.25 s for the 167 000 shells of the 1 M DOF plate on one
        // thread); the sink sorts what they report, so the report is that of one thread. Measured on a 1 M DOF plate.
        const auto ranges = static_cast<std::size_t>((rows + CHECK_RANGE_ROWS - 1) / CHECK_RANGE_ROWS);
        base::parallelFor(pool, ranges, 1,
                          [&](std::size_t first, std::size_t end)
                          {
                              for (std::size_t r = first; r < end; ++r)
                              {
                                  const auto begin = static_cast<std::int64_t>(r) * CHECK_RANGE_ROWS;
                                  batch.family->checkRows(batch.view, {begin, std::min(rows, begin + CHECK_RANGE_ROWS)},
                                                          sink);
                              }
                          });
    }
}

std::vector<std::size_t> ElementSet::batchesWith(elements::Facet facet) const
{
    std::vector<std::size_t> found;
    for (std::size_t b = 0; b < m_batches.size(); ++b)
    {
        if (m_batches[b].family->facets().has(facet))
            found.push_back(b);
    }
    return found;
}

} // namespace oos::assembly
