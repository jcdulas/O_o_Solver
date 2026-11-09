#include "oos/assembly/assemble/StiffnessAssembler.hpp"

#include "ElementMatrixSource.hpp"

namespace oos::assembly
{

void StiffnessAssembler::assemble(linalg::BlockSparseMatrix& target, base::ThreadPool* pool,
                                  const base::CancellationToken* cancellation, const ElementMatrixStore* kept) const
{
    if (kept != nullptr && kept->isKept())
    {
        std::size_t elementCount = 0;
        for (const std::size_t b : m_set->batchesWith(elements::Facet::Stiffness))
            elementCount += static_cast<std::size_t>(m_set->batches()[b].view.rowCount());
        linalg::SparseAssembler(m_options).assemble(
            elementCount, [kept](std::size_t begin, std::size_t end, linalg::ContributionBlock& block)
            { kept->writeContributions(begin, end, block); }, target, pool, cancellation);
        return;
    }
    const ElementMatrixSource source(
        *m_set, m_set->batchesWith(elements::Facet::Stiffness), *m_dofMap,
        [](std::size_t, const AssemblyBatch& batch, elements::ElementRange range, elements::ElementMatrixBlock& block)
        { batch.family->computeStiffness(batch.view, range, block); });
    linalg::SparseAssembler(m_options).assemble(
        source.elementCount(), [&source](std::size_t begin, std::size_t end, linalg::ContributionBlock& block)
        { source.writeContributions(begin, end, block); }, target, pool, cancellation);
}

} // namespace oos::assembly
