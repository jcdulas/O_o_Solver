#include "oos/assembly/assemble/DifferentialStiffnessAssembler.hpp"

#include "ElementMatrixSource.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::assembly
{

void DifferentialStiffnessAssembler::check(base::DiagnosticSink& sink) const
{
    for (const std::size_t b : m_set->batchesWith(elements::Facet::Stiffness))
    {
        const AssemblyBatch& batch = m_set->batches()[b];
        batch.family->checkDifferentialStiffness(batch.view, sink);
    }
}

void DifferentialStiffnessAssembler::assemble(std::span<const double> displacementsG,
                                              const elements::ElementTemperatures* temperatures,
                                              linalg::BlockSparseMatrix& target, base::ThreadPool* pool,
                                              const base::CancellationToken* cancellation) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (displacementsG.size() != gSize)
        throw std::invalid_argument(
            base::concatenate({"DifferentialStiffnessAssembler: expected ", std::to_string(gSize),
                               " preload displacements (the g set), "
                               "found ",
                               std::to_string(displacementsG.size())}));
    // The g-set order of DofMap is that of NodalDisplacements: six values per grid point row, then the scalar points.
    const std::size_t gridValues = 6 * static_cast<std::size_t>(m_dofMap->gridCount());
    const elements::NodalDisplacements preload{displacementsG.first(gridValues), displacementsG.subspan(gridValues),
                                               temperatures};
    const ElementMatrixSource source(
        *m_set, m_set->batchesWith(elements::Facet::Stiffness), *m_dofMap,
        [&preload](std::size_t, const AssemblyBatch& batch, elements::ElementRange range,
                   elements::ElementMatrixBlock& block)
        { batch.family->computeDifferentialStiffness(batch.view, range, preload, block); });
    linalg::SparseAssembler(m_options).assemble(
        source.elementCount(), [&source](std::size_t begin, std::size_t end, linalg::ContributionBlock& block)
        { source.writeContributions(begin, end, block); }, target, pool, cancellation);
}

} // namespace oos::assembly
