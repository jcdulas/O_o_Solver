#include "ElementMatrixSource.hpp"

#include <algorithm>
#include <utility>

namespace oos::assembly
{

ElementMatrixSource::ElementMatrixSource(const ElementSet& set, std::vector<std::size_t> batchIndices,
                                         const DofMap& dofMap, ElementMatrixFacet facet)
    : m_set(&set), m_batchIndices(std::move(batchIndices)), m_dofMap(&dofMap), m_facet(std::move(facet))
{
    m_firstElement.push_back(0);
    for (const std::size_t b : m_batchIndices)
        m_firstElement.push_back(m_firstElement.back() + static_cast<std::size_t>(m_set->batches()[b].view.rowCount()));
}

void ElementMatrixSource::compute(
    std::size_t begin, std::size_t end, elements::ElementMatrixBlock& block,
    const std::function<void(std::size_t, const elements::ElementMatrixBlock&)>& visit) const
{
    // The batch holding element `begin`: the last whose first element is at most begin.
    auto position = std::upper_bound(m_firstElement.begin(), m_firstElement.end(), begin);
    std::size_t k = static_cast<std::size_t>(position - m_firstElement.begin()) - 1;
    std::size_t element = begin;
    while (element < end && k < m_batchIndices.size())
    {
        const std::size_t last = std::min(end, m_firstElement[k + 1]);
        if (last > element)
        {
            const AssemblyBatch& batch = m_set->batches()[m_batchIndices[k]];
            block.clear();
            m_facet(m_batchIndices[k], batch,
                    {static_cast<std::int64_t>(element - m_firstElement[k]),
                     static_cast<std::int64_t>(last - m_firstElement[k])},
                    block);
            visit(m_batchIndices[k], block);
            element = last;
        }
        ++k;
    }
}

void ElementMatrixSource::writeContributions(std::size_t begin, std::size_t end,
                                             linalg::ContributionBlock& contributions) const
{
    elements::ElementMatrixBlock block;
    std::vector<linalg::NodeDof> nodeDofs; // bounded by an element
    compute(begin, end, block,
            [&](std::size_t, const elements::ElementMatrixBlock& matrices)
            {
                for (std::size_t i = 0; i < matrices.size(); ++i)
                {
                    const std::span<const elements::ElementDof> dofs = matrices.dofs(i);
                    nodeDofs.clear();
                    for (const elements::ElementDof& dof : dofs)
                        nodeDofs.push_back(m_dofMap->nodeDofOf(m_dofMap->dofOf(dof)));
                    contributions.append(nodeDofs, matrices.matrix(i));
                }
            });
}

} // namespace oos::assembly
