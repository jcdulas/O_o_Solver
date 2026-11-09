#pragma once

// Internal to the assembly module: the elements of several batches numbered one after the other, so that
// linalg::SparseAssembler and the connectivity pass see one list of elements, and the conversion of the element
// matrices (grid point rows and components 1 to 6) to the node DOFs of linalg.

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace oos::assembly
{

// Computes the matrices of the rows `range` of batch `batchIndex` (an index in ElementSet::batches()) into `block`.
using ElementMatrixFacet = std::function<void(std::size_t batchIndex, const AssemblyBatch& batch,
                                              elements::ElementRange range, elements::ElementMatrixBlock& block)>;

// The elements of the batches `batchIndices` of an element set as one list: element k of the list is a row of one
// batch, the batches in the order given and the rows in order (the order of every sum of the assembly).
//
// Thread-safety: const methods may run concurrently (each call uses its own blocks). Ownership: borrows the set and
// the DOF map.
class ElementMatrixSource
{
public:
    ElementMatrixSource(const ElementSet& set, std::vector<std::size_t> batchIndices, const DofMap& dofMap,
                        ElementMatrixFacet facet);

    [[nodiscard]] std::size_t elementCount() const noexcept { return m_firstElement.back(); }

    // Computes the matrices of the elements [begin, end) of the list into `block` (cleared first), batch range by
    // batch range, calling visit(batchIndex, block) after each range (batchIndex: the index in ElementSet::batches()).
    void compute(std::size_t begin, std::size_t end, elements::ElementMatrixBlock& block,
                 const std::function<void(std::size_t, const elements::ElementMatrixBlock&)>& visit) const;

    // Writes the matrices of the elements [begin, end) as contributions on the node DOFs: the ContributionSource of
    // linalg::SparseAssembler.
    void writeContributions(std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions) const;

    // The node DOF of a DOF of an element.
    [[nodiscard]] linalg::NodeDof nodeDofOf(const elements::ElementDof& dof) const noexcept
    {
        return m_dofMap->nodeDofOf(m_dofMap->dofOf(dof));
    }

    // The node of a DOF of an element.
    [[nodiscard]] std::int32_t nodeOf(const elements::ElementDof& dof) const noexcept
    {
        return m_dofMap->nodeDofOf(m_dofMap->dofOf(dof)).node;
    }

private:
    const ElementSet* m_set;
    std::vector<std::size_t> m_batchIndices; // bounded by the number of card types
    std::vector<std::size_t> m_firstElement; // per batch of the list, and the total last
    const DofMap* m_dofMap;
    ElementMatrixFacet m_facet;
};

} // namespace oos::assembly
