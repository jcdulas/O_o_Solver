#pragma once

#include "oos/assembly/assemble/ElementMatrixStore.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparseAssembler.hpp"

namespace oos::assembly
{

// The stiffness matrix Kgg of the g set: the element stiffness matrices of every batch
// whose family has the stiffness facet, in the displacement systems of the grid points, added by
// linalg::SparseAssembler into a BlockSparseMatrix on the node pattern of the model (the nodes of DofMap). The order of
// the sums is the order of the batches in the snapshot and of the rows in each batch, whatever the number of threads.
//
// Thread-safety: assemble() is const and may run concurrently on different targets. Ownership: borrows the set and the
// DOF map, which must outlive it.
class StiffnessAssembler
{
public:
    StiffnessAssembler(const ElementSet& set, const DofMap& dofMap, linalg::AssemblyOptions options = {})
        : m_set(&set), m_dofMap(&dofMap), m_options(options)
    {
    }

    // Adds Kgg to `target` (not cleared first). Precondition: the set passed its check, and the pattern of `target`
    // couples the nodes of every element (ElementConnectivity). kept: the matrices the connectivity pass kept, used in
    // place of a second computation when given and kept (the same matrices in the same order: the same bits). Throws
    // as linalg::SparseAssembler.
    void assemble(linalg::BlockSparseMatrix& target, base::ThreadPool* pool = nullptr,
                  const base::CancellationToken* cancellation = nullptr,
                  const ElementMatrixStore* kept = nullptr) const;

private:
    const ElementSet* m_set;
    const DofMap* m_dofMap;
    linalg::AssemblyOptions m_options;
};

} // namespace oos::assembly
