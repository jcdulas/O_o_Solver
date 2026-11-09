#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

namespace oos::assembly
{

// The 3 × 3 diagonal blocks of every grid point (translations, rotations) and the diagonal term of every scalar point
// of an assembled n-set stiffness Knn, as GridPointSingularities reads them (NodeDiagonalBlocks): taken from the
// diagonal node blocks of the BlockSparseMatrix, which are stored complete.
//
// Thread-safety: immutable. Ownership: owns its values (budgeted).
class NodeDiagonalBlockSet
{
public:
    // Precondition: `knn` is on the node pattern of `dofMap` (6 DOFs per grid point, then 1 per scalar point;
    // std::invalid_argument otherwise).
    NodeDiagonalBlockSet(const linalg::BlockSparseMatrix& knn, const DofMap& dofMap, base::MemoryBudget& budget);
    // The blocks summed by the caller (element by element, no Knn assembled), 18 per grid point (the
    // translations' 3 × 3 block then the rotations', row-major) and one per scalar point. Throws std::invalid_argument
    // when the sizes are not those of `dofMap`.
    NodeDiagonalBlockSet(base::BudgetedBuffer<double> gridBlocks, base::BudgetedBuffer<double> scalarDiagonals,
                         const DofMap& dofMap);

    [[nodiscard]] NodeDiagonalBlocks blocks() const noexcept { return {m_gridBlocks.span(), m_scalarDiagonals.span()}; }

private:
    base::BudgetedBuffer<double> m_gridBlocks;      // 18 per grid point
    base::BudgetedBuffer<double> m_scalarDiagonals; // 1 per scalar point
};

} // namespace oos::assembly
