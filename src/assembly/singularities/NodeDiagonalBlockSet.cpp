#include "oos/assembly/singularities/NodeDiagonalBlockSet.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace oos::assembly
{

NodeDiagonalBlockSet::NodeDiagonalBlockSet(const linalg::BlockSparseMatrix& knn, const DofMap& dofMap,
                                           base::MemoryBudget& budget)
    : m_gridBlocks(budget, 18 * static_cast<std::size_t>(dofMap.gridCount()), "node diagonal blocks"),
      m_scalarDiagonals(budget, static_cast<std::size_t>(dofMap.scalarCount()), "node diagonal blocks")
{
    const linalg::SparsePattern& pattern = knn.pattern();
    if (pattern.nodeCount() != dofMap.nodeCount())
        throw std::invalid_argument(
            "NodeDiagonalBlockSet: the matrix is not on the node pattern of the DOF map; assemble it on that pattern");
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
    {
        // The diagonal block is the first of its block column, 6 × 6 column-major, complete.
        const std::span<const double> block = knn.blockValues(pattern.columnBegin(row));
        if (block.size() != 36)
            throw std::invalid_argument(
                "NodeDiagonalBlockSet: a grid point node without 6 DOFs (a bug of O_o_Solver: report it)");
        double* out = m_gridBlocks.data() + 18 * static_cast<std::size_t>(row);
        for (std::size_t half = 0; half < 2; ++half)
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    out[9 * half + 3 * i + j] = block[(3 * half + i) + 6 * (3 * half + j)];
    }
    for (std::int32_t s = 0; s < dofMap.scalarCount(); ++s)
        m_scalarDiagonals[static_cast<std::size_t>(s)] =
            knn.blockValues(pattern.columnBegin(dofMap.gridCount() + s))[0];
}

NodeDiagonalBlockSet::NodeDiagonalBlockSet(base::BudgetedBuffer<double> gridBlocks,
                                           base::BudgetedBuffer<double> scalarDiagonals, const DofMap& dofMap)
    : m_gridBlocks(std::move(gridBlocks)), m_scalarDiagonals(std::move(scalarDiagonals))
{
    if (m_gridBlocks.size() != 18 * static_cast<std::size_t>(dofMap.gridCount()) ||
        m_scalarDiagonals.size() != static_cast<std::size_t>(dofMap.scalarCount()))
        throw std::invalid_argument(
            "NodeDiagonalBlockSet: expected 18 values per grid point and one per scalar point of the DOF map");
}

} // namespace oos::assembly
