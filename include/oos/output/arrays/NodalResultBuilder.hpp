#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace oos::output
{

// Makes the nodal result tables of the data contract from g-set vectors (one value per DOF of DofMap: six per grid
// point in its CD system, one per scalar point): every point of the model is a row, in the order of the node table,
// the grid points before the scalar points, the order of the tables of DISPLACEMENT = ALL and SPCFORCES = ALL.
//
// Thread-safety: const; safe concurrently. Ownership: borrows the DOF map.
class NodalResultBuilder
{
public:
    explicit NodalResultBuilder(const assembly::DofMap& dofMap) : m_dofMap(&dofMap) {}

    // Throws std::invalid_argument when `valuesG` is not of the g-set size.
    [[nodiscard]] NodalResultTable build(std::string kind, std::int32_t subcase, std::span<const double> valuesG,
                                         base::MemoryBudget& budget) const;
    // The table of the points of a SET only (`members` sorted; every point when `isAll`), in the same order.
    [[nodiscard]] NodalResultTable build(std::string kind, std::int32_t subcase, std::span<const double> valuesG,
                                         base::MemoryBudget& budget, bool isAll,
                                         std::span<const std::int32_t> members) const;

private:
    const assembly::DofMap* m_dofMap;
};

} // namespace oos::output
