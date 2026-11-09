#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/design/sensitivity/SensitivitySystem.hpp"

#include <cstddef>
#include <span>

namespace oos::design
{

// The right-hand sides solved at once by the sensitivity orders: a block bounds the memory of the solves
// (equations × SENSITIVITY_BLOCK_COLUMNS doubles) and is a multiple of the columns of FactorSolve's tiles
// (FactorSolve::RHS_TILE_COLUMNS, 128: two tiles, which FactorSolve solves at once).
inline constexpr std::size_t SENSITIVITY_BLOCK_COLUMNS = 256;

// What the sensitivity orders of one subcase work on: the factorised system of its constraint set, the pseudo-loads
// (the subcase's index there) and the budget of their working vectors.
struct SubcaseSolve
{
    const SensitivitySystem* system = nullptr;
    const PseudoLoads* loads = nullptr;
    std::size_t subcase = 0;
    base::MemoryBudget* budget = nullptr;
};

} // namespace oos::design
