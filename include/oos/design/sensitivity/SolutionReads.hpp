#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/design/sensitivity/SensitivitySystem.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::design
{

// The g-set DOFs at which a block of sensitivity solutions is read: the condensed path recovers
// only the units that hold one of them (SensitivitySystem::solve, BoundaryReducedSensitivity). The direct method reads
// du_j at the weights c of its items (c·du_j); the adjoint method reads λ_i at the pseudo-loads b_j (λ_iᵀ·b_j), so that
// the adjoint vectors are recovered in the driven units only. On the full path every solution is whole: the reads are
// empty (no flag).
class SolutionReads
{
public:
    // One flag per g-set DOF of `system`: the DOFs the weights of `items` read (empty on the full path).
    [[nodiscard]] static base::BudgetedBuffer<std::uint8_t> ofWeights(const SensitivitySystem& system,
                                                                      const LinearisedItems& linearised,
                                                                      std::span<const ItemKey> items,
                                                                      base::MemoryBudget& budget);
    // The DOFs the pseudo-loads of subcase `subcase` (PseudoLoads' index) are at (empty on the full path).
    [[nodiscard]] static base::BudgetedBuffer<std::uint8_t> ofPseudoLoads(const SensitivitySystem& system,
                                                                          const PseudoLoads& loads, std::size_t subcase,
                                                                          base::MemoryBudget& budget);
};

} // namespace oos::design
