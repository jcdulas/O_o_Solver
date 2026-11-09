#pragma once

#include "oos/design/sensitivity/SubcaseSolve.hpp"

#include <cstdint>
#include <span>

namespace oos::design
{

// The direct method of a subcase (K⁻¹·B, one solve per variable):
//     K_ll·du_l/dx_j = (Tᵀ·b_j)_l,   du_g/dx_j = T·du_l/dx_j,   dr/dx_j = ∂r/∂x_j + c·du_g/dx_j,
// for every variable whose pseudo-load b_j is non-zero in the subcase, by blocks of SENSITIVITY_BLOCK_COLUMNS
// right-hand sides on the factor of the system. Every item given gets its gradient from the same du/dx: a plain item
// from its weights, an equation item from its chained weights c_e (which equals combining its arguments).
//
// Thread-safety: runs on the calling thread (the solves use the system's pool). Ownership: none kept.
class DirectSensitivity
{
public:
    // Adds c·du/dx to the gradients of `items` (which start at their explicit parts). Returns the solves made.
    static std::int64_t solve(const SubcaseSolve& context, std::span<const ItemKey> items, LinearisedItems& linearised);
};

} // namespace oos::design
