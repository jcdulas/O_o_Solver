#pragma once

#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/sensitivity/SubcaseSolve.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

// The adjoint method of a subcase (K⁻¹·Cᵀ, one solve per response item):
// for an item r with displacement weights c = ∂r/∂u_g,
//     K_ll·λ_l = (Tᵀ·cᵀ)_l,   λ_g = T·λ_l,   dr/dx_j = ∂r/∂x_j + λ_gᵀ·b_j,
// which equals the direct c·du/dx_j because K_ll is symmetric: c·T·K_ll⁻¹·(Tᵀ·b_j)_l = ((Tᵀ·cᵀ)_l)ᵀ·K_ll⁻¹·(Tᵀ·b_j)_l.
// The products λ_gᵀ·b_j are sparse dot products with the pseudo-loads of the fields (PseudoLoads::dotAll): no
// right-hand side per variable is formed. Solved by blocks of SENSITIVITY_BLOCK_COLUMNS items on the factor.
//
// The order for the equations (combineArguments): the gradients of the argument items first, then
// de/dx = ∂h/∂x + Σ_k ∂h/∂r_k·dr_k/dx.
//
// Thread-safety: solve() runs on the calling thread; solveTogether() on the pool it is given. Ownership: none kept.
class AdjointSensitivity
{
public:
    // Adds λᵀ·b to the gradients of `items` (any item with weights: a plain one, or an equation item with its chained
    // weights). Returns the solves made (one per item that reads the displacements).
    static std::int64_t solve(const SubcaseSolve& context, std::span<const ItemKey> items, LinearisedItems& linearised);

    // The items of one subcase gathered with others of the same factor (solveTogether).
    struct SubcaseItems
    {
        std::size_t subcase = 0; // PseudoLoads' index
        std::vector<ItemKey> items;
    };
    // solve() for the items of several subcases on one full-path factor: their
    // columns (subcase, item) in the given order, at most SENSITIVITY_BLOCK_COLUMNS a solve over the subcases; the
    // right-hand sides from the items' entries (AdjointColumns::reduceToEquations: the values of toEquations()) and the
    // products by tiles of the g set (AdjointColumns::addProducts: the bits of dotAll), both on `pool`. Each item's
    // gradient receives the bits solve() gives it. Under inertia relief (the products read every DOF) each subcase is
    // solved by solve(). Returns the solves made per entry of `subcases`. Precondition: `system` on the full path.
    static std::vector<std::int64_t> solveTogether(const SensitivitySystem& system, const PseudoLoads& loads,
                                                   std::span<const SubcaseItems> subcases, LinearisedItems& linearised,
                                                   base::MemoryBudget& budget, base::ThreadPool* pool);

    // The gradients of equation items (in an order where each follows the equations it reads) from those of their
    // arguments: de/dx = ∂h/∂x + Σ_k ∂h/∂r_k·dr_k/dx (no solve).
    static void combineArguments(std::span<const ItemKey> equationItems, LinearisedItems& linearised);
};

} // namespace oos::design
