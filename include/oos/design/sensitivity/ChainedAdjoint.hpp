#pragma once

#include "oos/design/sensitivity/SubcaseSolve.hpp"

#include <cstdint>
#include <span>

namespace oos::design
{

// The chained adjoint of the equation responses. An equation item e = h(r_1, ..., r_p, x)
// whose argument items are linear in u at the current state (c_k = ∂r_k/∂u) has the gradient
//     de/dx_j = ∂h/∂x_j + Σ_k ∂h/∂r_k · ∂r_k/∂x_j|explicit + λ_eᵀ·b_j,   λ_e = K⁻¹·c_eᵀ,   c_e = Σ_k (∂h/∂r_k)·c_k:
// its adjoint right-hand side is formed at once from the weights of its arguments, so one solve serves the item
// whatever the number of its arguments and of the variables. Equations of equations chain their weights through every
// level (chain() of the readers after those of the equations they read).
//
// chain() forms c_e and the explicit part of an item from its arguments' (their own linearisations, formed before);
// solve() is the adjoint method on the equation items with these chained weights.
//
// Thread-safety: runs on the calling thread. Ownership: none kept.
class ChainedAdjoint
{
public:
    // c_e = Σ_k ∂h/∂r_k·c_k and ∂e/∂x = ∂h/∂x + Σ_k ∂h/∂r_k·∂r_k/∂x for an equation item; its arguments are formed.
    static void chain(const ItemKey& key, LinearisedItems& linearised);
    // λ_eᵀ·b added to the gradients of the equation items (one solve per item that reads the displacements).
    static std::int64_t solve(const SubcaseSolve& context, std::span<const ItemKey> equationItems,
                              LinearisedItems& linearised);
};

} // namespace oos::design
