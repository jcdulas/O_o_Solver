#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace oos::design::detail
{

// The element blocks of the Hessian of the Lagrangian of a design point and a set of weights y (internal to
// design/external, VectorJacobianProducts): with
//   ℒ(u, p) = Σᵢ yᵢ·rᵢ(u, p) + Σ_s λ̃_sᵀ·(P_s(p) − K(p)·u_s),
// λ̃_s the adjoint of Σ yᵢ·∂rᵢ/∂u of subcase s less the weighted SPC force rows (an SPC force item r = wᵀ·(K·u − P)
// enters ℒ as −y·w beside λ), each element e and subcase s give, over the element's DOFs and its designed field p
// (one field per diagonal block, plus separate symmetric blocks for pairs of fields),
//   A_uu = Σᵢ yᵢ·∂²rᵢ/∂u²,                         (its items not affine in u; empty otherwise)
//   A_up = Σᵢ yᵢ·∂²rᵢ/∂u∂p − (∂K_e/∂p)·λ̃_e,
//   A_pp = Σᵢ yᵢ·∂²rᵢ/∂p² + λ̃_eᵀ·(∂²P_e/∂p² − (∂²K_e/∂p²)·u_e),
// the second derivatives of ℒ restricted to the element, which do not depend on the direction of a product. A
// Hessian-vector product along v then needs, besides the solves of δu_s = K⁻¹·b_v and δλ_s, only
//   loads_s = Σ_e A_uu·δu_e + A_up·δp_e (the right-hand side of K·δλ_s),  terms_p = Σ_e A_upᵀ·δu_e + A_pp·δp_e,
// with δp = ∂p/∂x·v (the relations linear: the designed fields' second derivatives in x are zero), and
//   (H·v)_j = Σ_p ∂p/∂x_j·terms_p + Σ_s δλ_sᵀ·b_js.
//
// PERFORMANCE: the element second derivatives formed once per (x, y) and kept; a product then costs its two solves per
// subcase and one pass over the blocks (A_uu dense over the element's DOFs, 24 × 24 for a shell).
// Measured: a product after the first 0.16 s -> 1.5 ms
// on the 444-shell caisson (one thread), 0.44 s -> 5 ms on the 1 248-shell one, 0.49 s -> 6 ms on the 7 152-bar truss.
//
// Determinism: apply() sums the blocks in their order, one thread: a product has the same bits whenever it is made.
//
// Thread-safety: immutable once built; apply() is const. Ownership: owns its blocks, reserved in the budget given.
class HessianBlocks
{
public:
    static constexpr std::size_t NO_FIELD = std::numeric_limits<std::size_t>::max();

    struct Block
    {
        std::size_t subcase = 0;          // its position among the subcases of the linearisation
        std::size_t field = NO_FIELD;     // its designed quantity (a relation), or NO_FIELD
        std::vector<base::DofIndex> dofs; // the element's DOFs on the g set
        std::vector<double> displacement; // A_uu, dofs × dofs row-major, or empty (zero)
        std::vector<double> mixed;        // A_up, one per DOF, or empty (no field)
        double fieldTerm = 0.0;           // A_pp
        // A pair block has no displacement or mixed terms; fieldTerm is A_pq = A_qp.
        std::size_t otherField = NO_FIELD;
    };

    // Throws MemoryBudgetError when the blocks do not fit the budget.
    HessianBlocks(std::vector<Block> blocks, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t size() const noexcept { return m_blocks.size(); }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return m_reservation.bytes(); }
    // The blocks, in the order apply() sums them (DesignBlocks reads them).
    [[nodiscard]] std::span<const Block> blocks() const noexcept { return m_blocks; }

    // Adds to `loads[s]` (a g-set vector per subcase, sized by the caller, or empty when no block of the subcase is
    // expected) Σ_e A_uu·δu_e + A_up·δp_e, and to `terms[q]` Σ_e A_upᵀ·δu_e + A_pp·δp_q, for the moves `moves[s]` (the
    // g-set δu of each subcase; empty: it does not move) and `fieldMoves[q]` (δp of each designed quantity).
    void apply(std::span<const std::vector<double>> moves, std::span<const double> fieldMoves,
               std::vector<std::vector<double>>& loads, std::vector<double>& terms) const;

private:
    std::vector<Block> m_blocks;
    base::MemoryReservation m_reservation;
};

} // namespace oos::design::detail
