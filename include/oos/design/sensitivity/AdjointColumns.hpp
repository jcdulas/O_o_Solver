#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/sensitivity/PseudoLoadDots.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"
#include "oos/design/sensitivity/SensitivitySystem.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace oos::design
{

// The adjoint columns of a full-path factor and their products with the pseudo-loads (the routes of the
// vector-Jacobian products and the adjoint and chained methods of the gradients): what the products of the subcases
// of one system read of an adjoint vector, the right-hand side of a column made from its items' entries, and the
// products of many columns by tiles of the g set.
//
// The reads: the g-set DOFs at which the products read the adjoint vectors (PseudoLoads::markReads, the union over the
// subcases given), each with the equation or the Gm row fromEquations() takes its value from, and the equation of every
// g-set DOF with the n-set DOFs of Gm's rows, by which a right-hand side is reduced from the entries of its items
// alone. `readsEvery()`: the products read every DOF (a relieved subcase, or the adjoint vectors kept whole, asked by
// the caller); then only reduceToEquations() serves, the products being made from whole vectors.
//
// The products (addProducts): those of PseudoLoadDots on the read DOFs, the bits of PseudoLoads::dotAll followed by an
// addition per column in column order.
//
// Thread-safety: the const methods may run at once; addProducts() runs on the pool and is not reentrant
// (PseudoLoadDots). Ownership: borrows the system, the pseudo-loads, the budget and the pool, which must outlive it.
class AdjointColumns
{
public:
    static constexpr std::size_t DOT_TILE_DOFS = PseudoLoadDots::TILE_DOFS;
    static constexpr std::int32_t NONE = std::numeric_limits<std::int32_t>::min();

    // The reads of `system` for the subcases `subcases` (their indices in `loads`), every DOF when `readsEvery` or when
    // one of them reads every DOF.
    AdjointColumns(const SensitivitySystem& system, const PseudoLoads& loads, std::span<const std::size_t> subcases,
                   bool readsEvery, base::MemoryBudget& budget, base::ThreadPool* pool);

    [[nodiscard]] bool readsEvery() const noexcept { return m_readsEvery; }
    // The read DOFs, increasing (empty when readsEvery()).
    [[nodiscard]] std::span<const base::DofIndex> readDofs() const noexcept { return m_readDofs.span(); }

    // Tᵀ·v of a g-set vector `scratch` whose non-zero entries are at `touched`: reduced by Gm (reduceToN, as
    // toEquations() reduces), written to `equations` (the others +0), then `scratch` set back to zero and `touched`
    // cleared: the values of toEquations() in O(the entries + Gm) instead of O(g set).
    void reduceToEquations(std::vector<double>& scratch, std::vector<base::DofIndex>& touched,
                           std::span<double> equations) const;
    // The value fromEquations() gives at read DOF k (the same arithmetic: an l-set DOF its equation, an m-set DOF the
    // sum of its Gm row over the equations of its n-set DOFs, zero elsewhere). Precondition: !readsEvery().
    [[nodiscard]] double valueAtRead(std::span<const double> equations, std::size_t k) const;
    // The smallest equation of the DOFs of `weights` (an item's derivative), or the equation count when none has one:
    // the place of a right-hand side in the factor's elimination order.
    [[nodiscard]] std::int32_t firstEquation(const SparseVector& weights) const noexcept;
    // The positions of columns in a solve, by their first equations (increasing, stable): the columns a tile takes
    // together then share more of the forward sweep's reach (its supernodes from those equations to the root); a
    // column's values never depend on its position.
    [[nodiscard]] static std::vector<std::size_t> solveSlots(std::span<const std::int32_t> firstEquations);
    // The adjoint vector of a solved column on the g set at the read DOFs (the other DOFs of `vectorG` left as they
    // are), or fromEquations() whole when readsEvery().
    void expand(std::span<const double> equations, std::span<double> vectorG) const;

    using TileFill = PseudoLoadDots::TileFill;
    // PseudoLoadDots::addProducts on the read DOFs. Precondition: !readsEvery().
    void addProducts(std::span<const std::size_t> columnSubcases, std::span<double* const> destinations,
                     const TileFill& fill);

private:
    const SensitivitySystem* m_system;
    bool m_readsEvery = false;
    base::BudgetedBuffer<base::DofIndex> m_readDofs;    // increasing
    base::BudgetedBuffer<std::int32_t> m_readSource;    // per read DOF: its equation (≥ 0), −1 − its Gm row, or NONE
    base::BudgetedBuffer<std::int32_t> m_equationOfDof; // per g-set DOF, −1 for none
    base::BudgetedBuffer<base::DofIndex> m_gmTermDofs;  // the n-set DOFs of Gm's rows, each once
    std::unique_ptr<PseudoLoadDots> m_dots;             // when not readsEvery()
};

} // namespace oos::design
