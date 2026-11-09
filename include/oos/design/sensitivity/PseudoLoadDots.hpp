#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <span>

namespace oos::design
{

// The products of many vectors with the pseudo-loads of their subcases, by tiles of the g set (the
// adjoint vectors of the `vjp` and of the adjoint and chained methods, AdjointColumns):
//     dest_c[j] += b_{j,s_c}ᵀ·λ_c   for every column c (vector λ_c, subcase s_c) and variable j,
// with the bits of PseudoLoads::dotAll(s_c, λ_c) followed by the additions to dest_c in column order.
//
// The columns are grouped by subcase and cut into pieces the threads take. A piece walks the tiles of TILE_DOFS g-set
// DOFs in increasing order, asks the caller for its columns' values at the tile's read DOFs (TileFill) and adds every
// entry of its subcases' segments in the tile (PseudoLoads::segments) to the accumulator of its source (a field, or a
// variable's own load) and column. Each accumulator is then the chain of dotSparse over that source's entries (in DOF
// order from +0), and a variable's product Σ_r (∂v_r/∂x_j)·acc_r in the order of its fields plus its own load's
// accumulator (+0 without one), the sum of dotRange. A column's values never depend on the other columns, the pieces or
// the threads. Not for a relieved subcase (its relief term reads the whole vector: PseudoLoads::markReads false).
//
// Thread-safety: addProducts() runs on the pool and is not reentrant (it keeps the segments of the subcases it has
// seen). Ownership: borrows the pseudo-loads, the read DOFs, the budget and the pool, which must outlive it.
class PseudoLoadDots
{
public:
    // The g-set DOFs of a tile.
    // PERFORMANCE: a pseudo-load is read from memory once per piece and tile instead of once per column, and a piece's
    // values at a tile's read DOFs (TILE_DOFS × its columns, some 200 kB) stay in the L2 while its entries are added.
    // Measured: a model at 1/16 of its full size, the dots of the shared route 8.3 -> 2.3 s (mixed fidelity,
    // 50 cases), of the default route 9.3 -> 4.7 s (2d, scaled).
    static constexpr std::size_t TILE_DOFS = 256;

    // Writes the values of the columns `columns` (the call's column positions, in the piece's order) at the read DOFs
    // [firstRead, endRead) into block[(readDofs[r] − firstDof)·stride + k] for the k-th of `columns`.
    using TileFill = std::function<void(std::size_t firstRead, std::size_t endRead, base::DofIndex firstDof,
                                        std::span<const std::size_t> columns, double* block, std::size_t stride)>;

    // `readDofs`: increasing, every DOF at which the subcases' pseudo-loads have an entry (PseudoLoads::markReads of
    // each subcase the products will see), on a g set of `gSize` DOFs.
    PseudoLoadDots(const PseudoLoads& loads, std::span<const base::DofIndex> readDofs, std::size_t gSize,
                   base::MemoryBudget& budget, base::ThreadPool* pool)
        : m_loads(&loads), m_readDofs(readDofs), m_gSize(gSize), m_budget(&budget), m_pool(pool)
    {
    }

    // destinations[c][j] += b_{j,s}ᵀ·λ_c for every column c of subcase columnSubcases[c] (PseudoLoads' index) and
    // variable j, in column order for each j (the class comment).
    void addProducts(std::span<const std::size_t> columnSubcases, std::span<double* const> destinations,
                     const TileFill& fill);

private:
    const PseudoLoads* m_loads;
    std::span<const base::DofIndex> m_readDofs;
    std::size_t m_gSize;
    base::MemoryBudget* m_budget;
    base::ThreadPool* m_pool;
    std::map<std::size_t, PseudoLoads::LoadSegments> m_segments; // by subcase, made at its first products
    base::BudgetedBuffer<double> m_accumulators;                 // kept from one call to the next
};

} // namespace oos::design
