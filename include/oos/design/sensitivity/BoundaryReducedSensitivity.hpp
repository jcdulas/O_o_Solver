#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/sensitivity/BoundaryReducedStatistics.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/solutions/sol101/CondensedFactorisedSet.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::design
{

// The share of the memory budget left when the sensitivities start (BoundaryReducedSensitivity's construction) that
// the boundary bases kept between blocks may hold. A basis Y_u is dense, interface DOFs × n_a doubles: 230 MB for a
// unit of 48 boundary DOFs on the interface of 600 000 DOFs of 10 000 plates (two fasteners per joint), 23 MB on the
// 60 000 DOFs of a thousand, so 23 GB for a thousand units' bases (1.84 GB were measured kept for 300 plates):
// kept without a limit they exhaust the budget long before the factors do. A quarter leaves the rest of the budget
// to the blocks (their g-set columns) and to the interior factors made again.
inline constexpr double BOUNDARY_BASIS_RETENTION_SHARE = 0.25;

// The static sensitivity solves of a design point on the condensed path, reduced by the boundaries: X ← T·K⁻¹·Tᵀ·X for
// right-hand sides X on the g set (pseudo-loads b_j of the direct method, adjoint weights c of the adjoint method), on
// a condensed constraint set the analysis kept (solutions::CondensedFactorisedSet), in three steps:
//  1. condensation: each condensed unit whose interior rows a column loads condenses those rows onto its boundary with
//     its interior factor, f* = f_a − K_ao·K_oo⁻¹·f_o (the unit's relations and spiders applied first: UnitSystem's
//     equation loads); the rows of the interface's points are its own right-hand sides;
//  2. interface: the loads of a unit on the interface (its condensed loads and the columns' rows on its boundary)
//     span at most n_a vectors (its boundary DOFs), so when more than n_a columns of a block load a unit (its interior
//     or its boundary), the interface problem is solved for the n_a unit loads on its boundary (its boundary basis
//     Y_u) and each column takes Y_u·(f* + its rows on the boundary); otherwise a column's condensed loads are added to
//     its interface right-hand side. The columns with a right-hand side left are solved: per block at most
//     Σ_u min(columns of u, n_a) + the columns loading interface rows on no condensed unit's boundary (a full unit's
//     interior, an external unit's contour; BoundaryReducedStatistics). A basis is kept for the later blocks of the
//     set while the bases kept hold at most their share of the budget (BOUNDARY_BASIS_RETENTION_SHARE), the
//     least recently used released first and made again, to the same bits, by the next block that needs it; whether a
//     block takes a unit's basis depends on its columns only, never on what is kept, so the results do not depend on
//     the budget;
//  3. recovery: the units the caller reads (an interior DOF in `isReadDof`) are recovered,
//     u_o = K_oo⁻¹·(f_o − K_oa·u_a), only for the columns that load them or move their boundary; the interior rows of
//     the units not read are left at zero, as the caller does not read them.
// A unit neither loaded by a column nor read is not touched: no interior solve, no recovery, only its K* inside the
// interface factor. The full path's solve (SensitivitySystem) gives the same X to round-off: the same equations
// eliminated in another order.
//
// A unit whose interior factor the analysis released (FactorRetention) or never made (a cache hit) factorises its
// interior again, to the bits of its condensation (StaticCondensation::recomputeInteriorFactor), the first time a
// column needs it; the factor then stays in the analysis' budget with the kept set.
//
// Determinism: the units in the plan's order, the columns in order, the interface solves and the interior sweeps
// deterministic: the same bits whatever the threads of the kernels.
//
// Thread-safety: none (the statistics, the bases and the factors made again change). Ownership: shares the kept set;
// owns the unit bases (budgeted); borrows the budget, which must outlive it.
class BoundaryReducedSensitivity
{
public:
    // `basisRetentionBytes`: the bytes the bases kept between blocks may hold (default: BOUNDARY_BASIS_RETENTION_SHARE
    // of the budget available now; 0 keeps none). Throws std::invalid_argument for a null set.
    BoundaryReducedSensitivity(std::shared_ptr<const solutions::CondensedFactorisedSet> set, linalg::KernelKind kernel,
                               base::MemoryBudget& budget, base::ThreadPool* pool = nullptr,
                               std::optional<std::uint64_t> basisRetentionBytes = std::nullopt);
    BoundaryReducedSensitivity(const BoundaryReducedSensitivity&) = delete;
    BoundaryReducedSensitivity& operator=(const BoundaryReducedSensitivity&) = delete;
    BoundaryReducedSensitivity(BoundaryReducedSensitivity&&) = delete;
    BoundaryReducedSensitivity& operator=(BoundaryReducedSensitivity&&) = delete;
    ~BoundaryReducedSensitivity();

    [[nodiscard]] const solutions::CondensedFactorisedSet& set() const noexcept { return *m_set; }

    // X ← T·K⁻¹·Tᵀ·X (columnsG: g-set rows × columns), the interior rows of the units no DOF of `isReadDof` (one flag
    // per g-set DOF) reads left at zero; an empty `isReadDof` reads every unit. Throws std::invalid_argument on a size
    // mismatch, MemoryBudgetError, SingularityError (an interior factor made again: never, it was made once).
    void solve(linalg::DenseMatrixView columnsG, std::span<const std::uint8_t> isReadDof = {});

    [[nodiscard]] const BoundaryReducedStatistics& statistics() const noexcept { return m_statistics; }
    // Whether the analysis knows the displacement of a g-set DOF: an interface point's, or an interior point's of a
    // unit it recovered (the others' are not computed).
    [[nodiscard]] bool isKnownDof(base::DofIndex dof) const noexcept
    {
        return m_isKnown[static_cast<std::size_t>(dof)] != 0;
    }
    [[nodiscard]] bool isEveryDofKnown() const noexcept { return m_isEveryDofKnown; }

private:
    // The interface displacements (local DOFs × columns) of interface loads (local DOFs × columns, modified: their
    // relations folded), the held DOFs at zero.
    void solveInterface(linalg::DenseMatrixView loadsLocal, linalg::DenseMatrixView displacementsLocal);
    // Makes sure unit u has its interior factor.
    void requireFactor(std::size_t u);
    // Unit u's boundary basis, the interface displacements of a unit load on each row of its K*: the kept one, or made
    // now (n_a interface solves).
    const linalg::DenseMatrix& boundaryBasis(std::size_t u);
    // After block m_block used unit u's basis: keeps it while the kept bases fit their share, releasing first the
    // least recently used others (ties: the lower unit index), then u's itself.
    void retainOrRelease(std::size_t u);
    void releaseBasis(std::size_t u);

    std::shared_ptr<const solutions::CondensedFactorisedSet> m_set;
    base::MemoryBudget* m_budget;
    linalg::DenseKernels m_kernels;
    base::BudgetedBuffer<std::uint8_t> m_isKnown; // per g-set DOF
    bool m_isEveryDofKnown = true;
    std::vector<std::optional<linalg::DenseMatrix>> m_bases; // per unit, while kept
    std::vector<std::uint64_t> m_basisLastBlock;             // per unit: the last block that used its basis
    std::uint64_t m_basisRetentionBytes = 0;
    std::uint64_t m_retainedBasisBytes = 0;
    std::uint64_t m_block = 0; // the blocks solved so far
    BoundaryReducedStatistics m_statistics;
};

} // namespace oos::design
