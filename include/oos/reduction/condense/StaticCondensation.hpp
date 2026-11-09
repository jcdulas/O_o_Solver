#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/PartialFactor.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/reduction/condense/ReducedUnit.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/reduction/scheduling/KeptInteriorFactors.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace oos::reduction
{

// The load patterns of a unit (one column per vector of its load basis, UnitLoadBasis): the loads on its local DOFs
// (UnitSystem numbering) and, for the patterns of enforced displacements, the values of its DOFs held in every
// subcase (no column when no pattern has any).
struct UnitPatterns
{
    linalg::ConstDenseMatrixView loads;    // localDofCount() × patterns
    linalg::ConstDenseMatrixView enforced; // localDofCount() × patterns, or no column
    std::int64_t stamp = 0;                // UnitLoadBasis::patternStamp: what the patterns depend on
};

// The static condensation of one unit (Guyan condensation, exact for statics and trim), on the partial factorisation:
//     K* = K_aa − K_ao·K_oo⁻¹·K_oa,    f*_k = f_a,k − K_ao·K_oo⁻¹·f_o,k,
// on the unit's equations (UnitSystem: its constraints applied first, the spiders absorbed by change of basis). The
// load patterns are condensed by the forward sweep of the interior factor (PartialFactor::condenseLoads), so the
// interior responses g_k = K_oo⁻¹·f_o,k are never formed here: the recovery (UnitRecovery) solves them together with
// the boundary term. The unit's rigid-body mass properties and its six inertial load patterns are condensed with them.
//
// What is kept: the ReducedUnit, which is all a cache may store; the interior factor stays in memory for
// the recovery during the analysis and may be released (releaseInteriorFactor, decided by FactorRetention)
// and computed again (recomputeInteriorFactor) from the unit's equations, which this object keeps, to the same bits
// (the factorisation is deterministic, PartialFactor), so that a recovery gives the same bits either way. A release
// frees the whole partial factor, its own copy of K* with the interior panels (the reduced unit keeps K*), so that a
// released unit holds K* once; the recomputation makes a new partial factor and checks its K* against the reduced
// unit's, to the bit.
//
// Without its factor (the WithoutFactor constructor): the condensation is made by linalg's
// streamed factorisation (PartialFactor::condenseWithoutFactor), each panel used once for the forward sweep of the
// patterns and inertial patterns and released, so that no interior factor is held; the pattern work is made from the
// swept patterns and the pivots at once and kept (patternWork() gives it). K*, f*, the inertial patterns and G have the
// bits of the condensation that keeps its factor. Its recovery factorises its interior again
// (recomputeInteriorFactor), or, with enableRecoveryWithoutFactor, solves by regions of its elimination tree
// (PartialFactor::recoverWithoutFactor: the bits of the recovery from a factor, no region above the bound), its K*
// checked to the bit.
//
// A condensation taken from the condensation cache (the second constructor; CondensationCache) holds the cached
// reduced unit and no interior factor: recomputeInteriorFactor() then makes its first factorisation, and checks that
// its K* has the bits of the cached one (a cache entry that does not is refused with base::CacheError: a stale entry
// is never used silently).
//
// With a factor kept by an earlier analysis of the process (KeptInteriorFactors, taken for
// the same equations): the third constructor condenses the unit without factorising it, K* the factor's and the load
// and inertial patterns condensed by its forward sweep (condenseLoads, the computation of the first constructor: the
// bits of a fresh condensation); adoptInteriorFactor() gives a condensation from the cache, or one whose factor was
// released, the kept factor instead of recomputeInteriorFactor() (its K* checked to the bit the same way). The kept
// factor's buffers stay reserved in the budget of the analysis that made it, which the condensation keeps alive, and
// its bytes are reserved in the budget given (a loan, given back with the factor). giveInteriorFactor() hands the
// factor over at the end of the analysis, to be kept for the next one.
//
// Thread-safety: const methods may run concurrently; release, recompute, adopt and give need exclusive access.
// Ownership: shares the unit's equations and the reduced unit (with the cache, for a cached one); owns the partial
// factor (budgeted) and, for a kept one, what keeps its memory alive.
class StaticCondensation
{
public:
    // The tag of the condensation without the interior factor (the class comment).
    struct WithoutFactor
    {
    };

    // Throws std::invalid_argument when the patterns do not have the unit's local DOFs as rows, as
    // PartialFactor::factorise (SingularityError for a mechanism of the interior: the unit is not held by its
    // boundary), MemoryBudgetError, OperationCancelledError.
    StaticCondensation(std::shared_ptr<const UnitSystem> system, const UnitPatterns& patterns,
                       const linalg::DenseKernels& kernels, const linalg::FactorOptions& options,
                       base::MemoryBudget& budget);
    // The same without the interior factor (the class comment). Throws as the first constructor.
    StaticCondensation(std::shared_ptr<const UnitSystem> system, const UnitPatterns& patterns,
                       const linalg::DenseKernels& kernels, const linalg::FactorOptions& options,
                       base::MemoryBudget& budget, WithoutFactor);
    // The condensation of `system` taken from the cache: `reduced` its cached reduced unit (labels equal to the unit's
    // boundary labels: std::invalid_argument otherwise), `largestPivotRatio` the MAXRATIO diagnostic of the
    // factorisation that made it. No factorisation is made.
    StaticCondensation(std::shared_ptr<const UnitSystem> system, std::shared_ptr<const ReducedUnit> reduced,
                       double largestPivotRatio, const linalg::FactorOptions& options);
    // The condensation of `system` with the interior factor `kept` (taken from KeptInteriorFactors for the system's
    // equations): no factorisation, the bits of the first constructor. Throws std::invalid_argument when the patterns
    // do not have the unit's local DOFs as rows, base::InternalError when `kept` is not the factorisation of the
    // system's equations with these kernels and options, MemoryBudgetError when its bytes do not fit `budget` (the
    // factor is then freed), OperationCancelledError.
    StaticCondensation(std::shared_ptr<const UnitSystem> system, const UnitPatterns& patterns, KeptInteriorFactor kept,
                       const linalg::DenseKernels& kernels, const linalg::FactorOptions& options,
                       base::MemoryBudget& budget);

    [[nodiscard]] const UnitSystem& system() const noexcept { return *m_system; }
    [[nodiscard]] const ReducedUnit& reducedUnit() const noexcept { return *m_reduced; }
    // The reduced unit, shared (what the condensation cache stores).
    [[nodiscard]] const std::shared_ptr<const ReducedUnit>& sharedReducedUnit() const noexcept { return m_reduced; }
    [[nodiscard]] std::size_t patternCount() const noexcept { return m_reduced->patterns().columns(); }
    // Whether it came from the cache (no factorisation made until recomputeInteriorFactor()).
    [[nodiscard]] bool isFromCache() const noexcept { return m_isFromCache; }
    // The largest MAXRATIO ratio of its interior factorisation (the cached one's for a condensation from the cache).
    [[nodiscard]] double largestPivotRatio() const noexcept { return m_largestPivotRatio; }

    [[nodiscard]] bool isInteriorFactorKept() const noexcept { return m_factor.isInteriorFactorKept(); }
    // The bytes the partial factor holds while kept: its interior panels and their row indices, and its copy of K*.
    [[nodiscard]] std::uint64_t interiorFactorBytes() const noexcept;
    // Frees the partial factor, its interior panels and its copy of K* (the reduced unit keeps K*): a recovery needs
    // recomputeInteriorFactor() first.
    void releaseInteriorFactor() noexcept
    {
        m_factor = linalg::PartialFactor{};
        m_factorLoan.release();
        m_factorMemoryOwner.reset();
        m_isInteriorFactorReused = false;
    }
    // Whether the interior factor it holds was kept by an earlier analysis (the third constructor,
    // adoptInteriorFactor).
    [[nodiscard]] bool isInteriorFactorReused() const noexcept { return m_isInteriorFactorReused; }
    // Takes `kept` (taken from KeptInteriorFactors for the system's equations) as its interior factor, in place of
    // recomputeInteriorFactor(), with the same check of its K* against the reduced unit's to the bit (base::CacheError
    // for a condensation from the cache whose first factor it is, base::InternalError otherwise). Throws
    // std::logic_error when its factor is kept, std::invalid_argument for kernels of another family than the first
    // factorisation's, base::InternalError when `kept` is not the factorisation of the system's equations,
    // MemoryBudgetError when its bytes do not fit `budget` (the factor is then freed).
    void adoptInteriorFactor(KeptInteriorFactor kept, const linalg::DenseKernels& kernels, base::MemoryBudget& budget);
    // Hands its interior factor over (it then holds none) with a copy of its equations, to be kept for a later
    // analysis (KeptInteriorFactors::offer): `unitIdentity` the hash of the unit's element IDs, `analysisBudget` the
    // budget its factor was made in when it made it (shared: the kept factor keeps it alive). None when it holds no
    // factor or its system has no equations.
    [[nodiscard]] std::optional<KeptInteriorFactor> giveInteriorFactor(
        std::uint64_t unitIdentity, const std::shared_ptr<base::MemoryBudget>& analysisBudget);
    // Whether it was condensed without its interior factor (the WithoutFactor constructor).
    [[nodiscard]] bool isCondensedWithoutFactor() const noexcept { return m_isWithoutFactor; }
    // The recoveries (recover()) without the interior factor, by regions of at most regionValues panel values, their
    // reservations in `budget` (which outlives them), until disableRecoveryWithoutFactor(); the factor must not be kept
    // (std::logic_error). Each recovery factorises the interior again in regions (PartialFactor::recoverWithoutFactor)
    // and checks the K* it makes against the reduced unit's, to the bit (base::CacheError for a unit from the cache,
    // base::InternalError otherwise).
    void enableRecoveryWithoutFactor(std::int64_t regionValues, base::MemoryBudget& budget);
    void disableRecoveryWithoutFactor() noexcept;
    [[nodiscard]] bool isRecoveryWithoutFactor() const noexcept { return m_regionBudget != nullptr; }
    // The recoveries made without the factor so far.
    [[nodiscard]] std::int64_t regionRecoveryCount() const noexcept { return m_regionRecoveryCount; }
    // Factorises the interior again, with the kernel family of the first factorisation (another one gives other bits:
    // std::invalid_argument), and checks that the new K* has the bits of the reduced unit's. For a condensation from
    // the cache, its first factorisation: base::CacheError when its K* differs from the cached one by a bit (naming the
    // unit and what to do); otherwise a difference is a base::InternalError (the factorisation is deterministic).
    // The kernels may have another number of threads than the first factorisation's: the bits do not depend on it.
    void recomputeInteriorFactor(const linalg::DenseKernels& kernels, base::MemoryBudget& budget);
    // The same with another cancellation token than the first factorisation's (a condensation kept after its analysis
    // for the sensitivities of a design point: the analysis' token may be gone; null: none).
    void recomputeInteriorFactor(const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                 const base::CancellationToken* cancellation);

    // u_i = K_oo⁻¹·(f_i − K_oa·u_a) column by column (PartialFactor::recover): on entry the interior rows of
    // `loadsToSolution` (equationCount() rows) hold f_i, `boundaryValues` (boundaryDofCount() rows) u_a; on return
    // every row holds u. Without the factor (enableRecoveryWithoutFactor) by regions, the same bits. Throws
    // std::logic_error when the interior factor was released and no recovery without it is enabled.
    void recover(linalg::DenseMatrixView loadsToSolution, linalg::ConstDenseMatrixView boundaryValues,
                 const linalg::DenseKernels& kernels, const base::CancellationToken* cancellation = nullptr) const;

    // f* = f_b − K_bo·K_oo⁻¹·f_o = Ψ̃ᵀ·f for columns on the unit's equations (equationCount() rows): the condensation
    // of loads by the forward sweep of the interior factor (PartialFactor::condenseLoads), boundaryDofCount() rows
    // reserved in `budget`. Craig-Bampton projects its matrices onto the constraint modes Ψ̃ = [−K_oo⁻¹·K_ob; I] with
    // it (CraigBampton), and the sensitivities of a design point condense their pseudo-loads or adjoint weights onto
    // the boundary ("with condensation, reduced by the boundaries"; design::BoundaryReducedSensitivity).
    // Throws as recover().
    [[nodiscard]] linalg::DenseMatrix condenseLoads(linalg::ConstDenseMatrixView loads,
                                                    const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                                    const base::CancellationToken* cancellation = nullptr) const
    {
        return m_factor.condenseLoads(loads, kernels, budget, cancellation);
    }
    // The same f* without the interior factor (the load basis handed over in a memory mode that keeps no
    // factor): the interior factorised again by linalg's streamed factorisation (PartialFactor::
    // condenseWithoutFactor), each panel used for the forward sweep of the columns and released, so that its peak is
    // the fronts, not the factor; the bits of condenseLoads(), the K* it assembles checked against the reduced unit's
    // to the bit (base::CacheError for a unit from the cache, base::InternalError otherwise). Throws
    // std::invalid_argument for columns of another row count or kernels of another family than the condensation's, and
    // as the factorisation.
    [[nodiscard]] linalg::DenseMatrix condenseLoadsWithoutFactor(linalg::ConstDenseMatrixView loads,
                                                                 const linalg::DenseKernels& kernels,
                                                                 base::MemoryBudget& budget,
                                                                 const base::CancellationToken* cancellation) const;

    // The unit's equations replaced by another build of the same unit: the system without its
    // equations (UnitSystem's constructor from a UnitBoundaryData) when its equations are released after the
    // condensation, its equations built again (UnitSystem's first constructor, the same bits) for its recovery, after
    // which recomputeInteriorFactor checks the new factor's K* to the bit. `equationNamer` names the equations of its
    // factorisations on the new system (FactorOptions). Throws std::invalid_argument when the interior factor is kept
    // (it was made from the equations given up: release it first), or when `system` is another unit, has other boundary
    // labels (the rows of K*) or another count of interior equations.
    void rebindSystem(std::shared_ptr<const UnitSystem> system, linalg::EquationNamer equationNamer);

    // The interior factor (its inertia, its ratio test). Throws std::logic_error when it was released.
    [[nodiscard]] const linalg::SupernodalFactor& interiorFactor() const { return m_factor.interiorFactor(); }

    // The work of the patterns on their own interior responses, G = F_oᵀ·K_oo⁻¹·F_o (patterns × patterns, symmetric),
    // F_o the patterns on the interior equations: with K_oo = L·D·Lᵀ, F_oᵀ·K_oo⁻¹·F_o = (L⁻¹·F_o)ᵀ·D⁻¹·(L⁻¹·F_o), one
    // forward sweep and the diagonal (linalg::FactorSolve over the interior columns). The external work of a case is
    // then ½·(u_aᵀ·f_interface + Σ_units αᵀ·G·α) without recovering the units (the boundary terms of each unit cancel
    // between its u_oᵀ·f_o and its f*). Not part of
    // the reduced unit: an analysis quantity, never cached. Throws as recover().
    [[nodiscard]] linalg::DenseMatrix patternWork(const UnitPatterns& patterns, const linalg::DenseKernels& kernels,
                                                  base::MemoryBudget& budget) const;

private:
    void recompute(const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                   const linalg::FactorOptions& options);
    // Moves `kept`'s factor in, its bytes reserved in `budget` (MemoryBudgetError before anything changes), after
    // checking it is the factorisation of the system's equations (base::InternalError).
    void takeKeptFactor(KeptInteriorFactor kept, const linalg::DenseKernels& kernels, base::MemoryBudget& budget);
    // Whether the K* of the factor has the bits of the reduced unit's.
    [[nodiscard]] bool isSchurComplementTheReducedOne() const noexcept;

    std::shared_ptr<const UnitSystem> m_system;
    linalg::FactorOptions m_options;
    // A factor kept by an earlier analysis: the budget its buffers are reserved in (declared before the factor: freed
    // after it), and its bytes reserved in this analysis' budget while it is held.
    std::shared_ptr<base::MemoryBudget> m_factorMemoryOwner;
    base::MemoryReservation m_factorLoan;
    linalg::PartialFactor m_factor;
    bool m_isInteriorFactorReused = false;
    // Without the factor: the pattern work made at the condensation, and the recoveries by regions.
    bool m_isWithoutFactor = false;
    linalg::DenseMatrix m_patternWork;
    std::int64_t m_regionValues = 0;
    base::MemoryBudget* m_regionBudget = nullptr;
    mutable std::int64_t m_regionRecoveryCount = 0;
    std::shared_ptr<const ReducedUnit> m_reduced;
    bool m_isFromCache = false;
    bool m_hasFactorised = false;
    linalg::KernelKind m_kernel = linalg::KernelKind::Reference; // of the first factorisation
    double m_largestPivotRatio = 0.0;
};

} // namespace oos::reduction
