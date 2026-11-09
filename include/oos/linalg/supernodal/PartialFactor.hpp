#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstdint>

namespace oos::linalg
{

// What the condensation of a unit without its factor gives (PartialFactor::condenseWithoutFactor): K* and
// the pivots D of the interior (one value an interior equation, in equation order: what FactorSolve::solveDiagonal
// divides by), the largest ratio of the MAXRATIO test, and the working storage used.
struct PartialCondensation
{
    DenseMatrix schurComplement;
    base::BudgetedBuffer<double> pivots;
    double largestPivotRatio = 0.0;
    FactorWorkspaceUsage workspace;
};

// What the factorisation without its factor reserves (PartialFactor::estimateWithoutFactor): the fixed part (K*, the
// skeleton factor and its solve plan, the pivots), the values of a run (the working storage of the factorisation's
// schedule and the panels resident at once: a unit of the solve plan in each running subtree, or one top supernode),
// the rows of the forward sweep's accumulators per column, and the largest panel (the least region of a recovery).
struct WithoutFactorEstimate
{
    std::uint64_t fixedBytes = 0;
    std::int64_t runValues = 0;
    std::int64_t accumulatorRowCount = 0;
    std::int64_t largestPanelValues = 0;

    // The bytes for `columnCount` swept columns and keptPanelValues panel values kept (a recovery's region).
    [[nodiscard]] std::uint64_t bytes(std::size_t columnCount, std::int64_t keptPanelValues = 0) const noexcept;
};

// The partial factorisation of a unit: the interior equations eliminated, the trailing set of
// the symbolic factor (the boundary of the unit, ordered last) left as the exact Schur complement
//
//   K* = K_bb − K_bi·K_ii⁻¹·K_ib,
//
// with the same elimination applied to load vectors (condensed loads f* = f_b − K_bi·K_ii⁻¹·f_i) and the
// back-substitution of the interior from given boundary values (recovery u_i = K_ii⁻¹·(f_i − K_ib·u_b)).
//
// Method: the multifrontal factorisation of SupernodalFactor stops before the trailing supernode, whose front,
// assembled from the boundary entries of the matrix and the update matrices of its children, is K* itself. With the
// interior factor K_ii = L_ii·D_i·L_iiᵀ and L_bi = K_bi·L_ii⁻ᵀ·D_i⁻¹ (the rows of the boundary in the interior panels):
// the forward sweep over the interior panels leaves f_b − L_bi·L_ii⁻¹·f_i = f* in the boundary rows, and
// u_i = L_ii⁻ᵀ·(D_i⁻¹·L_ii⁻¹·f_i − L_biᵀ·u_b) is the diagonal and backward sweeps with the boundary rows set to u_b.
// The interior is positive definite (the unit held at its boundary): it is factorised without pivoting, with the
// MAXRATIO test, so no pivot is ever delayed into the boundary.
//
// Recovery data: the interior factor stays in memory while kept and is never written to disk;
// releaseInteriorFactor() frees it and recomputeInteriorFactor() computes it again, to the same bits (the
// factorisation is deterministic), so that a recovery gives the same bits either way. FactorRetention (M5) decides.
//
// Thread-safety: const members may run at once; release and recompute need exclusive access. Ownership: owns the
// interior factor and K* (budgeted); the budget must outlive it. Movable, not copyable.
class PartialFactor
{
public:
    PartialFactor() = default;

    // Throws std::invalid_argument when options.pivoting is not PositiveDefinite, and as SupernodalFactor::factorise
    // (MemoryBudgetError before any work, K* included; SingularityError for a mechanism of the interior).
    [[nodiscard]] static PartialFactor factorise(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                 const DenseKernels& kernels, const FactorOptions& options,
                                                 base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t dimension() const noexcept { return m_dimension; }
    [[nodiscard]] std::int32_t trailingDofCount() const noexcept { return m_trailingDofCount; }
    [[nodiscard]] std::int32_t interiorDofCount() const noexcept { return m_dimension - m_trailingDofCount; }

    // K*, trailingDofCount() square, both triangles, in the order of the trailing equations.
    [[nodiscard]] const DenseMatrix& schurComplement() const noexcept { return m_schurComplement; }

    // f* = f_b − K_bi·K_ii⁻¹·f_i for each column of `loads` (dimension() rows, in the numbering of the matrix):
    // trailingDofCount() × columns, reserved in `budget`. Throws std::logic_error when the interior factor was
    // released, std::invalid_argument on a row count mismatch.
    [[nodiscard]] DenseMatrix condenseLoads(ConstDenseMatrixView loads, const DenseKernels& kernels,
                                            base::MemoryBudget& budget,
                                            const base::CancellationToken* cancellation = nullptr) const;

    // Recovery: on entry the interior rows of `loadsToSolution` hold f_i (its trailing rows are ignored), and
    // trailingValues holds u_b (trailingDofCount() × the same columns); on return every row holds u (u_b copied).
    // Throws as condenseLoads.
    void recover(DenseMatrixView loadsToSolution, ConstDenseMatrixView trailingValues, const DenseKernels& kernels,
                 const base::CancellationToken* cancellation = nullptr) const;

    [[nodiscard]] bool isInteriorFactorKept() const noexcept { return m_isInteriorFactorKept; }
    // The kernel family and the pivot policy of its factorisation (what a recomputation must match).
    [[nodiscard]] KernelKind kernel() const noexcept { return m_kernel; }
    [[nodiscard]] const PivotPolicy& pivotPolicy() const noexcept { return m_pivotPolicy; }
    // Whether its interior factor is kept and is the factorisation of `matrix` on `symbolic` with the kernel family
    // `kernel` and the pivot policy `policy`: their shapes, the checksum of the matrix's structure and values, the
    // family and the policy (a factor kept beyond its analysis is taken only for the same equations; the
    // caller compares the equations themselves). O(the matrix).
    [[nodiscard]] bool isFactorisationOf(const SymbolicFactor& symbolic, const SymmetricCsc& matrix, KernelKind kernel,
                                         const PivotPolicy& policy) const noexcept;
    // Frees the interior factor (K* stays): recovery needs recomputeInteriorFactor() first.
    void releaseInteriorFactor() noexcept;
    // Computes the interior factor again from the same symbolic factor and matrix (the caller keeps them), with the
    // same kernel family and pivot policy: the same bits as the first time. Throws std::invalid_argument when they do
    // not have the shape of the first factorisation, when the matrix values differ (a checksum of its structure and
    // values), when the kernel family or the pivot policy (MAXRATIO, BAILOUT) differ, and as factorise.
    void recomputeInteriorFactor(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                 const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget);

    // The interior factor (its inertia, its ratio test). Throws std::logic_error when it was released.
    [[nodiscard]] const SupernodalFactor& interiorFactor() const;

    // ---- Without the interior factor (interior factors not kept in memory; substructuring inside the elimination
    // tree; streamed factorisation) ----

    // The condensation of factorise() followed by condenseLoads(), without keeping the interior factor: each panel is
    // used for the forward elimination of `loads` (dimension() rows, in place) as soon as it is made, then released
    // (linalg's StreamedFactorisation and ForwardStream). On return the interior rows of `loads` hold L_ii⁻¹·f_i and
    // its trailing rows f* = f_b − K_bi·K_ii⁻¹·f_i: the bits of condenseLoads (and of FactorSolve::eliminateForward
    // over the interior), K* and the pivots the bits of factorise(). Its peak is the fronts and the panels of the units
    // being swept, not the factor. Throws as factorise.
    [[nodiscard]] static PartialCondensation condenseWithoutFactor(const SymbolicFactor& symbolic,
                                                                   const SymmetricCsc& matrix, DenseMatrixView loads,
                                                                   const DenseKernels& kernels,
                                                                   const FactorOptions& options,
                                                                   base::MemoryBudget& budget);
    // The recovery of recover() without the interior factor: the forward sweep while the interior is factorised (as
    // condenseWithoutFactor, on the columns with a nonzero interior row), the trailing rows set to u_b, the diagonal
    // from the pivots, then the backward sweep by regions of the elimination tree, top-down, each region's panels made
    // again by the factorisation of its subtree and released once swept: the panels of the top of the tree kept from
    // the forward sweep up to `regionValues` panel values, then each remaining subtree with as many of its top
    // supernodes as fit `regionValues` (at least one), its rest the same way. The bits of recover() (every panel is the
    // factor's, every step FactorSolve's). Returns the K* its forward factorisation assembles (the caller checks it
    // against its own: the factorisation is deterministic). Throws as factorise and recover.
    // An upper bound of what condenseWithoutFactor and recoverWithoutFactor reserve with a pool of threadCount threads,
    // from the symbolic factor alone (no numeric work); their loads and swept columns are the caller's.
    [[nodiscard]] static WithoutFactorEstimate estimateWithoutFactor(const SymbolicFactor& symbolic,
                                                                     std::size_t threadCount);
    [[nodiscard]] static DenseMatrix recoverWithoutFactor(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                          DenseMatrixView loadsToSolution,
                                                          ConstDenseMatrixView trailingValues,
                                                          const DenseKernels& kernels, const FactorOptions& options,
                                                          base::MemoryBudget& budget, std::int64_t regionValues);

private:
    void requireKept(const char* operation) const;

    std::int32_t m_dimension = 0;
    std::int32_t m_trailingDofCount = 0;
    std::int32_t m_interiorSupernodeCount = 0;
    bool m_isInteriorFactorKept = false;
    // What a recomputation must match to give the same bits (and the same diagnostics).
    KernelKind m_kernel = KernelKind::Reference;
    PivotPolicy m_pivotPolicy;
    std::uint64_t m_matrixChecksum = 0;
    SupernodalFactor m_interior;
    DenseMatrix m_schurComplement;
};

} // namespace oos::linalg
