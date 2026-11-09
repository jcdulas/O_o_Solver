#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/KeptObject.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ProgressCounter.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/dense/Inertia.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/FactorStorage.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::linalg
{

class DenseMatrix;
class MultifrontalDriver;
class OutOfCoreFactor;
class PartialFactor;
class SolvePlan;
class StreamedFactorisation;

// How the pivots of a factorisation are chosen.
enum class PivotingMode : std::uint8_t
{
    // LDLᵀ without pivoting, in the order of the symbolic factor, with the MAXRATIO test per pivot: the stiffness
    // matrices of statics. A zero pivot stops the factorisation (a mechanism).
    PositiveDefinite,
    // LDLᵀ with Bunch-Kaufman pivoting inside each front, the pivots that cannot be taken there delayed to the parent
    // front: the shifted matrices of the eigenvalue and buckling solvers. Zero pivots are counted in the inertia.
    Indefinite
};

// The options of a factorisation. The pool is the one of the DenseKernels given: it runs the independent subtrees
// and the threads of the kernels inside large fronts.
struct FactorOptions
{
    PivotingMode pivoting = PivotingMode::PositiveDefinite;
    // The MAXRATIO and BAILOUT of the ratio test (positive definite mode only).
    PivotPolicy pivotPolicy;
    // Names the DOFs of the diagnostics; empty: equations by number.
    EquationNamer equationNamer;
    // Checked before every supernode, and inside a front between the tiles and blocks of the dense kernels; not owned,
    // may be null.
    const base::CancellationToken* cancellation = nullptr;
    // Advanced by the columns eliminated; not owned, may be null. Polled by the caller's thread.
    base::ProgressCounter* progress = nullptr;
    // The plan of the solves of an earlier factorisation of the same symbolic factor (SupernodalFactor::keptSolvePlan),
    // taken instead of building one in positive definite mode when no pivot is delayed: the plan depends on the
    // panels' rows, the tree and the cut of its subtrees only, which the symbolic factor fixes then. Its
    // bytes are reserved in the factorisation's budget while the factor lives. Null: the plan is built.
    std::shared_ptr<const base::KeptObject<SolvePlan>> keptSolvePlan;
    // Whether a plan the factorisation builds is kept in a budget of its own (SupernodalFactor::keptSolvePlan), its
    // bytes reserved in the factorisation's budget while the factor lives, for later factorisations of the same
    // symbolic factor.
    bool keepsSolvePlan = false;
    // The panels' memory of an earlier factor (SupernodalFactor::releaseStorage), taken as the new factor's panels when
    // its sizes are theirs (the same symbolic factor): its pages are mapped already, so no page faults; its
    // bytes are reserved in the budget as a new allocation's. Left as it is when its sizes differ. Not owned, may be
    // null.
    FactorStorage* storage = nullptr;
};

// The working storage of a factorisation (the fronts and the update matrices; the panels apart): the bound reserved
// up front (FactorSchedule) and the peak of the live values that reservation covered, measured buffer by buffer.
// peakValues <= boundValues is the invariant of the bound. Storage beyond the plan (fronts grown by delayed pivots, a
// subtree run while the planned number of subtrees were already running) is reserved in the budget on its own and is
// not in the peak. heldPeakValues is the peak of what the working storage held, the large blocks kept for reuse
// included (WorkspacePool): within the bound too, but for storage beyond the plan.
struct FactorWorkspaceUsage
{
    std::int64_t boundValues = 0;
    std::int64_t peakValues = 0;
    std::int64_t subtreesBeyondPlan = 0;
    std::int64_t heldPeakValues = 0;
};

// A subtree of the supernode tree, the consecutive supernodes [first, last] of a postorder, `last` its root: the
// independent subtrees of the factorisation (FactorSchedule's cut, which depends on the tree and its flops only, never
// on the threads), which the solves also run in parallel (FactorSolve).
struct SupernodeRange
{
    std::int32_t first = 0;
    std::int32_t last = 0;
};

// One supernode of a factor: its panel holds the eliminated columns of L (unit diagonal implied) with D on the
// diagonal of its first eliminatedCount rows (zero above it), over all its rows; rows[k] is the equation of row k (the
// eliminated columns first, in pivot order, then the rows below). subdiagonal holds the e_k of the 2 × 2 pivots (null
// in positive definite mode).
struct FactorPanel
{
    ConstDenseMatrixView values; // rows.size() × eliminatedCount
    std::span<const std::int32_t> rows;
    const double* subdiagonal = nullptr;

    [[nodiscard]] std::size_t eliminatedCount() const noexcept { return values.columns; }
    [[nodiscard]] BlockDiagonalView pivots() const noexcept
    {
        return {values.data, values.leadingDimension + 1, subdiagonal, values.columns};
    }
};

// The numeric LDLᵀ factorisation of a sparse symmetric matrix by the multifrontal method (the one
// supernodal driver): P·A·Pᵀ = L·D·Lᵀ with the order of the symbolic factor, P adding
// only the interchanges of delayed and Bunch-Kaufman pivots in indefinite mode.
//
// Method (Duff & Reid 1983; Liu, "The multifrontal method for sparse matrix solution", SIAM Review 34, 1992): the
// supernodes are processed children first; the front of supernode s, dense, of the order of its columns plus its rows,
// receives the entries of the matrix in its columns and the update matrices of its children (extend-add, children in
// increasing order); its columns are eliminated by the dense kernels (DenseKernels::factorFrontWithoutPivoting or the
// candidate form of factorWithPivoting), leaving in the trailing block the update matrix S = F_rr − L_r·D·L_rᵀ that
// its parent will receive. A positive definite front is assembled and eliminated in place: its columns in the panel
// of s, its trailing block in the buffer of its update matrix, which waits for the parent; an indefinite front (its
// pivots may be interchanged and delayed) is one matrix, whose panel and update matrix are copied out.
//
// Determinism: every entry of a front is summed in a fixed order (the matrix entry, then the children
// in increasing order), and each front is computed by the dense kernels, whose bits do not depend on the threads. The
// independent subtrees run in parallel and the top fronts use the kernels' threads (FactorSchedule), but neither
// changes what a front computes: the factor has the same bits for any number of threads, for a given kernel family.
//
// Memory: the panels (SymbolicFactor::panelValueCount values exactly when no pivot is delayed), the row
// indices and the bound of the working storage of FactorSchedule are reserved in the budget before any numeric work;
// a factorisation that does not fit is refused with MemoryBudgetError before it starts. Delayed pivots grow the fronts
// and panels concerned; the growth is reserved when it happens. The large blocks of the working storage released
// during the factorisation are kept for the next fronts within that bound (no fresh pages per front).
//
// The equations are those of the SymmetricCsc given, which must be in the order of the symbolic factor (a DofSelection
// built with SymbolicFactor::nodeOrder()); the diagnostics name them through FactorOptions::equationNamer.
//
// Thread-safety: immutable once built; FactorSolve reads it from any number of threads. Ownership: owns the panels
// (budgeted); the budget must outlive the factor. Movable, not copyable.
class SupernodalFactor
{
public:
    SupernodalFactor() = default;

    // The complete factorisation of `matrix` (the trailing supernode, if any, is factorised as the root it is).
    // Throws std::invalid_argument when the matrix is not of the dimension of the symbolic factor or has an entry
    // outside its structure; MemoryBudgetError before any work when it does not fit; SingularityError for a zero pivot
    // in positive definite mode (a mechanism, with its DOFs), or when the ratio test fails and the policy stops
    // (BAILOUT >= 0); OperationCancelledError when cancelled.
    [[nodiscard]] static SupernodalFactor factorise(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                    const DenseKernels& kernels, const FactorOptions& options,
                                                    base::MemoryBudget& budget);

    // An upper bound of the bytes factorise reserves for `symbolic` with a pool of threadCount threads in the mode
    // `pivoting`, before any growth from delayed pivots: the panels and their row indices (exact), the working storage
    // (the bound of FactorSchedule for that mode: a positive definite front holds no front buffer), the per-supernode
    // and per-equation arrays. The symbolic factor and the matrix are the caller's.
    [[nodiscard]] static std::uint64_t estimateMemoryBytes(const SymbolicFactor& symbolic, std::size_t threadCount,
                                                           PivotingMode pivoting);

    [[nodiscard]] std::int32_t dimension() const noexcept { return m_dimension; }
    // The columns eliminated: dimension() for a complete factor, the interior of a partial one.
    [[nodiscard]] std::int32_t eliminatedCount() const noexcept { return m_eliminatedCount; }
    [[nodiscard]] bool isComplete() const noexcept { return m_eliminatedCount == m_dimension; }
    [[nodiscard]] std::int32_t supernodeCount() const noexcept { return m_supernodeCount; }
    [[nodiscard]] PivotingMode pivoting() const noexcept { return m_pivoting; }
    [[nodiscard]] KernelKind kernel() const noexcept { return m_kernel; }

    // The signs of the pivots (Sylvester: the inertia of the matrix for a complete factor; the Sturm count of a
    // shifted matrix is inertia().negative).
    [[nodiscard]] const Inertia& inertia() const noexcept { return m_inertia; }
    // True when a pivot is zero (indefinite mode): the matrix is singular and cannot be solved with.
    [[nodiscard]] bool isSingular() const noexcept { return m_inertia.zero > 0; }
    // Pivots delayed from a front to its parent (each delay of a column counted once per front it leaves).
    [[nodiscard]] std::int64_t delayedPivotCount() const noexcept { return m_delayedPivotCount; }
    [[nodiscard]] std::int64_t twoByTwoPivotCount() const noexcept { return m_twoByTwoPivotCount; }
    // The ratio test of the positive definite mode (empty in indefinite mode).
    [[nodiscard]] const PivotDiagnostics& pivotDiagnostics() const noexcept { return m_diagnostics; }

    // The panel of a supernode in [0, supernodeCount()).
    [[nodiscard]] FactorPanel panel(std::int32_t supernode) const noexcept;
    // The values allocated for the panels: SymbolicFactor::panelValueCount() when no pivot was delayed.
    [[nodiscard]] std::int64_t allocatedPanelValueCount() const noexcept { return m_allocatedPanelValueCount; }
    // The largest panel row count (the workspace of a solve).
    [[nodiscard]] std::int32_t largestPanelRowCount() const noexcept { return m_largestPanelRowCount; }
    // The independent subtrees of the factorisation, in increasing order; the other supernodes are their ancestors.
    [[nodiscard]] std::span<const SupernodeRange> subtrees() const noexcept { return m_subtrees; }
    // The working storage of the factorisation, bound and measured peak (for the reports and the tests).
    [[nodiscard]] const FactorWorkspaceUsage& workspaceUsage() const noexcept { return m_workspaceUsage; }
    // The plan of the solves (internal to linalg/supernodal: FactorSolve), built with the factor; an empty plan for a
    // default factor.
    [[nodiscard]] const SolvePlan& solvePlan() const noexcept;
    // The memory of the panels, for a later factorisation of the same symbolic factor (FactorOptions::storage), their
    // reservations given back: the factor is then a default (empty) factor. The storage of a factor grown by delayed
    // pivots holds its symbolic panels only.
    [[nodiscard]] FactorStorage releaseStorage() noexcept;
    // The plan of the solves kept beyond the factor (FactorOptions::keptSolvePlan taken, or built with
    // keepsSolvePlan), for later factorisations of the same symbolic factor; null otherwise.
    [[nodiscard]] const std::shared_ptr<const base::KeptObject<SolvePlan>>& keptSolvePlan() const noexcept
    {
        return m_keptSolvePlan;
    }

private:
    friend class MultifrontalDriver;
    friend class OutOfCoreFactor;
    friend class PartialFactor;
    friend class StreamedFactorisation;

    // A panel whose shape differs from the symbolic one because pivots were delayed into its front.
    struct GrownPanel
    {
        base::BudgetedBuffer<double> values;
        std::vector<std::int32_t> rows; // bounded by a front
        std::vector<double> subdiagonal;
        std::int32_t eliminatedCount = 0;
    };

    // Factorises the supernodes of `symbolic` but its trailing one, whose front, assembled, is written into
    // `schurComplement` (allocated by the caller, trailingDofCount() square) when excludesTrailing.
    [[nodiscard]] static SupernodalFactor factoriseSupernodes(const SymbolicFactor& symbolic,
                                                              const SymmetricCsc& matrix, const DenseKernels& kernels,
                                                              const FactorOptions& options, bool excludesTrailing,
                                                              DenseMatrix* schurComplement, base::MemoryBudget& budget);

    std::int32_t m_dimension = 0;
    std::int32_t m_eliminatedCount = 0;
    std::int32_t m_supernodeCount = 0;
    PivotingMode m_pivoting = PivotingMode::PositiveDefinite;
    KernelKind m_kernel = KernelKind::Reference;
    Inertia m_inertia;
    std::int64_t m_delayedPivotCount = 0;
    std::int64_t m_twoByTwoPivotCount = 0;
    PivotDiagnostics m_diagnostics;
    std::int64_t m_allocatedPanelValueCount = 0;
    std::int32_t m_largestPanelRowCount = 0;
    std::vector<SupernodeRange> m_subtrees; // bounded by FactorSchedule's cut (SUBTREE_SPLIT_COUNT per level)
    FactorWorkspaceUsage m_workspaceUsage;

    // The panels in the symbolic shape: supernode s at m_panelOffset[s], c·(c + r) values, its c + r rows at
    // m_rowOffset[s], its subdiagonal (indefinite mode) at firstColumn(s).
    base::BudgetedBuffer<std::int64_t> m_panelOffset; // supernodeCount + 1
    base::BudgetedBuffer<std::int64_t> m_rowOffset;   // supernodeCount + 1
    base::BudgetedBuffer<std::int32_t> m_firstColumn; // supernodeCount
    base::BudgetedBuffer<std::int32_t> m_eliminated;  // per supernode
    base::BudgetedBuffer<std::int32_t> m_rowCount;    // per supernode: rows of its panel
    base::BudgetedBuffer<double> m_panelValues;
    base::BudgetedBuffer<std::int32_t> m_panelRows;
    // A streamed factor (StreamedFactorisation) holds no panel array: the panel of a supernode is a block
    // its factorisation lends while the supernode is resident (null otherwise), read through panel() as the others.
    // A factor out of core (OutOfCoreFactor) the same, its grown panels' values too.
    base::BudgetedBuffer<double*> m_panelPointer;
    base::BudgetedBuffer<double> m_subdiagonal; // indefinite mode: dimension values
    // Indefinite mode: one entry per supernode, null unless delayed pivots changed its shape. The table itself is
    // reserved in m_grownTableReservation (declared first: reserved before, given back after).
    base::MemoryReservation m_grownTableReservation;
    std::vector<std::unique_ptr<GrownPanel>> m_grownPanels;
    // Built last, from the final panels (shared: the factor stays movable with the plan's type incomplete here).
    std::shared_ptr<const SolvePlan> m_solvePlan;
    std::shared_ptr<const base::KeptObject<SolvePlan>> m_keptSolvePlan;
};

} // namespace oos::linalg
