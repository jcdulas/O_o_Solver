#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/PartialFactor.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/supernodal/UpdateLayout.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace oos::linalg
{

// The backward sweep by regions of a solve without the factor for one region bound (StreamedSolve, and the recovery of
// PartialFactor::recoverWithoutFactor): the panels the forward run keeps (the top of the tree), the subtrees factorised
// again for the regions below it and their flops, and the most panel values kept at once. Measured by a solve, or
// planned from the symbolic factor alone (StreamedSolve::regionCost): the same figures.
//
// With a retention budget (a solve of a front source), the regions retained are kept
// from the forward run instead of being factorised again: regionRunCount and recomputedFlops count the others, the
// retained figures those kept.
// A model of the upward pass on more threads, from the tasks a solve measured: the
// subtrees of the cut (each one task: its wall time) scheduled the most work first on p threads (list scheduling, the
// order the solve gives them), then the top part's flow between max(work / p, critical path) and work / p + critical
// path (base::TaskFlow). The tasks' times are those of the run's threads (sharing the memory bandwidth with them): a
// machine of more cores with no more bandwidth a core is the model's assumption.
struct TaskModel
{
    static constexpr std::size_t MODEL_THREADS[] = {1, 2, 4, 6, 12, 24};

    std::size_t measuredThreads = 0;
    double subtreePhaseSeconds = 0.0; // measured wall time of the subtrees' phase
    double topSeconds = 0.0;          // measured wall time of the top part
    double subtreeWorkSeconds = 0.0;  // the subtrees' times summed
    double largestSubtreeSeconds = 0.0;
    double topWorkSeconds = 0.0;
    double topCriticalPathSeconds = 0.0;
    std::vector<double>
        lowerSeconds; // per MODEL_THREADS: the upward pass's subtrees and top, the top at its lower bound
    std::vector<double> upperSeconds; // the top at its upper bound
};

struct RegionSweepCost
{
    std::int64_t regionValues = 0;     // the bound (panel values, in WorkspacePool's size classes)
    std::int64_t topValues = 0;        // the panels of the top kept from the forward run
    std::int64_t keptValues = 0;       // the most panel values kept at once (the top, or one region run)
    std::int64_t regionRunCount = 0;   // the subtrees factorised again
    double recomputedFlops = 0.0;      // their factorisation's flops (FactorSchedule::supernodeFlops)
    std::int64_t retainedValues = 0;   // the values of the regions retained (held from the forward run)
    std::int64_t retainedRunCount = 0; // the regions retained (not factorised again)
    double retainedFlops = 0.0;        // the flops their runs would have taken
    // The work of the runs (a front source's solve): their factorisation's flops and the forward sweep of
    // their right-hand sides (k·(c² + 2·r·c) a supernode), what the retention ranks and the plan minimises.
    double recomputedWork = 0.0;
    double retainedWork = 0.0;
    // Of an estimate (the plan's time): the work of the upward pass (one factorisation and its sweeps), of
    // it the work made in one postorder on the calling thread (the order Postorder, a cut of fewer than two subtrees),
    // and of the runs' work the part made so (a region inside one subtree of the cut); the rest on the threads.
    double upwardWork = 0.0;
    double serialUpwardWork = 0.0;
    double serialRunWork = 0.0;
    // The wall time of a solve's parts (zero in an estimate): the upward pass (one factorisation with the
    // forward sweeps), the region runs (factorised again), the downward pass with the sink's work beside each region.
    double upwardSeconds = 0.0;
    double regionSeconds = 0.0;
    double downwardSeconds = 0.0;
    // The upward pass's tasks and their model on more threads (a solve's).
    TaskModel taskModel{};
};

// What a solve without the factor reserves, from the symbolic factor alone (StreamedSolve::estimate): the streamed
// factorisation's figures (PartialFactor::estimateWithoutFactor, no trailing set: the skeleton, the pivots, the
// working storage and the resident panels of a run, the accumulators of the forward sweep), every panel of the factor
// it does not keep, and the flops of one factorisation.
struct StreamedSolveEstimate
{
    WithoutFactorEstimate run;
    std::int64_t panelValues = 0; // every panel, in WorkspacePool's size classes
    double flopCount = 0.0;
    std::int32_t supernodeCount = 0;

    // The bytes of a solve of `columnCount` right-hand sides by regions of at most regionValues panel values (at least
    // the largest panel: a region holds one; above panelValues, every panel kept from the forward run).
    [[nodiscard]] std::uint64_t bytes(std::size_t columnCount, std::int64_t regionValues) const noexcept;
};

// What a solve without the factor did.
struct StreamedSolveResult
{
    PivotDiagnostics diagnostics; // the ratio test of its factorisation (SupernodalFactor's)
    FactorWorkspaceUsage workspace;
    RegionSweepCost regions;
};

// The front of one supernode as a front source fills it (StreamedSolve::solve on a FrontSource): its
// positions are its columns [0, columnCount) (equations firstColumn + k) then its rows R(s), ascending; `panel` holds
// the columns over every position (order × columnCount), `contribution` the block of the rows (rowCount × rowCount,
// lower triangle, stored as UpdateLayout{rowCount, contributionBlockColumns} says), `rightHandColumns` and
// `rightHandRows` the right-hand sides of the columns and of the rows (× the columns of the solve). The caller adds to
// them; they hold the children's terms afterwards. In an indefinite factorisation's front (the factor out
// of core of a front source, MultifrontalDriver) the `delayedCount` columns its children delayed lie between its
// columns and its rows in the panel's rows: the positions stay those above (a source's terms never reach a delayed
// equation), at() skips them.
struct FrontTerms
{
    std::int32_t supernode = 0;
    std::int32_t firstColumn = 0;
    std::int32_t columnCount = 0;
    std::int32_t delayedCount = 0;
    std::span<const std::int32_t> rows;
    DenseMatrixView panel;
    DenseMatrixView contribution;
    std::int32_t contributionBlockColumns = 0;
    DenseMatrixView rightHandColumns;
    DenseMatrixView rightHandRows;

    [[nodiscard]] std::int32_t order() const noexcept { return columnCount + static_cast<std::int32_t>(rows.size()); }
    // The position of an equation in the front, or −1 when the front does not hold it.
    [[nodiscard]] std::int32_t positionOf(std::int32_t equation) const noexcept;
    // The term (i, j) of the front, for positions i and j in either order (the lower triangle is stored).
    [[nodiscard]] double& at(std::int32_t i, std::int32_t j) noexcept
    {
        if (i < j)
            std::swap(i, j);
        if (j < columnCount)
            return panel(static_cast<std::size_t>(i < columnCount ? i : i + delayedCount), static_cast<std::size_t>(j));
        const UpdateLayout layout{static_cast<std::int64_t>(rows.size()), contributionBlockColumns};
        const std::int64_t column = j - columnCount;
        const std::int64_t offset = layout.isBlocked()
                                        ? layout.columnOffset(column)
                                        : column * static_cast<std::int64_t>(contribution.leadingDimension);
        return contribution.data[offset + (i - columnCount)];
    }
    // The right-hand side of position i in column `column`.
    [[nodiscard]] double& rightHandSide(std::int32_t i, std::size_t column) noexcept
    {
        return i < columnCount ? rightHandColumns(static_cast<std::size_t>(i), column)
                               : rightHandRows(static_cast<std::size_t>(i - columnCount), column);
    }
};

// The matrix and the right-hand sides of a solve without the factor given front by front (the whole model streamed,
// no matrix assembled): when the front of a supernode is assembled, the caller adds its terms
// (each term of the matrix and of the right-hand sides once over all the fronts: a term (i, j) in the front of a
// supernode that holds both i and j, eliminated there or carried up to an ancestor by the update matrices). Its terms
// come before the children's update matrices in each sum, the order a matrix's entries take (assembleFront).
//
// Thread-safety: addTerms() is called once for each supernode by every factorisation that makes it (the regions
// factorise again), concurrently for distinct supernodes (the subtrees run in parallel): it must give the same terms
// in the same order every time and be safe concurrently.
class FrontSource
{
public:
    virtual ~FrontSource() = default;
    // Adds the terms of the front of supernode s (FrontTerms; cleared before).
    virtual void addTerms(FrontTerms& front) = 0;
    // The diagonal of the matrix (every term on it summed) on the columns of supernode s, for the MAXRATIO ratio of the
    // pivot diagnostics (a_jj / d_j): `values` holds columnCount(s) values.
    virtual void diagonal(std::int32_t supernode, std::span<double> values) const = 0;
};

// What takes the solution of a solve of a front source, supernode by supernode, top-down: when the
// solution of the columns of s is known, with that of its rows R(s) (every ancestor's columns it reads, done before
// it), take() receives both (columnCount × columns and rowCount × columns, in the order of FrontTerms). On the calling
// thread, each supernode once.
class SolutionSink
{
public:
    virtual ~SolutionSink() = default;
    virtual void take(std::int32_t supernode, ConstDenseMatrixView columnValues, ConstDenseMatrixView rowValues) = 0;
};

// How the factorisations of a solve of a front source run: the subtrees of FactorSchedule's cut at once
// on the threads (each serially, the top after them: the fastest, every finished subtree's update held until the top
// takes it), or one postorder of the whole tree on the calling thread with the kernels' threads (the least working
// storage: the stack of the update matrices of Liu's multifrontal order, "The multifrontal method for sparse matrix
// solution", SIAM Review 34, 1992).
enum class FrontStreamOrder : std::uint8_t
{
    Parallel,
    Postorder
};

// What a solve of a front source reserves for a region bound and a number of right-hand sides, from the
// symbolic factor alone (StreamedSolve::estimate on a source): the working storage of the fronts and update matrices
// with their right-hand sides (the schedule's subtrees at once, its top serial), the panels and right-hand sides of a
// run resident at once, those kept (the top, a region), the regions retained, the solutions of the
// supernodes the downward pass still reads, the per-supernode arrays. bytes() is the reservation of solve(): an upper
// bound of its peak.
struct FrontStreamEstimate
{
    std::int64_t workspaceValues = 0; // the fronts, update matrices and their right-hand sides at once (one run)
    // Of the working storage: the packed panels of the fronts made by tasks with several threads (TaskedFront, step
    // 10.36), two buffers of the largest front's (one for each front in flight in the top part).
    std::int64_t packingValues = 0;
    std::int64_t residentValues = 0; // the panels and right-hand sides made at once beside those kept
    std::int64_t keptValues = 0;     // the most panels and right-hand sides kept at once (the top, or a region run)
    std::int64_t retainedValues = 0; // the regions retained, held from the forward run to their visit (regions)
    std::int64_t solutionValues = 0; // the most solutions held at once by the downward pass, and its gathered rows
    std::uint64_t arrayBytes = 0;    // the per-supernode arrays
    RegionSweepCost regions;         // the regions of the bound
    double flopCount = 0.0;          // one factorisation
    // At the peak of the working storage (the binding consumer): the update matrices waiting on the stack, and the
    // supernode being made with its panel and y_c and its own update (FrontStreamOrder::Postorder; the parallel order's
    // peak is the sum of its subtrees' and of its top's).
    std::int64_t peakStackValues = 0;
    std::int64_t peakMadeValues = 0;
    std::int32_t peakSupernode = -1;

    [[nodiscard]] std::uint64_t bytes() const noexcept;
};

// The solve of a positive definite system A·X = B without keeping its factor (streamed factorisation): the interface
// problem of a condensed analysis whose factor does not fit.
//
// Method: the multifrontal factorisation streamed (StreamedFactorisation: each panel made in a block of its own and
// released once used), FactorSolve's forward sweep run unit by unit as the panels are made, keeping the panels of the
// top of the tree up to the bound; the diagonal from the pivots kept; then the backward sweep top-down: the top's
// panels, then each subtree below it factorised again with as many of its top supernodes kept as fit the bound (at
// least its root), swept and released, its rest a forest handled the same way (the recovery by regions of a
// unit carried to a complete factor). Every panel is the factor's (the same fronts, kernels and schedule) and every
// value keeps FactorSolve's chain of terms, so X has the bits of FactorSolve::solve on a SupernodalFactor of the same
// symbolic factor, matrix and kernel family, whatever the bound and the threads.
//
// Cost: the forward run is one factorisation; the regions factorise again the subtrees below the top (none when the
// bound holds every panel: the factor made for the solve and released after it). Every right-hand side is swept in
// the one pass: a later solve factorises again.
//
// Thread-safety: stateless; the calls share nothing. Ownership: none (the caller's matrices outlive the call).
class StreamedSolve
{
public:
    // B ← A⁻¹·B (B of symbolic.dofCount() rows, in the equation numbering of the matrix), by regions of at most
    // regionValues panel values. Throws std::invalid_argument when the matrix or B do not have the symbolic factor's
    // dimension, the symbolic factor has a trailing set, or the options are not positive definite; MemoryBudgetError
    // before any numeric work; SingularityError for a zero pivot or when the ratio test fails and the policy stops (a
    // mechanism, as SupernodalFactor::factorise); OperationCancelledError.
    [[nodiscard]] static StreamedSolveResult solve(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                   DenseMatrixView rightHandSides, const DenseKernels& kernels,
                                                   const FactorOptions& options, base::MemoryBudget& budget,
                                                   std::int64_t regionValues);

    // An upper bound of what solve() reserves with a pool of threadCount threads (the right-hand sides are the
    // caller's), from the symbolic factor alone.
    [[nodiscard]] static StreamedSolveEstimate estimate(const SymbolicFactor& symbolic, std::size_t threadCount);
    // The regions solve() sweeps with this bound, from the symbolic factor alone (the figures solve() reports).
    [[nodiscard]] static RegionSweepCost regionCost(const SymbolicFactor& symbolic, std::int64_t regionValues);

    // The solve of a matrix and right-hand sides given front by front (FrontSource) without keeping
    // the factor, its solution handed to `sink` supernode by supernode: the upward pass factorises every front with
    // its source's terms and carries the right-hand sides (columnCount of them) in the fronts and update matrices (the
    // forward sweep: y_c = L_cc⁻¹·b_c, b_r −= L_rc·y_c), keeping the panels and y of the top up to regionValues; the
    // downward pass solves each supernode of the top (x_c = L_ccᵀ⁻¹·(D⁻¹·y_c − L_rcᵀ·x_r)), then each region below it
    // factorised again from the source (StreamedSolve's regions), each solution handed over as soon as it is known and
    // kept while a descendant reads it. Every value has the same bits whatever the bound and the threads; with a
    // source that adds a matrix's entries of each supernode's columns, the panels are those of a SupernodalFactor and
    // the solution FactorSolve's to round-off (its forward sweep sums the updates in the order of the tree).
    // `updateBlockColumns`: the update matrices stored by blocks of that many columns (UpdateLayout; 0 square): the
    // same bits whatever it is.
    // `retainedValues`: the most values of the regions below the top kept from the
    // upward pass to their visit instead of being factorised again, chosen from the symbolic factor alone (the
    // regions of the most flops saved a value first, each taken when it fits what is left); the same bits whatever it
    // is (a retained panel and y are those a region's run makes again).
    // Throws std::invalid_argument for a symbolic factor with a trailing set or options not positive definite,
    // MemoryBudgetError before any numeric work (estimate().bytes() of the same arguments is the reservation),
    // SingularityError (a zero pivot, or the ratio test of the first run under the policy), OperationCancelledError,
    // and what the source and the sink throw.
    [[nodiscard]] static StreamedSolveResult solve(
        const SymbolicFactor& symbolic, FrontSource& source, SolutionSink& sink, std::size_t columnCount,
        const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget,
        std::int64_t regionValues, FrontStreamOrder order = FrontStreamOrder::Parallel,
        std::int32_t updateBlockColumns = UpdateLayout::STREAMED_BLOCK_COLUMNS, std::int64_t retainedValues = 0);
    [[nodiscard]] static FrontStreamEstimate estimate(
        const SymbolicFactor& symbolic, std::size_t threadCount, std::size_t columnCount, std::int64_t regionValues,
        FrontStreamOrder order = FrontStreamOrder::Parallel,
        std::int32_t updateBlockColumns = UpdateLayout::STREAMED_BLOCK_COLUMNS, std::int64_t retainedValues = 0);
};

} // namespace oos::linalg
