#pragma once

#include "FactorSchedule.hpp"
#include "FrontAssembly.hpp"
#include "SolvePlan.hpp"
#include "WorkBuffer.hpp"
#include "WorkspacePool.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/KernelWorkspace.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::linalg
{

// The multifrontal factorisation of a positive definite matrix that keeps no factor (internal to linalg/supernodal;
// the streamed factorisation): each supernode's panel is made in a block of its
// own, handed with the other panels of its unit of the solve plan to the run's consumer (a forward sweep,
// StreamedSweeps) once the unit is complete, then released, unless the run keeps it (a region of a backward sweep). The
// fronts, the update matrices and their order are those of MultifrontalDriver in positive definite mode (the same
// assembly, the same kernel on the same front, the same schedule): every panel has the bits it has in a
// SupernodalFactor of the same symbolic factor, matrix and kernel family, whatever the threads, and so do K* and the
// pivots.
//
// The factor() it lends holds the index arrays of a SupernodalFactor (rows, eliminated columns, subtrees of the cut)
// and its solve plan, built before any numeric work from the symbolic structure (no pivot is delayed in positive
// definite mode): a panel is read through factor().panel(s) while it is resident. The pivots D of every eliminated
// column are kept (pivots(): one value an equation), and the ratio test of the first complete run gives diagnostics()
// as a factor's.
//
// Runs. run(): every subtree of the schedule's cut in parallel (the units of each handed over on its task's thread, in
// order), then afterSubtrees(), then the top supernodes in increasing order on the calling thread (each a unit of its
// own, with the kernels' threads), and K* assembled when asked. With a plan not by tree (a cut of fewer than two
// subtrees) every unit in order on the calling thread. runSubtreeOf(): the complete subtree of one supernode, nothing
// handed over (the panels a region keeps): the subtrees of the cut inside it in parallel and its top supernodes in
// order, or serially when it lies inside one subtree of the cut.
//
// Memory: the working storage of FactorSchedule and a bound of the resident panels (the units of the running subtrees,
// one top unit, the panels kept) are reserved before each run (MemoryBudgetError before any numeric work); the panels'
// blocks come from the same WorkspacePool as the fronts.
//
// Thread-safety: one run at a time; a run's tasks write only the supernodes they own. Ownership: references its
// arguments, which outlive it; owns the factor's arrays, the kept panels and the pivots (budgeted).
class StreamedFactorisation
{
public:
    // What a run hands over (run()).
    class UnitConsumer
    {
    public:
        virtual ~UnitConsumer() = default;
        // Every panel of the plan's unit `unit` is resident (factor().panel()) and its pivots are in pivots(). On the
        // thread of its subtree's task for a unit of a subtree (the units of one subtree in increasing order, the
        // subtrees at once); on the calling thread for a top unit, in increasing order after afterSubtrees(). A plan
        // not by tree: every unit in order on the calling thread.
        virtual void consumeUnit(std::size_t unit) = 0;
        // A plan by tree: every subtree's units consumed, before the first top unit; on the calling thread.
        virtual void afterSubtrees() = 0;
    };

    // Throws std::invalid_argument when the matrix is not of the symbolic factor's dimension or options are not
    // positive definite; MemoryBudgetError when the factor's arrays and plan do not fit.
    StreamedFactorisation(const SymbolicFactor& symbolic, const SymmetricCsc& matrix, const DenseKernels& kernels,
                          const FactorOptions& options, bool excludesTrailing, base::MemoryBudget& budget);
    StreamedFactorisation(const StreamedFactorisation&) = delete;
    StreamedFactorisation& operator=(const StreamedFactorisation&) = delete;
    ~StreamedFactorisation();

    [[nodiscard]] const SupernodalFactor& factor() const noexcept { return m_factor; }
    [[nodiscard]] const SolvePlan& plan() const noexcept { return m_factor.solvePlan(); }
    [[nodiscard]] std::int32_t supernodeCount() const noexcept { return m_factor.supernodeCount(); }
    // The pivots of the eliminated columns, by equation (eliminatedCount() values): those of the supernodes run so far.
    [[nodiscard]] std::span<const double> pivots() const noexcept { return m_pivots.span(); }
    // The ratio test of the first complete run (PivotDiagnostics of SupernodalFactor).
    [[nodiscard]] const PivotDiagnostics& diagnostics() const noexcept { return m_diagnostics; }
    // The panel values of supernode s (c·(c + r)).
    [[nodiscard]] std::int64_t panelValueCount(std::int32_t s) const noexcept;
    // The flops of the elimination of supernode s (FactorSchedule::supernodeFlops).
    [[nodiscard]] double supernodeFlops(std::int32_t s) const noexcept;
    // The working storage's bound and measured peak of the runs (the panels' blocks included, planned part).
    [[nodiscard]] const FactorWorkspaceUsage& workspaceUsage() const noexcept { return m_workspaceUsage; }
    // Factorises every supernode, handing each unit of the plan to `consumer` (may be null) once complete; keeps the
    // panels isKept marks (empty: none) until releaseKept(). With schurComplement (a partial factorisation): the
    // trailing front assembled into it, both triangles (PartialFactor's K*). Throws MemoryBudgetError before any
    // numeric work, SingularityError for a zero pivot (a mechanism, as SupernodalFactor) or when the ratio test fails
    // and the policy stops (the first run), OperationCancelledError, and what the consumer throws.
    void run(UnitConsumer* consumer, std::span<const std::uint8_t> isKept, DenseMatrix* schurComplement);
    // Factorises the complete subtree of supernode `root`, keeping the panels isKept marks. Throws as run().
    void runSubtreeOf(std::int32_t root, std::span<const std::uint8_t> isKept);
    // The first supernode of the subtree of s (its supernodes are [subtreeFirst(s), s]).
    [[nodiscard]] std::int32_t subtreeFirst(std::int32_t s) const noexcept
    {
        return m_subtreeFirst[static_cast<std::size_t>(s)];
    }
    // Releases the panels the runs kept.
    void releaseKept() noexcept;

private:
    enum class Status : std::uint8_t
    {
        Pending,
        Done,
        ZeroPivot,
        Skipped
    };
    struct PendingUpdate
    {
        WorkBuffer values;
        std::int32_t order = 0;
    };
    struct StackEntry
    {
        WorkBuffer values;
        std::int32_t order = 0;
    };

    // Assembles and eliminates supernode s from its children's update matrices into a panel block of its own (lent to
    // the factor) and an update matrix (returned, empty when it failed or was skipped).
    [[nodiscard]] WorkBuffer eliminate(std::int32_t s, std::span<const UpdateMatrix> children, bool isPlanned,
                                       const DenseKernels& kernels, base::ThreadPool* assemblyPool,
                                       std::int32_t& updateOrder);
    // The supernodes [first, last] (complete subtrees) in postorder on this thread (isParallel: as a task of the
    // subtrees, with the kernels on this thread), the units handed to `consumer` (null: every panel not kept released
    // once made); handsToSlots: an update matrix whose parent is beyond the range goes to its pending slot (a run),
    // else it is dropped (a region).
    void runSubtree(std::int32_t first, std::int32_t last, bool isParallel, bool handsToSlots, UnitConsumer* consumer,
                    std::span<const std::uint8_t> isKept);
    void runTop(std::int32_t s, UnitConsumer* consumer, std::span<const std::uint8_t> isKept);
    // The supernodes [first, last] (complete subtrees of the root `last`, or every supernode), as run() describes.
    void runWindow(std::int32_t first, std::int32_t last, UnitConsumer* consumer, std::span<const std::uint8_t> isKept,
                   DenseMatrix* schurComplement);
    void assembleTrailing(DenseMatrix& schurComplement);
    // After the supernodes of a unit are done: the consumer, then the panels not kept released.
    void finishUnit(std::size_t unit, UnitConsumer* consumer, std::span<const std::uint8_t> isKept);
    void releasePanel(std::int32_t s) noexcept;
    void throwIfSingular() const;
    void finishFirstRun();
    [[nodiscard]] base::MemoryReservation reserveRun(std::int64_t panelValues);

    const SymbolicFactor& m_symbolic;
    const SymmetricCsc& m_matrix;
    KernelWorkspace m_kernelWorkspace; // declared before the kernels that name it
    const DenseKernels m_kernels;
    FactorOptions m_options;
    base::MemoryBudget& m_budget;
    FactorSchedule m_schedule;
    std::size_t m_plannedConcurrency = 1;
    std::atomic<std::size_t> m_runningSubtrees{0};
    WorkspaceMeter m_meter;
    // Made per run (its capacity the run's reservation), kept with the reservation while panels are kept (they go back
    // to it): declared before every WorkBuffer member.
    base::MemoryReservation m_keptReservation;
    std::unique_ptr<WorkspacePool> m_pool;
    SupernodalFactor m_factor;
    base::MemoryReservation m_panelTableReservation; // m_panels (declared first: reserved before)
    std::vector<WorkBuffer> m_panels;                // per supernode: its block while resident
    base::BudgetedBuffer<Status> m_status;
    base::BudgetedBuffer<std::int32_t> m_zeroPivotAt;
    base::BudgetedBuffer<std::int32_t> m_unitEndingAt; // per supernode: the plan unit it ends, or −1
    base::BudgetedBuffer<std::int32_t> m_subtreeFirst; // per supernode: the first supernode of its subtree
    base::BudgetedBuffer<std::uint8_t> m_isTop;        // per supernode: above the subtrees of the cut
    base::BudgetedBuffer<double> m_pivots;
    base::MemoryReservation m_pendingReservation;
    std::vector<PendingUpdate> m_pending; // per slot of the schedule
    PivotDiagnostics m_diagnostics;
    bool m_hasRunAll = false;
    std::int64_t m_largestSubtreeUnitValues = 0; // the largest panels of a unit of a subtree
    std::int64_t m_largestTopValues = 0;         // the largest panel of a top supernode
    FactorWorkspaceUsage m_workspaceUsage;
};

} // namespace oos::linalg
