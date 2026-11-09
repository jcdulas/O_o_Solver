#pragma once

#include "FactorSchedule.hpp"
#include "FrontAssembly.hpp"
#include "WorkBuffer.hpp"
#include "WorkspacePool.hpp"

#include "oos/base/memory/AlignedBuffer.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/KernelWorkspace.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
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

// The solve without the factor of a matrix and right-hand sides given front by front (internal to linalg/supernodal;
// StreamedSolve::solve on a FrontSource: the whole model streamed).
//
// Upward pass: the multifrontal factorisation in positive definite mode (MultifrontalDriver's fronts in place, the
// kernel factorFrontWithoutPivoting, FactorSchedule's subtrees in parallel and its top serially), each front assembled
// as cleared, then the source's terms, then the children's update matrices (assembleFrontInPhases), its right-hand
// sides the same way beside it. Once a front is factorised, the forward sweep of its right-hand sides (Golub and Van
// Loan, 4.1: L·y = b by blocks): y_c = L_cc⁻¹·b_c, then b_r ← b_r − L_rc·y_c, which goes up to the parent with the
// update matrix. Each panel and y_c is released at once but those of the supernodes kept (the top, or a region).
//
// Downward pass (L·D·Lᵀ·x = b: x_c = L_ccᵀ⁻¹·(D_c⁻¹·y_c − L_rcᵀ·x_r)): the top kept from the upward pass, from the root
// down; then the regions of StreamedSolve (regions::forEach), each subtree factorised again with its right-hand sides
// (the same terms, the same arithmetic: the same bits), its kept supernodes solved top-down. x_r is gathered from the
// solutions of the ancestors, each kept until every supernode of its subtree is solved. Each solution is handed to the
// sink as soon as it is known.
//
// Partial factor retention: with a retention budget, the regions whose runs save
// the most flops a value (planRetention, from the symbolic factor alone) keep their panels and y from the upward pass
// (a kept slot a supernode: the retained regions' first, then the top's or the current run's), and the downward pass
// solves each from them when it is visited instead of factorising its subtree again, then releases it. A retained
// panel and y are those its region's run would make (the same terms, order and kernels), so the bits are the same
// whatever the retention.
//
// Memory: what estimate() gives, reserved before any numeric work (MemoryBudgetError): the working storage (fronts,
// update matrices and their right-hand sides along the schedule, the panel and y being made), the largest kept set
// (panels and y), the regions retained, the solutions the downward pass holds and the rows it gathers, the
// per-supernode arrays. Every buffer of the passes comes from that reservation.
//
// Thread-safety: one solve at a time; the subtrees' tasks write only the supernodes they own. Ownership: references its
// arguments, which outlive it.
class FrontStream
{
public:
    // Throws std::invalid_argument for a symbolic factor with a trailing set or options not positive definite;
    // MemoryBudgetError for its arrays.
    FrontStream(const SymbolicFactor& symbolic, FrontSource& source, std::size_t columnCount,
                const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget,
                FrontStreamOrder order = FrontStreamOrder::Parallel,
                std::int32_t updateBlockColumns = UpdateLayout::STREAMED_BLOCK_COLUMNS);
    FrontStream(const FrontStream&) = delete;
    FrontStream& operator=(const FrontStream&) = delete;
    ~FrontStream();

    // Both passes, the solutions handed to `sink`. Throws as StreamedSolve::solve.
    [[nodiscard]] RegionSweepCost solve(SolutionSink& sink, std::int64_t regionValues, std::int64_t retainedValues = 0);

    [[nodiscard]] const PivotDiagnostics& diagnostics() const noexcept { return m_diagnostics; }
    [[nodiscard]] const FactorWorkspaceUsage& workspaceUsage() const noexcept { return m_workspaceUsage; }

    // The buffers of packed panels for the fronts made by tasks (TaskedFront): one for each front in flight.
    static constexpr std::int64_t PACKING_BUFFERS = 2;

    // StreamedSolve::estimate on a source.
    [[nodiscard]] static FrontStreamEstimate estimate(
        const SymbolicFactor& symbolic, std::size_t threadCount, std::size_t columnCount, std::int64_t regionValues,
        FrontStreamOrder order = FrontStreamOrder::Parallel,
        std::int32_t updateBlockColumns = UpdateLayout::STREAMED_BLOCK_COLUMNS, std::int64_t retainedValues = 0);

private:
    enum class Status : std::uint8_t
    {
        Pending,
        Done,
        ZeroPivot,
        Skipped
    };
    // An update matrix and its right-hand sides, handed to the parent.
    struct Update
    {
        WorkBuffer matrix;
        WorkBuffer rightHands;
        std::int32_t order = 0;
    };
    // A supernode kept from a factorisation for the downward pass: its panel and its y_c.
    struct Kept
    {
        WorkBuffer panel;
        WorkBuffer forward;
    };
    // The pivot ratios of the first factorisation of some supernodes (one task's), merged in the supernodes' order.
    struct RatioTest
    {
        std::vector<PivotIssue> issues; // the first RECORDED_ISSUE_COUNT of the task, in its order
        std::int64_t issueCount = 0;
        double largestRatio = 0.0;
        std::int32_t equationOfLargest = -1;
    };

    // The ratio test of MultifrontalDriver::finish in positive definite mode on an eliminated front's columns.
    void testRatios(const FrontLayout& layout, ConstDenseMatrixView panel, RatioTest& ratios) const;
    [[nodiscard]] Update eliminate(std::int32_t s, std::span<const UpdateMatrix> children,
                                   std::span<const UpdateRightHandSides> childRightHands, bool isPlanned,
                                   const DenseKernels& kernels, base::ThreadPool* assemblyPool, RatioTest& ratios);
    void runSubtree(std::int32_t first, std::int32_t last, bool isParallel, bool handsToSlots, RatioTest& ratios);
    void runTop(std::int32_t s, RatioTest& ratios);
    // The top supernodes `tops` (increasing) as one flow of tasks with one supernode of lookahead:
    // what runTop does for each, with the same bits, on the kernels' threads.
    struct TopFront;
    void runTopFlow(std::span<const std::int32_t> tops, RatioTest& ratios);
    // The upward pass's measured tasks made into the model on more threads (TaskModel).
    void modelTasks(std::span<const double> subtreeSeconds);
    // Factorises the supernodes [first, last] (the whole tree, or the complete subtree of `last`), keeping those of
    // [keptFirst, keptLast] that have a kept slot (m_keptSlot).
    void runWindow(std::int32_t first, std::int32_t last, std::int32_t keptFirst, std::int32_t keptLast);
    // Releases the kept supernodes of [first, last] (their panels and y, their slots).
    void releaseSlots(std::int32_t first, std::int32_t last) noexcept;
    void releaseKept() noexcept;
    void throwIfSingular(std::int32_t first, std::int32_t last) const;
    void finishRatioTest(std::vector<RatioTest>& tests);
    // The downward step of supernode s (kept): its solution made, handed to the sink, kept while its subtree reads it.
    void solveBackward(std::int32_t s, SolutionSink& sink);
    // Releases the solution of s and of every ancestor whose subtree is now solved.
    void completeSubtree(std::int32_t s) noexcept;
    // The supernode whose columns hold an equation.
    [[nodiscard]] std::int32_t supernodeOf(std::int32_t equation) const noexcept;
    [[nodiscard]] bool isKept(std::int32_t s) const noexcept
    {
        return m_keptFirst <= s && s <= m_keptLast && m_keptSlot[static_cast<std::size_t>(s)] >= 0;
    }

    const SymbolicFactor& m_symbolic;
    FrontSource& m_source;
    std::size_t m_columnCount = 0;
    FrontStreamOrder m_order = FrontStreamOrder::Parallel;
    std::int32_t m_blockColumns = UpdateLayout::STREAMED_BLOCK_COLUMNS;
    KernelWorkspace m_kernelWorkspace; // declared before the kernels that name it
    const DenseKernels m_kernels;
    FactorOptions m_options;
    base::MemoryBudget& m_budget;
    FactorSchedule m_schedule;
    std::size_t m_plannedConcurrency = 1;
    std::atomic<std::size_t> m_runningSubtrees{0};
    WorkspaceMeter m_meter;
    std::unique_ptr<WorkspacePool> m_pool;
    WorkBuffer m_packing[PACKING_BUFFERS]; // the packed panels of the fronts made by tasks (planned, from the pool)
    std::int32_t m_count = 0;
    base::BudgetedBuffer<Status> m_status;
    base::BudgetedBuffer<std::int32_t> m_zeroPivotAt;
    base::BudgetedBuffer<std::int32_t> m_subtreeFirst;
    base::BudgetedBuffer<std::uint8_t> m_isTop;
    base::BudgetedBuffer<std::int32_t> m_pendingChildren; // the downward pass: children whose subtree is not solved
    base::BudgetedBuffer<std::uint8_t> m_isSolved;
    base::BudgetedBuffer<std::int32_t> m_keptSlot;       // per supernode: its entry of m_kept, or −1
    base::BudgetedBuffer<std::uint8_t> m_isRetainedRoot; // per supernode: the root of a retained region
    base::MemoryReservation m_tableReservation;          // the pending slots, the kept and solution tables
    std::vector<Update> m_pending;                       // per slot of the schedule
    std::vector<Kept> m_kept;                            // by slot: the retained regions', then the top's or a run's
    std::int32_t m_keptFirst = 0; // the current run keeps the slotted supernodes of [first, last]
    std::int32_t m_keptLast = -1;
    std::vector<base::AlignedBuffer<double>> m_solutions; // per supernode: x_c while its subtree reads it
    std::int64_t m_liveSolutionValues = 0;
    std::int64_t m_solutionPeakValues = 0;
    PivotDiagnostics m_diagnostics;
    TaskModel m_taskModel; // the upward pass's (the first run of every supernode)
    bool m_hasRunAll = false;
    FactorWorkspaceUsage m_workspaceUsage;
};

} // namespace oos::linalg
