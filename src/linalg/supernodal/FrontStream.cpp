#include "FrontStream.hpp"

#include "RegionSweep.hpp"
#include "SolveSweeps.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/base/threading/TaskFlow.hpp"
#include "oos/base/threading/TaskGraph.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/TaskedFront.hpp"

#include <algorithm>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{
namespace
{

constexpr std::uint64_t VALUE_BYTES = sizeof(double);
// What the per-supernode arrays of a solve hold (FrontStream's statuses, zero pivots, subtree firsts, top flags,
// pending children, solved flags, kept slots, retained roots and solutions; FactorSchedule's children and slots; the
// flops and work of the regions), with room: bytes per supernode. The retention's candidates are counted apart.
constexpr std::uint64_t ARRAY_BYTES_PER_SUPERNODE = 64;
// The pending slots and the kept supernodes' tables: bytes per entry, with room (two WorkBuffers and an order).
constexpr std::uint64_t TABLE_BYTES_PER_ENTRY = 256;

std::size_t threadCountOf(const DenseKernels& kernels) noexcept
{
    return kernels.options().pool == nullptr ? 1 : kernels.options().pool->threadCount();
}

DenseKernelOptions withFactorisation(DenseKernelOptions options, const base::CancellationToken* cancellation,
                                     KernelWorkspace& workspace)
{
    if (options.cancellation == nullptr)
        options.cancellation = cancellation;
    if (options.workspace == nullptr)
        options.workspace = &workspace;
    return options;
}

// Counts a running subtree task for as long as it runs (MultifrontalDriver's admission).
class RunningSubtree
{
public:
    explicit RunningSubtree(std::atomic<std::size_t>& running) noexcept
        : m_running(running), m_count(running.fetch_add(1) + 1)
    {
    }
    RunningSubtree(const RunningSubtree&) = delete;
    RunningSubtree& operator=(const RunningSubtree&) = delete;
    ~RunningSubtree() { m_running.fetch_sub(1); }
    [[nodiscard]] std::size_t countAtStart() const noexcept { return m_count; }

private:
    std::atomic<std::size_t>& m_running;
    std::size_t m_count;
};

// The values of the buffers a supernode makes (WorkspacePool's class sizes): its panel and y_c, its update matrix and
// the right-hand sides it hands up.
struct SupernodeValues
{
    const SymbolicFactor& symbolic;
    std::int64_t columns;
    std::int64_t blockColumns;

    [[nodiscard]] std::int64_t panel(std::int32_t s) const noexcept
    {
        const std::int64_t c = symbolic.columnCount(s);
        return WorkspacePool::classSize(c * (c + symbolic.rowCount(s)));
    }
    [[nodiscard]] std::int64_t forward(std::int32_t s) const noexcept
    {
        return columns == 0 ? 0 : WorkspacePool::classSize(symbolic.columnCount(s) * columns);
    }
    [[nodiscard]] std::int64_t update(std::int32_t s) const noexcept
    {
        const std::int64_t r = symbolic.rowCount(s);
        return WorkspacePool::classSize(UpdateLayout{r, blockColumns}.valueCount()) +
               (columns == 0 ? 0 : WorkspacePool::classSize(r * columns));
    }
    // What a region keeps of it (the bound of the regions).
    [[nodiscard]] std::int64_t kept(std::int32_t s) const noexcept { return panel(s) + forward(s); }
};

// The packed panels of the fronts made by tasks (TaskedFront) with several threads: two buffers (the fronts
// in flight in the top part, FrontStream::runTopFlow) of the largest front's, in WorkspacePool's class sizes; none on
// one thread (the tasks are not taken there).
std::int64_t packingOf(const SymbolicFactor& symbolic, std::size_t threadCount, std::int64_t blockColumns)
{
    if (threadCount < 2)
        return 0;
    std::int64_t largest = 0;
    for (std::int32_t s = 0; s < symbolic.supernodeCount(); ++s)
        largest = std::max(largest, static_cast<std::int64_t>(TaskedFront::packingValues(
                                        static_cast<std::size_t>(symbolic.columnCount(s)),
                                        static_cast<std::size_t>(symbolic.rowCount(s)), blockColumns)));
    return largest == 0 ? 0 : FrontStream::PACKING_BUFFERS * WorkspacePool::classSize(largest);
}

// The working storage of a run of every supernode (FactorSchedule's bound with the right-hand sides, and the panel and
// y_c of the supernode being made counted with it): the subtrees of the cut serially, threadCount of them at once
// beside the update of every finished one, then the top one supernode after the other. A run of a region's subtree
// holds no more (its subtrees are some of the cut's, its top part a part of the top with fewer updates alive).
std::int64_t workspaceOf(const FactorSchedule& schedule, const SupernodeValues& values, std::size_t threadCount)
{
    std::vector<std::int64_t> excess; // per subtree of the cut, bounded by the cut
    std::int64_t finished = 0;
    for (const FactorSubtree& subtree : schedule.subtrees())
    {
        std::int64_t stack = 0;
        std::int64_t peak = 0;
        for (std::int32_t s = subtree.first; s <= subtree.last; ++s)
        {
            peak = std::max(peak, stack + values.update(s) + values.panel(s) + values.forward(s));
            for (const std::int32_t child : schedule.children(s))
                stack -= values.update(child);
            stack += values.update(s);
        }
        finished += values.update(subtree.last);
        excess.push_back(peak);
    }
    std::sort(excess.begin(), excess.end(), std::greater<>());
    std::int64_t peak = finished;
    for (std::size_t k = 0; k < std::min(threadCount, excess.size()); ++k)
        peak += excess[k];
    std::int64_t alive = finished;
    for (const std::int32_t s : schedule.topSupernodes())
    {
        peak = std::max(peak, alive + values.update(s) + values.panel(s) + values.forward(s));
        alive += values.update(s);
        for (const std::int32_t child : schedule.children(s))
            alive -= values.update(child);
    }
    return peak;
}

// The working storage of one postorder of every supernode on one thread (FrontStreamOrder::Postorder): Liu's stack of
// the update matrices with their right-hand sides, each supernode's update, panel and y_c made beside its children's.
std::int64_t postorderWorkspaceOf(const SymbolicFactor& symbolic, const FactorSchedule& schedule,
                                  const SupernodeValues& values, FrontStreamEstimate& estimate)
{
    std::int64_t stack = 0;
    std::int64_t peak = 0;
    for (std::int32_t s = 0; s < symbolic.supernodeCount(); ++s)
    {
        const std::int64_t made = values.update(s) + values.panel(s) + values.forward(s);
        if (stack + made > peak)
        {
            peak = stack + made;
            estimate.peakStackValues = stack;
            estimate.peakMadeValues = made;
            estimate.peakSupernode = s;
        }
        for (const std::int32_t child : schedule.children(s))
            stack -= values.update(child);
        stack += values.update(s);
    }
    return peak;
}

// The first supernode of the subtree of every supernode (a postorder: a child before its parent).
void fillSubtreeFirsts(const SymbolicFactor& symbolic, std::span<std::int32_t> first)
{
    const std::int32_t count = symbolic.supernodeCount();
    for (std::int32_t s = 0; s < count; ++s)
        first[static_cast<std::size_t>(s)] = s;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent != Supernodes::ROOT)
            first[static_cast<std::size_t>(parent)] =
                std::min(first[static_cast<std::size_t>(parent)], first[static_cast<std::size_t>(s)]);
    }
}

// The order in which the downward pass solves the supernodes with a region bound: the top from the root down, then the
// kept supernodes of each region from its root down (regions::forEach), visit(s) for each; region(kept, root, values)
// before the supernodes of each region.
template <typename Visit, typename Region>
void forEachDownward(const SymbolicFactor& symbolic, std::span<const std::int32_t> subtreeFirst,
                     const SupernodeValues& values, std::int64_t regionValues, Visit visit, Region region)
{
    const std::int32_t count = symbolic.supernodeCount();
    const auto keptOf = [&](std::int32_t s) { return values.kept(s); };
    const std::int32_t top = regions::topFirst(count, regionValues, keptOf);
    for (std::int32_t s = count - 1; s >= top; --s)
        visit(s);
    regions::forEach(
        top, regionValues, keptOf, [&](std::int32_t s) { return subtreeFirst[static_cast<std::size_t>(s)]; },
        [&](std::int32_t kept, std::int32_t root, std::int64_t keptValues)
        {
            region(kept, root, keptValues);
            for (std::int32_t s = root; s >= kept; --s)
                visit(s);
        });
}

// The retention of the regions below the top: the regions retained, their values,
// supernodes, and the flops their runs would take.
struct RetentionPlan
{
    std::int64_t values = 0;
    std::int64_t regionCount = 0;
    std::int64_t supernodeCount = 0;
    double flops = 0.0;
    std::int64_t candidateCount = 0; // the regions ranked (the plan's transient buffers)
};

// A region below the top as the retention ranks it.
struct RetentionCandidate
{
    std::int32_t kept = 0;
    std::int32_t root = 0;
    std::int64_t values = 0;
    double flops = 0.0;
    double work = 0.0;
    double workPerValue = 0.0;
    std::int32_t visit = 0;
};

// The work of making supernode s again in a region's run: its factorisation's flops and the forward sweep
// of the right-hand sides through it, y_c = L_cc⁻¹·b_c (c² flops a column) and b_r −= L_rc·y_c (2·r·c): with many
// right-hand sides the sweeps of the small supernodes outweigh their factorisation.
double runWork(std::int64_t columns, std::int64_t rows, std::int64_t rightHandSides) noexcept
{
    return FactorSchedule::supernodeFlops(columns, rows) +
           static_cast<double>(rightHandSides) * static_cast<double>(columns * columns + 2 * rows * columns);
}

// Marks in isRetainedRoot the roots of the regions below `top` retained within retainedValues: the regions of the
// bound (regions::forEach, its visit order), each saving the work of its run (the factorisation of its whole subtree
// and the sweeps of its right-hand sides, workBefore's prefix sums over the postorder; flopsBefore's the factorisation
// alone) for its own values (its kept supernodes' panels and y), taken in decreasing order of work a value (ties: the
// larger saving, then the region visited first) when they fit what is
// left (a greedy knapsack: the regions are a partition of the supernodes below the top and their runs independent, so
// the saving of a region does not depend on which others are retained). The candidates are budgeted and released here.
template <typename SubtreeFirst>
RetentionPlan planRetention(std::int32_t top, std::int64_t regionValues, std::int64_t retainedValues,
                            const SupernodeValues& values, SubtreeFirst subtreeFirst,
                            std::span<const double> flopsBefore, std::span<const double> workBefore,
                            std::span<std::uint8_t> isRetainedRoot, base::MemoryBudget& budget)
{
    std::fill(isRetainedRoot.begin(), isRetainedRoot.end(), std::uint8_t{0});
    RetentionPlan plan;
    if (retainedValues <= 0 || top <= 0)
        return plan;
    const auto keptOf = [&](std::int32_t s) { return values.kept(s); };
    std::size_t regionCount = 0;
    regions::forEach(top, regionValues, keptOf, subtreeFirst,
                     [&](std::int32_t, std::int32_t, std::int64_t) { ++regionCount; });
    plan.candidateCount = static_cast<std::int64_t>(regionCount);
    base::BudgetedBuffer<RetentionCandidate> candidates(budget, regionCount, "front stream: regions to retain");
    base::BudgetedBuffer<std::int32_t> order(budget, regionCount, "front stream: retention order");
    std::int32_t visit = 0;
    regions::forEach(top, regionValues, keptOf, subtreeFirst,
                     [&](std::int32_t kept, std::int32_t root, std::int64_t keptValues)
                     {
                         const auto first = static_cast<std::size_t>(subtreeFirst(root));
                         const double flops = flopsBefore[static_cast<std::size_t>(root) + 1] - flopsBefore[first];
                         const double work = workBefore[static_cast<std::size_t>(root) + 1] - workBefore[first];
                         const double perValue = work / static_cast<double>(std::max<std::int64_t>(keptValues, 1));
                         candidates[static_cast<std::size_t>(visit)] = {kept, root,     keptValues, flops,
                                                                        work, perValue, visit};
                         order[static_cast<std::size_t>(visit)] = visit;
                         ++visit;
                     });
    std::sort(order.span().begin(), order.span().end(),
              [&](std::int32_t a, std::int32_t b)
              {
                  const RetentionCandidate& x = candidates[static_cast<std::size_t>(a)];
                  const RetentionCandidate& y = candidates[static_cast<std::size_t>(b)];
                  if (x.workPerValue != y.workPerValue)
                      return x.workPerValue > y.workPerValue;
                  if (x.work != y.work)
                      return x.work > y.work;
                  return x.visit < y.visit;
              });
    for (const std::int32_t at : order.span())
    {
        const RetentionCandidate& candidate = candidates[static_cast<std::size_t>(at)];
        if (candidate.values > retainedValues - plan.values)
            continue;
        isRetainedRoot[static_cast<std::size_t>(candidate.root)] = 1;
        plan.values += candidate.values;
        ++plan.regionCount;
        plan.supernodeCount += candidate.root - candidate.kept + 1;
        plan.flops += candidate.flops;
    }
    return plan;
}

} // namespace

std::int32_t FrontTerms::positionOf(std::int32_t equation) const noexcept
{
    if (equation >= firstColumn && equation < firstColumn + columnCount)
        return equation - firstColumn;
    const auto found = std::lower_bound(rows.begin(), rows.end(), equation);
    if (found == rows.end() || *found != equation)
        return -1;
    return columnCount + static_cast<std::int32_t>(found - rows.begin());
}

std::uint64_t FrontStreamEstimate::bytes() const noexcept
{
    return static_cast<std::uint64_t>(workspaceValues + keptValues + retainedValues + solutionValues) * VALUE_BYTES +
           arrayBytes;
}

FrontStreamEstimate FrontStream::estimate(const SymbolicFactor& symbolic, std::size_t threadCount,
                                          std::size_t columnCount, std::int64_t regionValues, FrontStreamOrder order,
                                          std::int32_t updateBlockColumns, std::int64_t retainedValues)
{
    FrontStreamEstimate estimate;
    const std::int32_t count = symbolic.supernodeCount();
    estimate.regions.regionValues = regionValues;
    if (count == 0)
        return estimate;
    base::MemoryBudget scratch(base::MemoryBudget::UNLIMITED);
    const FactorSchedule schedule(symbolic, false, std::max<std::size_t>(threadCount, 1),
                                  PivotingMode::PositiveDefinite, scratch, FactorSchedule::TopOrder::Serial);
    const SupernodeValues values{symbolic, static_cast<std::int64_t>(columnCount), updateBlockColumns};
    estimate.workspaceValues = order == FrontStreamOrder::Postorder
                                   ? postorderWorkspaceOf(symbolic, schedule, values, estimate)
                                   : workspaceOf(schedule, values, std::max<std::size_t>(threadCount, 1));
    estimate.packingValues = packingOf(symbolic, threadCount, updateBlockColumns);
    estimate.workspaceValues += estimate.packingValues;
    const auto total = static_cast<std::size_t>(count);
    base::BudgetedBuffer<std::int32_t> subtreeFirst(scratch, total, "front stream estimate: subtree firsts");
    fillSubtreeFirsts(symbolic, subtreeFirst.span());
    base::BudgetedBuffer<double> flopsBefore(scratch, total + 1, "front stream estimate: flops");
    base::BudgetedBuffer<double> workBefore(scratch, total + 1, "front stream estimate: work");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        const double flops = FactorSchedule::supernodeFlops(symbolic.columnCount(s), symbolic.rowCount(s));
        flopsBefore[at + 1] = flopsBefore[at] + flops;
        workBefore[at + 1] = workBefore[at] + runWork(symbolic.columnCount(s), symbolic.rowCount(s),
                                                      static_cast<std::int64_t>(columnCount));
        estimate.residentValues = std::max(estimate.residentValues, values.kept(s));
    }
    estimate.flopCount = flopsBefore[total];
    estimate.regions.upwardWork = workBefore[total];
    const bool isSerialOrder = order == FrontStreamOrder::Postorder || schedule.subtrees().size() < 2;
    if (isSerialOrder)
        estimate.regions.serialUpwardWork = estimate.regions.upwardWork;

    // The downward pass, as solve() runs it: the solutions held (exact sizes) and the rows gathered for a supernode.
    base::BudgetedBuffer<std::int32_t> pending(scratch, total, "front stream estimate: pending children");
    base::BudgetedBuffer<std::uint8_t> isSolved(scratch, total, "front stream estimate: solved");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent != Supernodes::ROOT)
            ++pending[static_cast<std::size_t>(parent)];
    }
    const auto k = static_cast<std::int64_t>(columnCount);
    std::int64_t live = 0;
    std::int64_t keptCount = 0;
    const std::int32_t top = regions::topFirst(count, regionValues, [&](std::int32_t s) { return values.kept(s); });
    for (std::int32_t s = top; s < count; ++s)
        estimate.regions.topValues += values.kept(s);
    estimate.keptValues = estimate.regions.topValues;
    estimate.regions.keptValues = estimate.regions.topValues;
    keptCount = count - top;
    base::BudgetedBuffer<std::uint8_t> isRetainedRoot(scratch, total, "front stream estimate: retained roots");
    const RetentionPlan retention = planRetention(
        top, regionValues, retainedValues, values,
        [&](std::int32_t s) { return subtreeFirst[static_cast<std::size_t>(s)]; }, flopsBefore.span(),
        workBefore.span(), isRetainedRoot.span(), scratch);
    estimate.retainedValues = retention.values;
    forEachDownward(
        symbolic, subtreeFirst.span(), values, regionValues,
        [&](std::int32_t s)
        {
            const auto at = static_cast<std::size_t>(s);
            live += symbolic.columnCount(s) * k;
            estimate.solutionValues = std::max(estimate.solutionValues, live + symbolic.rowCount(s) * k);
            isSolved[at] = 1;
            for (std::int32_t done = s; done != Supernodes::ROOT && isSolved[static_cast<std::size_t>(done)] != 0 &&
                                        pending[static_cast<std::size_t>(done)] == 0;)
            {
                live -= symbolic.columnCount(done) * k;
                isSolved[static_cast<std::size_t>(done)] = 2; // released
                const std::int32_t parent = symbolic.supernodes().parent(done);
                if (parent == Supernodes::ROOT)
                    break;
                --pending[static_cast<std::size_t>(parent)];
                done = isSolved[static_cast<std::size_t>(parent)] == 1 ? parent : Supernodes::ROOT;
            }
        },
        [&](std::int32_t kept, std::int32_t root, std::int64_t keptValues)
        {
            const std::int32_t first = subtreeFirst[static_cast<std::size_t>(root)];
            const double flops =
                flopsBefore[static_cast<std::size_t>(root) + 1] - flopsBefore[static_cast<std::size_t>(first)];
            const double work =
                workBefore[static_cast<std::size_t>(root) + 1] - workBefore[static_cast<std::size_t>(first)];
            if (isRetainedRoot[static_cast<std::size_t>(root)] != 0)
            {
                ++estimate.regions.retainedRunCount;
                estimate.regions.retainedValues += keptValues;
                estimate.regions.retainedFlops += flops;
                estimate.regions.retainedWork += work;
                return;
            }
            ++estimate.regions.regionRunCount;
            // A run made in one postorder on the calling thread (runWindow's rule): its subtrees not on the threads.
            bool isSerial = isSerialOrder;
            for (const FactorSubtree& subtree : schedule.subtrees())
                isSerial = isSerial || (subtree.first <= first && root <= subtree.last);
            if (isSerial)
                estimate.regions.serialRunWork += work;
            estimate.keptValues = std::max(estimate.keptValues, keptValues);
            estimate.regions.keptValues = std::max(estimate.regions.keptValues, keptValues);
            keptCount = std::max<std::int64_t>(keptCount, root - kept + 1);
            estimate.regions.recomputedFlops += flops;
            estimate.regions.recomputedWork += work;
        });
    // The kept slots: the retained regions' supernodes, then the top's or a run's (FrontStream::solve).
    estimate.arrayBytes =
        static_cast<std::uint64_t>(count) * ARRAY_BYTES_PER_SUPERNODE +
        (static_cast<std::uint64_t>(schedule.slotCount()) + static_cast<std::uint64_t>(keptCount) +
         static_cast<std::uint64_t>(retention.supernodeCount)) *
            TABLE_BYTES_PER_ENTRY +
        static_cast<std::uint64_t>(retention.candidateCount) * (sizeof(RetentionCandidate) + sizeof(std::int32_t));
    return estimate;
}

FrontStream::FrontStream(const SymbolicFactor& symbolic, FrontSource& source, std::size_t columnCount,
                         const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget,
                         FrontStreamOrder order, std::int32_t updateBlockColumns)
    : m_symbolic(symbolic),
      m_source(source),
      m_columnCount(columnCount),
      m_order(order),
      m_blockColumns(updateBlockColumns),
      m_kernels(kernels.kind(), withFactorisation(kernels.options(), options.cancellation, m_kernelWorkspace)),
      m_options(options),
      m_budget(budget),
      m_schedule(symbolic, false, threadCountOf(kernels), PivotingMode::PositiveDefinite, budget,
                 FactorSchedule::TopOrder::Serial),
      m_plannedConcurrency(threadCountOf(kernels)),
      m_count(symbolic.supernodeCount())
{
    if (options.pivoting != PivotingMode::PositiveDefinite)
        throw std::invalid_argument(
            "StreamedSolve::solve: only a positive definite matrix is solved without its factor (no pivot is "
            "delayed); pass PivotingMode::PositiveDefinite");
    if (symbolic.trailingDofCount() > 0)
        throw std::invalid_argument(base::concatenate(
            {"StreamedSolve::solve: the symbolic factor leaves ", std::to_string(symbolic.trailingDofCount()),
             " trailing equations unfactorised; analyse the whole matrix, without a trailing set"}));
    const auto total = static_cast<std::size_t>(m_count);
    m_status = base::BudgetedBuffer<Status>(budget, total, "front stream: supernode status");
    m_zeroPivotAt = base::BudgetedBuffer<std::int32_t>(budget, total, "front stream: zero pivots");
    m_subtreeFirst = base::BudgetedBuffer<std::int32_t>(budget, total, "front stream: subtree firsts");
    fillSubtreeFirsts(symbolic, m_subtreeFirst.span());
    m_isTop = base::BudgetedBuffer<std::uint8_t>(budget, total, "front stream: top supernodes");
    for (const std::int32_t s : m_schedule.topSupernodes())
        m_isTop[static_cast<std::size_t>(s)] = 1;
    m_pendingChildren = base::BudgetedBuffer<std::int32_t>(budget, total, "front stream: pending children");
    m_isSolved = base::BudgetedBuffer<std::uint8_t>(budget, total, "front stream: solved supernodes");
    m_keptSlot = base::BudgetedBuffer<std::int32_t>(budget, total, "front stream: kept slots", base::FOR_OVERWRITE);
    std::fill(m_keptSlot.span().begin(), m_keptSlot.span().end(), -1);
    m_isRetainedRoot = base::BudgetedBuffer<std::uint8_t>(budget, total, "front stream: retained roots");
    const auto slots = static_cast<std::size_t>(m_schedule.slotCount());
    m_tableReservation = budget.reserve(slots * sizeof(Update) + total * sizeof(base::AlignedBuffer<double>),
                                        "front stream: pending slots and solutions");
    m_pending.resize(slots);
    m_solutions.resize(total);
}

FrontStream::~FrontStream()
{
    releaseKept();
    m_pending.clear();
}

std::int32_t FrontStream::supernodeOf(std::int32_t equation) const noexcept
{
    // The columns of the supernodes increase with them (a postorder numbers the columns in order).
    std::int32_t low = 0;
    std::int32_t high = m_count - 1;
    while (low < high)
    {
        const std::int32_t middle = low + (high - low + 1) / 2;
        if (m_symbolic.firstColumn(middle) <= equation)
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

FrontStream::Update FrontStream::eliminate(std::int32_t s, std::span<const UpdateMatrix> children,
                                           std::span<const UpdateRightHandSides> childRightHands, bool isPlanned,
                                           const DenseKernels& kernels, base::ThreadPool* assemblyPool,
                                           RatioTest& ratios)
{
    const auto at = static_cast<std::size_t>(s);
    FrontLayout layout;
    layout.supernode = s;
    layout.firstColumn = m_symbolic.firstColumn(s);
    layout.columnCount = m_symbolic.columnCount(s);
    layout.rows = m_symbolic.rows(s);
    const auto c = static_cast<std::size_t>(layout.columnCount);
    const std::size_t r = layout.rows.size();
    const std::size_t order = c + r;
    const std::size_t k = m_columnCount;
    const SupernodeValues values{m_symbolic, static_cast<std::int64_t>(k), m_blockColumns};

    WorkBuffer panel(isPlanned ? values.panel(s) : 0, "front stream: panel", &m_meter, m_pool.get());
    panel.ensure(static_cast<std::int64_t>(c * order), m_budget);
    WorkBuffer forward(isPlanned ? values.forward(s) : 0, "front stream: forward sweep", &m_meter, m_pool.get());
    forward.ensure(static_cast<std::int64_t>(c * k), m_budget);
    // PERFORMANCE: the update matrix by blocks of columns (UpdateLayout), its upper triangle not stored: the working
    // storage of the top of the tree, where the update matrices dominate, about halved; each block's update is made by
    // updateSymmetric and updateRectangular, every element the chain of factorFrontWithoutPivoting.
    // Measured: a large model's whole tree, the postorder's working storage.
    const UpdateLayout updateLayout{static_cast<std::int64_t>(r), m_blockColumns};
    Update update;
    update.matrix = WorkBuffer(isPlanned ? WorkspacePool::classSize(updateLayout.valueCount()) : 0,
                               "front stream: update matrix", &m_meter, m_pool.get());
    update.matrix.ensure(updateLayout.valueCount(), m_budget);
    update.rightHands = WorkBuffer(isPlanned && k > 0 ? WorkspacePool::classSize(static_cast<std::int64_t>(r * k)) : 0,
                                   "front stream: right-hand sides handed up", &m_meter, m_pool.get());
    update.rightHands.ensure(static_cast<std::int64_t>(r * k), m_budget);

    const DenseMatrixView panelView{panel.data(), order, c, std::max<std::size_t>(order, 1)};
    const DenseMatrixView contribution{update.matrix.data(), r, r, std::max<std::size_t>(r, 1)};
    const DenseMatrixView columnsView{forward.data(), c, k, std::max<std::size_t>(c, 1)};
    const DenseMatrixView rowsView{update.rightHands.data(), r, k, std::max<std::size_t>(r, 1)};
    FrontTerms terms;
    terms.supernode = s;
    terms.firstColumn = layout.firstColumn;
    terms.columnCount = layout.columnCount;
    terms.rows = layout.rows;
    terms.panel = panelView;
    terms.contribution = contribution;
    terms.contributionBlockColumns = m_blockColumns;
    terms.rightHandColumns = columnsView;
    terms.rightHandRows = rowsView;
    // The front: cleared, the source's terms, then the children's (assembleFront's order for a matrix's entries); its
    // right-hand sides the same way.
    assembleFrontInPhases(
        layout, children, FrontStorage::split(panelView, contribution, m_blockColumns),
        [&]
        {
            clearRightHandSides(columnsView, rowsView);
            m_source.addTerms(terms);
        },
        assemblyPool, m_options.cancellation);
    addChildRightHandSides(layout, childRightHands, columnsView, rowsView);

    // The panel and the update matrix by blocks (DenseKernels: the tasks of a large front with the kernels' threads,
    // the packed panels in the first of the planned buffers: one front at a time here).
    const std::span<double> packing =
        kernels.options().pool == nullptr || m_packing[0].capacity() == 0
            ? std::span<double>()
            : std::span<double>(m_packing[0].data(), static_cast<std::size_t>(m_packing[0].capacity()));
    const LdltOutcome outcome =
        kernels.factorFrontWithoutPivoting(panelView, contribution, updateLayout.blockColumns, packing);
    if (!outcome.isComplete)
    {
        m_status[at] = Status::ZeroPivot;
        m_zeroPivotAt[at] = layout.firstColumn + static_cast<std::int32_t>(outcome.eliminatedCount);
        return {};
    }
    if (!m_hasRunAll)
        testRatios(layout, panelView, ratios);
    // The forward sweep of the right-hand sides: y_c = L_cc⁻¹·b_c, b_r ← b_r − L_rc·y_c.
    if (k > 0 && c > 0)
    {
        kernels.solveUnitLower(panelView.block(0, 0, c, c), columnsView);
        if (r > 0)
            kernels.subtractProduct(rowsView, panelView.block(c, 0, r, c), columnsView);
    }
    m_status[at] = Status::Done;
    update.order = static_cast<std::int32_t>(r);
    if (isKept(s))
    {
        Kept& kept = m_kept[static_cast<std::size_t>(m_keptSlot[at])];
        kept.panel = std::move(panel);
        kept.forward = std::move(forward);
    }
    return update;
}

void FrontStream::testRatios(const FrontLayout& layout, ConstDenseMatrixView panel, RatioTest& ratios) const
{
    // The ratio test of MultifrontalDriver::finish in positive definite mode, a_jj / d_j per column.
    const auto c = static_cast<std::size_t>(layout.columnCount);
    std::vector<double> diagonal(c); // bounded by the supernode
    m_source.diagonal(layout.supernode, diagonal);
    for (std::size_t j = 0; j < c; ++j)
    {
        const std::int32_t equation = layout.firstColumn + static_cast<std::int32_t>(j);
        const double pivot = panel(j, j);
        const double ratio = diagonal[j] / pivot;
        if (ratios.equationOfLargest < 0 || ratio > ratios.largestRatio)
        {
            ratios.largestRatio = ratio;
            ratios.equationOfLargest = equation;
        }
        if (ratio > m_options.pivotPolicy.maxRatio || ratio < 0.0)
        {
            ++ratios.issueCount;
            if (ratios.issues.size() < PivotDiagnostics::RECORDED_ISSUE_COUNT)
                ratios.issues.push_back(
                    {equation, diagonal[j], pivot, ratio,
                     ratio < 0.0 ? PivotIssueKind::NegativePivot : PivotIssueKind::RatioAboveMaximum});
        }
    }
}

void FrontStream::runSubtree(std::int32_t first, std::int32_t last, bool isParallel, bool handsToSlots,
                             RatioTest& ratios)
{
    std::unique_ptr<RunningSubtree> running;
    if (isParallel)
        running = std::make_unique<RunningSubtree>(m_runningSubtrees);
    const bool isPlanned = !isParallel || running->countAtStart() <= m_plannedConcurrency;
    DenseKernelOptions serialOptions = m_kernels.options();
    serialOptions.pool = nullptr;
    const DenseKernels serial(m_kernels.kind(), serialOptions);
    const DenseKernels& kernels = isParallel ? serial : m_kernels;
    std::vector<Update> entries; // bounded by the depth of the range
    std::vector<UpdateMatrix> children;
    std::vector<UpdateRightHandSides> childRightHands;
    for (std::int32_t s = first; s <= last; ++s)
    {
        if (m_options.cancellation != nullptr)
            m_options.cancellation->throwIfCancelled();
        const auto childSupernodes = m_schedule.children(s);
        OOS_ASSERT(entries.size() >= childSupernodes.size());
        const std::size_t firstChildEntry = entries.size() - childSupernodes.size();
        bool areChildrenDone = true;
        for (const std::int32_t child : childSupernodes)
            areChildrenDone = areChildrenDone && m_status[static_cast<std::size_t>(child)] == Status::Done;
        Update entry;
        if (areChildrenDone)
        {
            children.clear();
            childRightHands.clear();
            for (std::size_t k = firstChildEntry; k < entries.size(); ++k)
            {
                const auto rows = m_symbolic.rows(childSupernodes[k - firstChildEntry]);
                children.push_back({entries[k].matrix.data(), entries[k].order, {}, rows, m_blockColumns});
                childRightHands.push_back({entries[k].rightHands.data(), rows});
            }
            entry = eliminate(s, children, childRightHands, isPlanned, kernels, nullptr, ratios);
        }
        else
            m_status[static_cast<std::size_t>(s)] = Status::Skipped;
        entries.resize(firstChildEntry);
        const std::int32_t parent = m_symbolic.supernodes().parent(s);
        if (parent == Supernodes::ROOT || parent > last)
        {
            const std::int32_t slot = m_schedule.slotOf(s);
            if (handsToSlots && slot != FactorSchedule::NO_SLOT &&
                m_status[static_cast<std::size_t>(s)] == Status::Done)
                m_pending[static_cast<std::size_t>(slot)] = std::move(entry);
        }
        else
            entries.push_back(std::move(entry));
        if (m_options.progress != nullptr && !m_hasRunAll)
            m_options.progress->advance(static_cast<std::uint64_t>(m_symbolic.columnCount(s)));
    }
    OOS_ASSERT(entries.empty());
}

void FrontStream::runTop(std::int32_t s, RatioTest& ratios)
{
    if (m_options.cancellation != nullptr)
        m_options.cancellation->throwIfCancelled();
    const auto childSupernodes = m_schedule.children(s);
    bool areChildrenDone = true;
    for (const std::int32_t child : childSupernodes)
        areChildrenDone = areChildrenDone && m_status[static_cast<std::size_t>(child)] == Status::Done;
    if (!areChildrenDone)
    {
        m_status[static_cast<std::size_t>(s)] = Status::Skipped;
        for (const std::int32_t child : childSupernodes)
            m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))] = {};
        return;
    }
    std::vector<UpdateMatrix> children;
    std::vector<UpdateRightHandSides> childRightHands;
    for (const std::int32_t child : childSupernodes)
    {
        const Update& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))];
        children.push_back({pending.matrix.data(), pending.order, {}, m_symbolic.rows(child), m_blockColumns});
        childRightHands.push_back({pending.rightHands.data(), m_symbolic.rows(child)});
    }
    Update update = eliminate(s, children, childRightHands, true, m_kernels, m_kernels.options().pool, ratios);
    for (const std::int32_t child : childSupernodes)
        m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))] = {};
    if (m_status[static_cast<std::size_t>(s)] == Status::Done)
        m_pending[static_cast<std::size_t>(m_schedule.slotOf(s))] = std::move(update);
    if (m_options.progress != nullptr && !m_hasRunAll)
        m_options.progress->advance(static_cast<std::uint64_t>(m_symbolic.columnCount(s)));
}

// A top supernode in flight in the top flow (runTopFlow): its buffers, its views, its children and its tasks. Written
// by its own tasks; read by the next supernode's tasks only after the tasks that wrote it (the flow's dependencies).
struct FrontStream::TopFront
{
    std::int32_t s = 0;
    bool isAssembled = false; // its children were done: its front is made
    FrontLayout layout;
    WorkBuffer panel;
    WorkBuffer forward;
    Update update;
    const double* updateValues = nullptr;    // update.matrix's values (the WorkBuffer moves to its slot when done)
    const double* rightHandValues = nullptr; // update.rightHands' values
    std::vector<UpdateMatrix> children;
    std::vector<UpdateRightHandSides> childRightHands;
    std::size_t pendingChild = NO_PENDING_CHILD; // the child in flight (the supernode before it), by index
    std::unique_ptr<FrontAssembler> assembler;
    FrontTerms terms;
    DenseMatrixView panelView;
    DenseMatrixView contribution;
    DenseMatrixView columnsView;
    DenseMatrixView rowsView;
    std::unique_ptr<TaskedFront> tasked;    // null: the front's elimination in one task
    std::size_t eliminated = 0;             // the one task's
    std::vector<std::size_t> trailingFirst; // the update matrix's blocks of columns (and r)
    base::TaskFlow::TaskId panelDone = 0;
    base::TaskFlow::TaskId done = 0;
    base::TaskFlow::TaskId forwardDone = 0;
    base::TaskFlow::TaskId ratioDone = 0;
    base::TaskFlow::TaskId released = 0;
    base::TaskFlow::TaskId finished = 0;
    std::vector<base::TaskFlow::TaskId> trailingDone; // per block of the update matrix

    static constexpr std::size_t NO_PENDING_CHILD = static_cast<std::size_t>(-1);

    [[nodiscard]] std::size_t eliminatedCount() const noexcept
    {
        return tasked ? tasked->eliminatedCount() : eliminated;
    }
    [[nodiscard]] bool isComplete() const noexcept
    {
        return isAssembled && eliminatedCount() == static_cast<std::size_t>(layout.columnCount);
    }
};

void FrontStream::runTopFlow(std::span<const std::int32_t> tops, RatioTest& ratios)
{
    // PERFORMANCE: the top part as one flow of tasks: each supernode's clearing, own terms,
    // extend-add by ranges of columns, front tasks (TaskedFront), ratio test and forward sweep, and one supernode of
    // lookahead: the next one starts once this one's panel is eliminated and its children consumed, so that its
    // assembly (memory-bound) and its panel run beside this one's trailing update (compute-bound), and each extend-add
    // range waits only for the blocks of the pending child's update matrix it reads. Every element of every front
    // receives the sum and the chain it receives one supernode after the other: the same bits.
    // Measured on a series of large models, six threads.
    using TaskId = base::TaskFlow::TaskId;
    const std::size_t k = m_columnCount;
    const SupernodeValues values{m_symbolic, static_cast<std::int64_t>(k), m_blockColumns};
    DenseKernelOptions serialOptions = m_kernels.options();
    serialOptions.pool = nullptr;
    serialOptions.cancellation = nullptr;
    const DenseKernels serial(m_kernels.kind(), serialOptions);
    const bool isTaskedFamily = m_kernels.kind() != KernelKind::Reference;
    std::deque<TopFront> jobs; // stable while the flow adds them
    jobs.resize(tops.size());
    base::TaskFlow flow;
    // Priorities: the earlier supernode first; each supernode's assembly and panel above every trailing tile.
    constexpr std::int64_t SUPERNODE_PRIORITY_STRIDE = std::int64_t{1} << 34;
    constexpr std::int64_t ASSEMBLY_PRIORITY = TaskedFront::PANEL_PRIORITY + (std::int64_t{1} << 30);
    // The ranges of a front's columns its clearing and extend-add tasks take: within the panel or one block of the
    // update matrix, of about ASSEMBLY_RANGE_VALUES values.
    constexpr std::size_t ASSEMBLY_RANGE_VALUES = std::size_t{1} << 17;
    // The rows of a task of the forward sweep's b_r −= L_rc·y_c.
    constexpr std::size_t FORWARD_ROWS = 1024;

    std::function<void(std::size_t)> addStart;
    const auto start = [&](std::size_t i) -> bool
    {
        TopFront& job = jobs[i];
        const std::int32_t s = tops[i];
        job.s = s;
        const auto at = static_cast<std::size_t>(s);
        const std::int64_t base = -static_cast<std::int64_t>(i) * SUPERNODE_PRIORITY_STRIDE;
        TopFront* previous = i > 0 ? &jobs[i - 1] : nullptr;
        const auto childSupernodes = m_schedule.children(s);
        bool areChildrenDone = true;
        for (const std::int32_t child : childSupernodes)
        {
            if (previous != nullptr && child == previous->s)
                areChildrenDone = areChildrenDone && previous->isComplete();
            else
                areChildrenDone = areChildrenDone && m_status[static_cast<std::size_t>(child)] == Status::Done;
        }
        const auto always = [] { return true; };
        if (!areChildrenDone)
        {
            // As runTop: skipped, its children's update matrices released (the pending one's when it is finished).
            m_status[at] = Status::Skipped;
            job.layout.columnCount = m_symbolic.columnCount(s);
            const TaskId skipped = flow.add(
                [this, childSupernodes]
                {
                    for (const std::int32_t child : childSupernodes)
                        m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))] = {};
                    return true;
                },
                previous == nullptr ? std::vector<TaskId>{} : std::vector<TaskId>{previous->finished},
                base + ASSEMBLY_PRIORITY);
            job.panelDone = job.done = job.forwardDone = job.ratioDone = job.released = job.finished = skipped;
            if (i + 1 < jobs.size())
                addStart(i + 1);
            return true;
        }

        // The buffers (eliminate()'s), the views and the children.
        job.isAssembled = true;
        FrontLayout& layout = job.layout;
        layout.supernode = s;
        layout.firstColumn = m_symbolic.firstColumn(s);
        layout.columnCount = m_symbolic.columnCount(s);
        layout.rows = m_symbolic.rows(s);
        const auto c = static_cast<std::size_t>(layout.columnCount);
        const std::size_t r = layout.rows.size();
        const std::size_t order = c + r;
        job.panel = WorkBuffer(values.panel(s), "front stream: panel", &m_meter, m_pool.get());
        job.panel.ensure(static_cast<std::int64_t>(c * order), m_budget);
        job.forward = WorkBuffer(values.forward(s), "front stream: forward sweep", &m_meter, m_pool.get());
        job.forward.ensure(static_cast<std::int64_t>(c * k), m_budget);
        const UpdateLayout updateLayout{static_cast<std::int64_t>(r), m_blockColumns};
        job.update.matrix = WorkBuffer(WorkspacePool::classSize(updateLayout.valueCount()),
                                       "front stream: update matrix", &m_meter, m_pool.get());
        job.update.matrix.ensure(updateLayout.valueCount(), m_budget);
        job.update.rightHands = WorkBuffer(k > 0 ? WorkspacePool::classSize(static_cast<std::int64_t>(r * k)) : 0,
                                           "front stream: right-hand sides handed up", &m_meter, m_pool.get());
        job.update.rightHands.ensure(static_cast<std::int64_t>(r * k), m_budget);
        job.updateValues = job.update.matrix.data();
        job.rightHandValues = job.update.rightHands.data();
        job.panelView = DenseMatrixView{job.panel.data(), order, c, std::max<std::size_t>(order, 1)};
        job.contribution = DenseMatrixView{job.update.matrix.data(), r, r, std::max<std::size_t>(r, 1)};
        job.columnsView = DenseMatrixView{job.forward.data(), c, k, std::max<std::size_t>(c, 1)};
        job.rowsView = DenseMatrixView{job.update.rightHands.data(), r, k, std::max<std::size_t>(r, 1)};
        job.terms.supernode = s;
        job.terms.firstColumn = layout.firstColumn;
        job.terms.columnCount = layout.columnCount;
        job.terms.rows = layout.rows;
        job.terms.panel = job.panelView;
        job.terms.contribution = job.contribution;
        job.terms.contributionBlockColumns = m_blockColumns;
        job.terms.rightHandColumns = job.columnsView;
        job.terms.rightHandRows = job.rowsView;
        for (const std::int32_t child : childSupernodes)
        {
            const auto rows = m_symbolic.rows(child);
            if (previous != nullptr && child == previous->s)
            {
                job.pendingChild = job.children.size();
                job.children.push_back(
                    {previous->updateValues, static_cast<std::int32_t>(rows.size()), {}, rows, m_blockColumns});
                job.childRightHands.push_back({previous->rightHandValues, rows});
                continue;
            }
            const Update& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))];
            job.children.push_back({pending.matrix.data(), pending.order, {}, rows, m_blockColumns});
            job.childRightHands.push_back({pending.rightHands.data(), rows});
        }
        job.assembler = std::make_unique<FrontAssembler>(
            layout, nullptr, job.children, FrontStorage::split(job.panelView, job.contribution, m_blockColumns));
        job.trailingFirst = TaskedFront::trailingBlockFirsts(r, m_blockColumns);
        const std::size_t trailingBlocks = job.trailingFirst.size() - 1;

        // The ranges of columns: the panel's, then each block's of the update matrix (positions).
        struct Range
        {
            std::size_t first = 0;
            std::size_t end = 0;
            std::size_t block = 0; // 0 the panel, 1 + j block j of the update matrix
        };
        std::vector<Range> ranges;
        const auto cut = [&](std::size_t first, std::size_t end, std::size_t block)
        {
            std::size_t rangeFirst = first;
            std::size_t held = 0;
            for (std::size_t p = first; p < end; ++p)
            {
                held += order - p;
                if (held >= ASSEMBLY_RANGE_VALUES || p + 1 == end)
                {
                    ranges.push_back({rangeFirst, p + 1, block});
                    rangeFirst = p + 1;
                    held = 0;
                }
            }
        };
        cut(0, c, 0);
        for (std::size_t j = 0; j < trailingBlocks; ++j)
            cut(c + job.trailingFirst[j], c + job.trailingFirst[j + 1], 1 + j);

        // The clearing, the source's terms, the children by ranges, the right-hand sides.
        std::vector<TaskId> cleared;
        for (const Range& range : ranges)
            cleared.push_back(flow.add(
                [&job, range]
                {
                    job.assembler->assembleColumns(range.first, range.end, AssemblyPhase::ClearOnly);
                    return true;
                },
                std::span<const TaskId>(), base + ASSEMBLY_PRIORITY));
        const TaskId ownTerms = flow.add(
            [this, &job]
            {
                clearRightHandSides(job.columnsView, job.rowsView);
                m_source.addTerms(job.terms);
                return true;
            },
            cleared, base + ASSEMBLY_PRIORITY);
        std::vector<TaskId> panelAdded;
        std::vector<std::vector<TaskId>> blockAdded(trailingBlocks);
        for (const Range& range : ranges)
        {
            std::vector<TaskId> before{ownTerms};
            if (job.pendingChild != TopFront::NO_PENDING_CHILD)
            {
                // The pending child's columns that land in the range, final once their blocks are.
                const auto [begin, stop] = job.assembler->childColumnsIn(job.pendingChild, range.first, range.end);
                for (std::size_t b = 0; begin < stop && b + 1 < previous->trailingFirst.size(); ++b)
                    if (previous->trailingFirst[b] < stop && previous->trailingFirst[b + 1] > begin)
                        before.push_back(previous->trailingDone[b]);
            }
            const TaskId added = flow.add(
                [&job, range]
                {
                    job.assembler->assembleColumns(range.first, range.end, AssemblyPhase::ChildrenOnly);
                    return true;
                },
                before, base + ASSEMBLY_PRIORITY);
            (range.block == 0 ? panelAdded : blockAdded[range.block - 1]).push_back(added);
        }
        std::vector<TaskId> rightHandsBefore{ownTerms};
        if (job.pendingChild != TopFront::NO_PENDING_CHILD)
            rightHandsBefore.push_back(previous->forwardDone);
        const TaskId rightHands = flow.add(
            [&job]
            {
                addChildRightHandSides(job.layout, job.childRightHands, job.columnsView, job.rowsView);
                return true;
            },
            rightHandsBefore, base + ASSEMBLY_PRIORITY);

        // The front's elimination: its tasks, or one task for a small front.
        std::vector<TaskId> blockReady;
        for (std::size_t j = 0; j < trailingBlocks; ++j)
            blockReady.push_back(flow.add(always, blockAdded[j], base + ASSEMBLY_PRIORITY));
        const std::size_t packingSlot = i % static_cast<std::size_t>(PACKING_BUFFERS);
        WorkBuffer& packing = m_packing[packingSlot];
        const bool isTasked =
            isTaskedFamily && TaskedFront::isWorthTasks(c, r, m_kernels.options().taskedFrontMinimumMultiplyAdds) &&
            static_cast<std::size_t>(packing.capacity()) >= TaskedFront::packingValues(c, r, m_blockColumns);
        if (isTasked)
        {
            job.tasked = std::make_unique<TaskedFront>(
                m_kernels.kind(), job.panelView, job.contribution, m_blockColumns,
                std::span<double>(packing.data(), static_cast<std::size_t>(packing.capacity())), &m_kernelWorkspace);
            OOS_ASSERT(job.tasked->trailingBlockCount() == trailingBlocks);
            job.tasked->addTasks(flow, panelAdded, blockReady, base);
            job.panelDone = job.tasked->panelDone();
            job.done = job.tasked->done();
            for (std::size_t j = 0; j < trailingBlocks; ++j)
                job.trailingDone.push_back(job.tasked->trailingBlockDone(j));
        }
        else
        {
            std::vector<TaskId> before = panelAdded;
            before.insert(before.end(), blockReady.begin(), blockReady.end());
            job.done = flow.add(
                [&job, &serial, this]
                {
                    job.eliminated = serial.factorFrontWithoutPivoting(job.panelView, job.contribution, m_blockColumns)
                                         .eliminatedCount;
                    return true;
                },
                before, base + TaskedFront::PANEL_PRIORITY);
            job.panelDone = job.done;
            job.trailingDone.assign(trailingBlocks, job.done);
        }

        // The ratio test, in the supernodes' order.
        std::vector<TaskId> ratioBefore{job.panelDone};
        if (previous != nullptr)
            ratioBefore.push_back(previous->ratioDone);
        job.ratioDone = flow.add(
            [this, &job, &ratios]
            {
                if (job.isComplete() && !m_hasRunAll)
                    testRatios(job.layout, job.panelView, ratios);
                return true;
            },
            ratioBefore, base + TaskedFront::PANEL_PRIORITY);

        // The forward sweep of the right-hand sides: y_c = L_cc⁻¹·b_c, then b_r ← b_r − L_rc·y_c by pieces of rows.
        const TaskId columnsSolved = flow.add(
            [&job, &serial, k, c]
            {
                if (job.isComplete() && k > 0 && c > 0)
                    serial.solveUnitLower(job.panelView.block(0, 0, c, c), job.columnsView);
                return true;
            },
            {job.panelDone, rightHands}, base + TaskedFront::PANEL_PRIORITY);
        std::vector<TaskId> rowsSolved{columnsSolved};
        for (std::size_t first = 0; first < r && k > 0; first += FORWARD_ROWS)
        {
            const std::size_t count = std::min(FORWARD_ROWS, r - first);
            rowsSolved.push_back(flow.add(
                [&job, &serial, first, count, c]
                {
                    if (job.isComplete() && c > 0)
                        serial.subtractProduct(job.rowsView.block(first, 0, count, job.rowsView.columns),
                                               job.panelView.block(c + first, 0, count, c), job.columnsView);
                    return true;
                },
                {columnsSolved}, base + TaskedFront::PANEL_PRIORITY));
        }
        job.forwardDone = flow.add(always, rowsSolved, base + TaskedFront::PANEL_PRIORITY);

        // The children consumed (their update matrices and right-hand sides released, as runTop), then the hand-over.
        std::vector<TaskId> releaseBefore = panelAdded;
        for (const std::vector<TaskId>& added : blockAdded)
            releaseBefore.insert(releaseBefore.end(), added.begin(), added.end());
        releaseBefore.push_back(rightHands);
        if (previous != nullptr)
            releaseBefore.push_back(previous->finished);
        job.released = flow.add(
            [this, childSupernodes]
            {
                for (const std::int32_t child : childSupernodes)
                    m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))] = {};
                return true;
            },
            releaseBefore, base + ASSEMBLY_PRIORITY);
        job.finished = flow.add(
            [this, &job, at, r]
            {
                job.assembler.reset();
                if (!job.isComplete())
                {
                    m_status[at] = Status::ZeroPivot;
                    m_zeroPivotAt[at] = job.layout.firstColumn + static_cast<std::int32_t>(job.eliminatedCount());
                    job.update = {};
                }
                else
                {
                    m_status[at] = Status::Done;
                    job.update.order = static_cast<std::int32_t>(r);
                    if (isKept(job.s))
                    {
                        Kept& kept = m_kept[static_cast<std::size_t>(m_keptSlot[at])];
                        kept.panel = std::move(job.panel);
                        kept.forward = std::move(job.forward);
                    }
                    m_pending[static_cast<std::size_t>(m_schedule.slotOf(job.s))] = std::move(job.update);
                }
                job.panel.release();
                job.forward.release();
                if (m_options.progress != nullptr && !m_hasRunAll)
                    m_options.progress->advance(static_cast<std::uint64_t>(job.layout.columnCount));
                return true;
            },
            {job.done, job.forwardDone, job.ratioDone, job.released}, base + TaskedFront::PANEL_PRIORITY);
        if (i + 1 < jobs.size())
            addStart(i + 1);
        return true;
    };
    addStart = [&](std::size_t next)
    {
        // One supernode of lookahead: the next starts once this one's children are consumed and its panel eliminated
        // (kept: its panel's memory is the plan's anyway; otherwise once it is finished), the one before it finished.
        std::vector<TaskId> before;
        if (next > 0)
        {
            const TopFront& current = jobs[next - 1];
            before.push_back(current.released);
            before.push_back(current.isAssembled && isKept(current.s) ? current.panelDone : current.finished);
            if (next > 1)
                before.push_back(jobs[next - 2].finished);
        }
        const std::int64_t base = -static_cast<std::int64_t>(next) * SUPERNODE_PRIORITY_STRIDE;
        flow.add([&start, next] { return start(next); }, before, base + ASSEMBLY_PRIORITY + 1);
    };
    if (!tops.empty())
        addStart(0);
    flow.run(m_kernels.options().pool, m_options.cancellation);
    if (!m_hasRunAll)
    {
        m_taskModel.topWorkSeconds = flow.workSeconds();
        m_taskModel.topCriticalPathSeconds = flow.criticalPathSeconds();
    }
}

void FrontStream::modelTasks(std::span<const double> subtreeSeconds)
{
    TaskModel& model = m_taskModel;
    model.measuredThreads = threadCountOf(m_kernels);
    model.subtreeWorkSeconds = 0.0;
    model.largestSubtreeSeconds = 0.0;
    for (const double seconds : subtreeSeconds)
    {
        model.subtreeWorkSeconds += seconds;
        model.largestSubtreeSeconds = std::max(model.largestSubtreeSeconds, seconds);
    }
    model.lowerSeconds.clear();
    model.upperSeconds.clear();
    for (const std::size_t threads : TaskModel::MODEL_THREADS)
    {
        // The subtrees in the run's order, each to the thread free first (the pool's batch: list scheduling).
        std::vector<double> busyUntil(threads, 0.0);
        for (const double seconds : subtreeSeconds)
        {
            const auto first = std::min_element(busyUntil.begin(), busyUntil.end());
            *first += seconds;
        }
        const double subtrees = *std::max_element(busyUntil.begin(), busyUntil.end());
        const double work = model.topWorkSeconds / static_cast<double>(threads);
        model.lowerSeconds.push_back(subtrees + std::max(work, model.topCriticalPathSeconds));
        model.upperSeconds.push_back(subtrees + work + model.topCriticalPathSeconds);
    }
}

void FrontStream::runWindow(std::int32_t first, std::int32_t last, std::int32_t keptFirst, std::int32_t keptLast)
{
    const bool isWhole = first == 0 && last == m_count - 1;
    const std::vector<FactorSubtree>& cut = m_schedule.subtrees();
    bool isSerial = cut.size() < 2 || m_order == FrontStreamOrder::Postorder;
    for (const FactorSubtree& subtree : cut)
        isSerial = isSerial || (subtree.first <= first && last <= subtree.last && !isWhole);
    m_keptFirst = keptFirst;
    m_keptLast = keptLast;
    for (std::int32_t s = first; s <= last; ++s)
        m_status[static_cast<std::size_t>(s)] = Status::Pending;
    std::vector<RatioTest> ratios; // per task: the subtrees of the cut in their order, then the top (or one, serially)
    try
    {
        if (isSerial)
        {
            ratios.resize(1);
            runSubtree(first, last, false, false, ratios.front());
        }
        else
        {
            std::vector<const FactorSubtree*> inWindow; // bounded by the cut
            for (const FactorSubtree& subtree : cut)
                if (first <= subtree.first && subtree.last <= last)
                    inWindow.push_back(&subtree);
            ratios.resize(inWindow.size() + 1);
            // PERFORMANCE: the subtrees the most work first (FactorSchedule::subtreesByWork), each its ratio test in
            // the cut's order. Measured on a barrel model of four frames, six threads.
            std::vector<std::size_t> byWork(inWindow.size());
            for (std::size_t t = 0; t < byWork.size(); ++t)
                byWork[t] = t;
            std::stable_sort(byWork.begin(), byWork.end(),
                             [&](std::size_t a, std::size_t b) { return inWindow[a]->flops > inWindow[b]->flops; });
            base::TaskGraph graph;
            std::vector<double> subtreeSeconds(inWindow.size()); // per subtree, in the order of byWork
            for (std::size_t order = 0; order < byWork.size(); ++order)
            {
                const FactorSubtree* subtree = inWindow[byWork[order]];
                RatioTest* test = &ratios[byWork[order]];
                double* seconds = &subtreeSeconds[order];
                graph.add(
                    [this, subtree, test, seconds]
                    {
                        base::Stopwatch watch;
                        runSubtree(subtree->first, subtree->last, true, true, *test);
                        *seconds = watch.lap();
                    });
            }
            base::Stopwatch phases;
            graph.run(m_kernels.options().pool, m_options.cancellation);
            const double subtreePhase = phases.lap();
            std::vector<std::int32_t> tops; // bounded by the top part
            for (const std::int32_t s : m_schedule.topSupernodes())
                if (first <= s && s <= last)
                    tops.push_back(s);
            if (threadCountOf(m_kernels) >= 2)
                runTopFlow(tops, ratios.back());
            else
                for (const std::int32_t s : tops)
                    runTop(s, ratios.back());
            const double topPhase = phases.lap();
            if (isWhole && !m_hasRunAll)
            {
                m_taskModel.subtreePhaseSeconds = subtreePhase;
                m_taskModel.topSeconds = topPhase;
                if (threadCountOf(m_kernels) < 2)
                    m_taskModel.topWorkSeconds = m_taskModel.topCriticalPathSeconds = topPhase;
                modelTasks(subtreeSeconds);
            }
        }
        throwIfSingular(first, last);
    }
    catch (...)
    {
        for (Update& pending : m_pending)
            pending = {};
        releaseKept();
        throw;
    }
    for (Update& pending : m_pending)
        pending = {};
    m_workspaceUsage.peakValues = std::max(m_workspaceUsage.peakValues, m_meter.peak());
    m_workspaceUsage.heldPeakValues = std::max(m_workspaceUsage.heldPeakValues, m_pool->heldPeak());
    if (isWhole && !m_hasRunAll)
        finishRatioTest(ratios);
}

void FrontStream::releaseSlots(std::int32_t first, std::int32_t last) noexcept
{
    for (std::int32_t s = first; s <= last; ++s)
    {
        std::int32_t& slot = m_keptSlot[static_cast<std::size_t>(s)];
        if (slot >= 0)
            m_kept[static_cast<std::size_t>(slot)] = {};
        slot = -1;
    }
}

void FrontStream::releaseKept() noexcept
{
    m_kept.clear();
    std::fill(m_keptSlot.span().begin(), m_keptSlot.span().end(), -1);
    m_keptFirst = 0;
    m_keptLast = -1;
}

void FrontStream::throwIfSingular(std::int32_t first, std::int32_t last) const
{
    // The zero pivots in the order of the supernodes (MultifrontalDriver::finish): the factor is unusable.
    std::vector<base::GridComponent> zeroPivotDofs;
    std::string zeroPivotNames;
    std::size_t zeroPivotCount = 0;
    for (std::int32_t s = first; s <= last; ++s)
    {
        if (m_status[static_cast<std::size_t>(s)] != Status::ZeroPivot)
            continue;
        const std::int32_t equation = m_zeroPivotAt[static_cast<std::size_t>(s)];
        ++zeroPivotCount;
        if (zeroPivotDofs.size() < PivotDiagnostics::RECORDED_ISSUE_COUNT)
            zeroPivotDofs.push_back(gridComponentOf(m_options.equationNamer, equation));
        if (zeroPivotCount <= PivotDiagnostics::LISTED_ISSUE_COUNT)
            zeroPivotNames.append("  ").append(describeEquation(m_options.equationNamer, equation)).append("\n");
    }
    if (zeroPivotCount == 0)
        return;
    if (zeroPivotCount > PivotDiagnostics::LISTED_ISSUE_COUNT)
        zeroPivotNames.append("  and ")
            .append(std::to_string(zeroPivotCount - PivotDiagnostics::LISTED_ISSUE_COUNT))
            .append(" more\n");
    throw base::SingularityError(
        base::concatenate(
            {"the stiffness matrix is singular: ", std::to_string(zeroPivotCount),
             " zero (or not finite) pivots met in the factorisation, a mechanism or an unconstrained "
             "DOF at:\n",
             zeroPivotNames,
             "constrain these DOFs (single-point constraints) or connect them to the structure; with a matrix "
             "that is meant to be indefinite, factorise it in indefinite mode"}),
        std::move(zeroPivotDofs));
}

void FrontStream::finishRatioTest(std::vector<RatioTest>& tests)
{
    // The tasks' issues merged in elimination order (the equations increase with the supernodes); the largest ratio
    // the first of the largest in that order, as one pass over the supernodes finds it.
    std::vector<PivotIssue> issues;
    std::int64_t issueCount = 0;
    double largestRatio = 0.0;
    std::int32_t equationOfLargest = -1;
    for (RatioTest& test : tests)
    {
        issueCount += test.issueCount;
        issues.insert(issues.end(), test.issues.begin(), test.issues.end());
        if (test.equationOfLargest < 0)
            continue;
        if (equationOfLargest < 0 || test.largestRatio > largestRatio ||
            (test.largestRatio == largestRatio && test.equationOfLargest < equationOfLargest))
        {
            largestRatio = test.largestRatio;
            equationOfLargest = test.equationOfLargest;
        }
    }
    std::sort(issues.begin(), issues.end(),
              [](const PivotIssue& a, const PivotIssue& b) { return a.equation < b.equation; });
    if (issues.size() > PivotDiagnostics::RECORDED_ISSUE_COUNT)
        issues.resize(PivotDiagnostics::RECORDED_ISSUE_COUNT);
    m_diagnostics = PivotDiagnostics(std::move(issues), issueCount, largestRatio, equationOfLargest,
                                     m_options.pivotPolicy.maxRatio);
    m_hasRunAll = true;
    m_diagnostics.enforce(m_options.pivotPolicy, m_options.equationNamer);
}

void FrontStream::solveBackward(std::int32_t s, SolutionSink& sink)
{
    if (m_options.cancellation != nullptr)
        m_options.cancellation->throwIfCancelled();
    const auto at = static_cast<std::size_t>(s);
    const auto c = static_cast<std::size_t>(m_symbolic.columnCount(s));
    const std::span<const std::int32_t> rows = m_symbolic.rows(s);
    const std::size_t r = rows.size();
    const std::size_t k = m_columnCount;
    OOS_ASSERT(m_keptSlot[at] >= 0);
    Kept& kept = m_kept[static_cast<std::size_t>(m_keptSlot[at])];
    const DenseMatrixView panel{kept.panel.data(), c + r, c, std::max<std::size_t>(c + r, 1)};

    // x_r: the rows of the ancestors' solutions (each ancestor's columns are consecutive rows of R(s)).
    base::AlignedBuffer<double> gathered(r * k, base::FOR_OVERWRITE);
    const DenseMatrixView rowValues{gathered.data(), r, k, std::max<std::size_t>(r, 1)};
    for (std::size_t i = 0; i < r;)
    {
        const std::int32_t owner = supernodeOf(rows[i]);
        const std::int32_t ownerFirst = m_symbolic.firstColumn(owner);
        const auto ownerColumns = static_cast<std::size_t>(m_symbolic.columnCount(owner));
        const base::AlignedBuffer<double>& solution = m_solutions[static_cast<std::size_t>(owner)];
        OOS_ASSERT(solution.size() == ownerColumns * k);
        std::size_t end = i;
        while (end < r && rows[end] < ownerFirst + static_cast<std::int32_t>(ownerColumns))
            ++end;
        for (std::size_t j = 0; j < k; ++j)
            for (std::size_t row = i; row < end; ++row)
                rowValues(row, j) = solution[static_cast<std::size_t>(rows[row] - ownerFirst) + j * ownerColumns];
        i = end;
    }

    // x_c = L_ccᵀ⁻¹·(D_c⁻¹·y_c − L_rcᵀ·x_r) (FactorSolve's diagonal step: no −0).
    base::AlignedBuffer<double> solution(c * k, base::FOR_OVERWRITE);
    const DenseMatrixView columnValues{solution.data(), c, k, std::max<std::size_t>(c, 1)};
    for (std::size_t j = 0; j < k; ++j)
        for (std::size_t e = 0; e < c; ++e)
            columnValues(e, j) = sweeps::canonical(kept.forward.data()[e + j * c] / panel(e, e));
    if (k > 0 && c > 0)
    {
        if (r > 0)
            m_kernels.subtractTransposedProduct(columnValues, panel.block(c, 0, r, c), rowValues);
        m_kernels.solveUnitLowerTransposed(panel.block(0, 0, c, c), columnValues);
    }
    m_liveSolutionValues += static_cast<std::int64_t>(c * k);
    m_solutionPeakValues = std::max(m_solutionPeakValues, m_liveSolutionValues + static_cast<std::int64_t>(r * k));
    m_solutions[at] = std::move(solution);
    sink.take(s, columnValues, rowValues);
    m_isSolved[at] = 1;
    if (m_pendingChildren[at] == 0)
        completeSubtree(s);
}

void FrontStream::completeSubtree(std::int32_t s) noexcept
{
    for (std::int32_t done = s; done != Supernodes::ROOT;)
    {
        const auto at = static_cast<std::size_t>(done);
        m_liveSolutionValues -= static_cast<std::int64_t>(m_solutions[at].size());
        m_solutions[at] = {};
        m_isSolved[at] = 2;
        const std::int32_t parent = m_symbolic.supernodes().parent(done);
        if (parent == Supernodes::ROOT)
            return;
        const auto parentAt = static_cast<std::size_t>(parent);
        --m_pendingChildren[parentAt];
        done = m_isSolved[parentAt] == 1 && m_pendingChildren[parentAt] == 0 ? parent : Supernodes::ROOT;
    }
}

RegionSweepCost FrontStream::solve(SolutionSink& sink, std::int64_t regionValues, std::int64_t retainedValues)
{
    RegionSweepCost cost;
    cost.regionValues = regionValues;
    if (m_count == 0)
        return cost;
    const std::size_t threads = threadCountOf(m_kernels);
    const FrontStreamEstimate planned =
        estimate(m_symbolic, threads, m_columnCount, regionValues, m_order, m_blockColumns, retainedValues);
    const SupernodeValues values{m_symbolic, static_cast<std::int64_t>(m_columnCount), m_blockColumns};
    const auto keptOf = [&](std::int32_t s) { return values.kept(s); };
    const auto subtreeFirstOf = [&](std::int32_t s) { return m_subtreeFirst[static_cast<std::size_t>(s)]; };
    const auto total = static_cast<std::size_t>(m_count);
    base::BudgetedBuffer<double> flopsBefore(m_budget, total + 1, "front stream: flops");
    base::BudgetedBuffer<double> workBefore(m_budget, total + 1, "front stream: work");
    for (std::int32_t s = 0; s < m_count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        flopsBefore[at + 1] =
            flopsBefore[at] + FactorSchedule::supernodeFlops(m_symbolic.columnCount(s), m_symbolic.rowCount(s));
        workBefore[at + 1] = workBefore[at] + runWork(m_symbolic.columnCount(s), m_symbolic.rowCount(s),
                                                      static_cast<std::int64_t>(m_columnCount));
    }
    // The retained regions (estimate()'s plan, made the same way) before the reservations, so that its candidates come
    // and go within the arrays' share.
    const std::int32_t top = regions::topFirst(m_count, regionValues, keptOf);
    const RetentionPlan retention =
        planRetention(top, regionValues, retainedValues, values, subtreeFirstOf, flopsBefore.span(), workBefore.span(),
                      m_isRetainedRoot.span(), m_budget);
    OOS_ASSERT(retention.values == planned.retainedValues);

    // The working storage, the kept supernodes and the retained regions come from the pool (its capacity reserved);
    // the solutions and the rows gathered are reserved apart; the arrays are budgeted where they are made.
    const std::int64_t poolValues = planned.workspaceValues + planned.keptValues + planned.retainedValues;
    const base::MemoryReservation poolReservation =
        m_budget.reserve(static_cast<std::uint64_t>(poolValues) * VALUE_BYTES,
                         "front stream: fronts, update matrices, right-hand sides, panels kept and retained");
    const base::MemoryReservation solutionReservation =
        m_budget.reserve(static_cast<std::uint64_t>(planned.solutionValues) * VALUE_BYTES,
                         "front stream: solutions of the downward pass");
    for (WorkBuffer& buffer : m_packing)
        buffer.release(); // a solve that threw left them in its pool
    m_pool = std::make_unique<WorkspacePool>(poolValues);
    m_workspaceUsage.boundValues = poolValues;
    // The packed panels' buffers of the fronts made by tasks (estimate's packingValues): planned, from the pool.
    for (WorkBuffer& buffer : m_packing)
    {
        const std::int64_t values = planned.packingValues / PACKING_BUFFERS;
        buffer = WorkBuffer(values, "front stream: packed panels of the fronts' tasks", &m_meter, m_pool.get());
        if (values > 0)
            buffer.ensure(values, m_budget);
    }
    for (std::int32_t s = 0; s < m_count; ++s)
    {
        m_isSolved[static_cast<std::size_t>(s)] = 0;
        m_pendingChildren[static_cast<std::size_t>(s)] = 0;
    }
    for (std::int32_t s = 0; s < m_count; ++s)
    {
        const std::int32_t parent = m_symbolic.supernodes().parent(s);
        if (parent != Supernodes::ROOT)
            ++m_pendingChildren[static_cast<std::size_t>(parent)];
    }
    m_liveSolutionValues = 0;

    // The kept slots: the retained regions' supernodes first (held to their visit), then the shared ones of the top
    // and, after it, of each region's run.
    std::int32_t retainedSlots = 0;
    std::int32_t largestRun = m_count - top;
    regions::forEach(top, regionValues, keptOf, subtreeFirstOf,
                     [&](std::int32_t kept, std::int32_t root, std::int64_t)
                     {
                         if (m_isRetainedRoot[static_cast<std::size_t>(root)] == 0)
                             largestRun = std::max(largestRun, root - kept + 1);
                         else
                             for (std::int32_t s = kept; s <= root; ++s)
                                 m_keptSlot[static_cast<std::size_t>(s)] = retainedSlots++;
                     });
    m_kept.clear();
    m_kept.resize(static_cast<std::size_t>(retainedSlots) + static_cast<std::size_t>(largestRun));

    // The upward pass, the top and the retained regions kept; then the top solved from the root down.
    for (std::int32_t s = top; s < m_count; ++s)
    {
        m_keptSlot[static_cast<std::size_t>(s)] = retainedSlots + (s - top);
        cost.topValues += keptOf(s);
    }
    cost.keptValues = cost.topValues;
    base::Stopwatch watch;
    runWindow(0, m_count - 1, 0, m_count - 1);
    cost.upwardSeconds = watch.lap();
    cost.taskModel = m_taskModel;
    for (std::int32_t s = m_count - 1; s >= top; --s)
        solveBackward(s, sink);
    releaseSlots(top, m_count - 1);
    cost.downwardSeconds += watch.lap();

    // The regions below the top, each solved from its root down: a retained one from the panels the upward pass kept,
    // another factorised again with its loads (its run keeps only its own supernodes: those of retained regions below
    // it keep the upward pass's).
    regions::forEach(top, regionValues, keptOf, subtreeFirstOf,
                     [&](std::int32_t kept, std::int32_t root, std::int64_t keptValues)
                     {
                         const std::int32_t first = m_subtreeFirst[static_cast<std::size_t>(root)];
                         const double flops = flopsBefore[static_cast<std::size_t>(root) + 1] -
                                              flopsBefore[static_cast<std::size_t>(first)];
                         const double work = workBefore[static_cast<std::size_t>(root) + 1] -
                                             workBefore[static_cast<std::size_t>(first)];
                         const bool isRetained = m_isRetainedRoot[static_cast<std::size_t>(root)] != 0;
                         if (!isRetained)
                         {
                             for (std::int32_t s = kept; s <= root; ++s)
                                 m_keptSlot[static_cast<std::size_t>(s)] = retainedSlots + (s - kept);
                             runWindow(first, root, kept, root);
                         }
                         cost.regionSeconds += watch.lap();
                         for (std::int32_t s = root; s >= kept; --s)
                             solveBackward(s, sink);
                         releaseSlots(kept, root);
                         cost.downwardSeconds += watch.lap();
                         if (isRetained)
                         {
                             ++cost.retainedRunCount;
                             cost.retainedValues += keptValues;
                             cost.retainedFlops += flops;
                             cost.retainedWork += work;
                             return;
                         }
                         ++cost.regionRunCount;
                         cost.keptValues = std::max(cost.keptValues, keptValues);
                         cost.recomputedFlops += flops;
                         cost.recomputedWork += work;
                     });
    releaseKept();
    for (WorkBuffer& buffer : m_packing)
        buffer.release();
    OOS_ASSERT(m_liveSolutionValues == 0);
    OOS_ASSERT(m_solutionPeakValues <= planned.solutionValues);
    m_pool.reset();
    return cost;
}

} // namespace oos::linalg
