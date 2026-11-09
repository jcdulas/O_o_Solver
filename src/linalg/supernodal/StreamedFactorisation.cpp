#include "StreamedFactorisation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/threading/TaskGraph.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{
namespace
{

constexpr std::uint64_t VALUE_BYTES = sizeof(double);
constexpr std::int32_t NO_UNIT = -1;

std::size_t toSize(std::int64_t value) noexcept
{
    return static_cast<std::size_t>(value);
}

std::size_t threadCountOf(const DenseKernels& kernels) noexcept
{
    return kernels.options().pool == nullptr ? 1 : kernels.options().pool->threadCount();
}

// The caller's kernel options with the factorisation's cancellation token and kernel workspace (MultifrontalDriver).
// With the factorisation's budget for the packed panels of the fronts made by tasks (TaskedFront: reserved
// when a front starts, the recursion when they do not fit).
DenseKernelOptions withFactorisation(DenseKernelOptions options, const base::CancellationToken* cancellation,
                                     KernelWorkspace& workspace, base::MemoryBudget& budget)
{
    if (options.budget == nullptr)
        options.budget = &budget;
    if (options.cancellation == nullptr)
        options.cancellation = cancellation;
    if (options.workspace == nullptr)
        options.workspace = &workspace;
    return options;
}

// The diagonal of column j of the matrix (0 when it is not stored): the a_jj of the MAXRATIO ratio.
double matrixDiagonal(const SymmetricCsc& matrix, std::int32_t column) noexcept
{
    const auto begin = toSize(matrix.columnOffsets()[static_cast<std::size_t>(column)]);
    const auto end = toSize(matrix.columnOffsets()[static_cast<std::size_t>(column) + 1]);
    if (begin < end && matrix.rowIndices()[begin] == column)
        return matrix.values()[begin];
    return 0.0;
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

} // namespace

StreamedFactorisation::StreamedFactorisation(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                             const DenseKernels& kernels, const FactorOptions& options,
                                             bool excludesTrailing, base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_matrix(matrix),
      m_kernels(kernels.kind(), withFactorisation(kernels.options(), options.cancellation, m_kernelWorkspace, budget)),
      m_options(options),
      m_budget(budget),
      m_schedule(symbolic, excludesTrailing, threadCountOf(kernels), PivotingMode::PositiveDefinite, budget,
                 FactorSchedule::TopOrder::Serial),
      m_plannedConcurrency(threadCountOf(kernels))
{
    if (options.pivoting != PivotingMode::PositiveDefinite)
        throw std::invalid_argument(
            "StreamedFactorisation: only a positive definite matrix is factorised without its "
            "factor (no pivot is delayed); pass PivotingMode::PositiveDefinite");
    if (matrix.dimension() != symbolic.dofCount())
        throw std::invalid_argument(base::concatenate(
            {"StreamedFactorisation: the matrix has ", std::to_string(matrix.dimension()),
             " equations and the symbolic factor ", std::to_string(symbolic.dofCount()),
             "; factorise the matrix assembled on the graph and in the order of its symbolic analysis"}));
    // The factor's arrays, as MultifrontalDriver::allocateFactor makes them, but no panel array: the rows are the
    // symbolic ones (positive definite: no delayed pivot).
    const std::int32_t trailing = m_schedule.excludedTrailing();
    const std::int32_t count = trailing >= 0 ? symbolic.supernodeCount() - 1 : symbolic.supernodeCount();
    const auto total = static_cast<std::size_t>(count);
    SupernodalFactor& factor = m_factor;
    factor.m_dimension = symbolic.dofCount();
    factor.m_eliminatedCount = trailing >= 0 ? symbolic.dofCount() - symbolic.trailingDofCount() : symbolic.dofCount();
    factor.m_supernodeCount = count;
    factor.m_pivoting = PivotingMode::PositiveDefinite;
    factor.m_kernel = m_kernels.kind();
    factor.m_panelOffset = base::BudgetedBuffer<std::int64_t>(budget, total + 1, "streamed factor: panel offsets");
    factor.m_rowOffset = base::BudgetedBuffer<std::int64_t>(budget, total + 1, "streamed factor: row offsets");
    factor.m_firstColumn = base::BudgetedBuffer<std::int32_t>(budget, total, "streamed factor: first columns");
    factor.m_eliminated = base::BudgetedBuffer<std::int32_t>(budget, total, "streamed factor: eliminated columns");
    factor.m_rowCount = base::BudgetedBuffer<std::int32_t>(budget, total, "streamed factor: panel rows");
    factor.m_panelPointer = base::BudgetedBuffer<double*>(budget, total, "streamed factor: resident panels");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        const std::int64_t c = symbolic.columnCount(s);
        const std::int64_t r = symbolic.rowCount(s);
        factor.m_panelOffset[at + 1] = factor.m_panelOffset[at] + c * (c + r);
        factor.m_rowOffset[at + 1] = factor.m_rowOffset[at] + c + r;
        factor.m_firstColumn[at] = symbolic.firstColumn(s);
        factor.m_eliminated[at] = static_cast<std::int32_t>(c);
        factor.m_rowCount[at] = static_cast<std::int32_t>(c + r);
        factor.m_largestPanelRowCount = std::max(factor.m_largestPanelRowCount, static_cast<std::int32_t>(c + r));
    }
    factor.m_panelRows = base::BudgetedBuffer<std::int32_t>(budget, toSize(factor.m_rowOffset[total]),
                                                            "streamed factor: rows", base::FOR_OVERWRITE);
    for (std::int32_t s = 0; s < count; ++s)
    {
        std::int32_t* rows = factor.m_panelRows.data() + factor.m_rowOffset[static_cast<std::size_t>(s)];
        for (std::int32_t k = 0; k < symbolic.columnCount(s); ++k)
            *rows++ = symbolic.firstColumn(s) + k;
        for (const std::int32_t row : symbolic.rows(s))
            *rows++ = row;
    }
    for (const FactorSubtree& subtree : m_schedule.subtrees())
        factor.m_subtrees.push_back({subtree.first, subtree.last});
    // The plan of the sweeps from the rows alone (SolvePlan reads no value): the parent of a supernode below an
    // excluded trailing one is none, as MultifrontalDriver::finish gives it.
    std::vector<std::int32_t> parents(total); // one per supernode
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t parent = symbolic.supernodes().parent(s);
        parents[static_cast<std::size_t>(s)] =
            parent == Supernodes::ROOT || parent >= count ? SolvePlan::NO_PARENT : parent;
    }
    factor.m_solvePlan = std::make_shared<const SolvePlan>(factor, parents, budget);

    m_panelTableReservation = budget.reserve(total * sizeof(WorkBuffer), "streamed factorisation: panel table");
    m_panels.resize(total);
    m_status = base::BudgetedBuffer<Status>(budget, total, "streamed factorisation: supernode status");
    m_zeroPivotAt = base::BudgetedBuffer<std::int32_t>(budget, total, "streamed factorisation: zero pivots");
    m_unitEndingAt = base::BudgetedBuffer<std::int32_t>(budget, total, "streamed factorisation: unit ends");
    for (std::size_t s = 0; s < total; ++s)
        m_unitEndingAt[s] = NO_UNIT;
    const std::span<const SolvePlan::Unit> units = plan().units();
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        m_unitEndingAt[static_cast<std::size_t>(units[u].lastSupernode)] = static_cast<std::int32_t>(u);
        std::int64_t values = 0;
        for (std::int32_t s = units[u].firstSupernode; s <= units[u].lastSupernode; ++s)
            values += WorkspacePool::classSize(panelValueCount(s));
        if (plan().isByTree() && units[u].subtree == SolvePlan::NO_SUBTREE)
            m_largestTopValues = std::max(m_largestTopValues, values);
        else
            m_largestSubtreeUnitValues = std::max(m_largestSubtreeUnitValues, values);
    }
    m_isTop = base::BudgetedBuffer<std::uint8_t>(budget, total, "streamed factorisation: top supernodes");
    for (const std::int32_t s : m_schedule.topSupernodes())
        if (s < count)
            m_isTop[static_cast<std::size_t>(s)] = 1;
    m_subtreeFirst = base::BudgetedBuffer<std::int32_t>(budget, total, "streamed factorisation: subtree firsts");
    for (std::int32_t s = 0; s < count; ++s)
        m_subtreeFirst[static_cast<std::size_t>(s)] = s;
    for (std::int32_t s = 0; s < count; ++s)
    {
        // Postorder: a child before its parent, so the first of a child is final when its parent takes it.
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent != Supernodes::ROOT && parent < count)
            m_subtreeFirst[static_cast<std::size_t>(parent)] =
                std::min(m_subtreeFirst[static_cast<std::size_t>(parent)], m_subtreeFirst[static_cast<std::size_t>(s)]);
    }
    m_pivots = base::BudgetedBuffer<double>(budget, toSize(factor.m_eliminatedCount), "streamed factorisation: pivots");
    const auto slots = static_cast<std::size_t>(m_schedule.slotCount());
    m_pendingReservation =
        budget.reserve(slots * sizeof(PendingUpdate), "streamed factorisation: pending update matrices");
    m_pending.resize(slots);
}

StreamedFactorisation::~StreamedFactorisation()
{
    releaseKept();
    m_pending.clear();
}

std::int64_t StreamedFactorisation::panelValueCount(std::int32_t s) const noexcept
{
    const std::int64_t c = m_symbolic.columnCount(s);
    return c * (c + m_symbolic.rowCount(s));
}

double StreamedFactorisation::supernodeFlops(std::int32_t s) const noexcept
{
    return FactorSchedule::supernodeFlops(m_symbolic.columnCount(s), m_symbolic.rowCount(s));
}

base::MemoryReservation StreamedFactorisation::reserveRun(std::int64_t panelValues)
{
    OOS_ASSERT(m_pool == nullptr);
    const std::int64_t capacity = m_schedule.workspaceValues() + panelValues;
    base::MemoryReservation reservation = m_budget.reserve(static_cast<std::uint64_t>(capacity) * VALUE_BYTES,
                                                           "streamed factorisation: fronts, update matrices, panels");
    m_pool = std::make_unique<WorkspacePool>(capacity);
    return reservation;
}

WorkBuffer StreamedFactorisation::eliminate(std::int32_t s, std::span<const UpdateMatrix> children, bool isPlanned,
                                            const DenseKernels& kernels, base::ThreadPool* assemblyPool,
                                            std::int32_t& updateOrder)
{
    const auto at = static_cast<std::size_t>(s);
    FrontLayout layout;
    layout.supernode = s;
    layout.firstColumn = m_symbolic.firstColumn(s);
    layout.columnCount = m_symbolic.columnCount(s);
    layout.rows = m_symbolic.rows(s);
    const auto columns = static_cast<std::size_t>(layout.columnCount);
    const std::size_t order = columns + layout.rows.size();
    const std::size_t rowCount = layout.rows.size();
    // The front in place, as MultifrontalDriver assembles a positive definite one: its panel (here a block of its own,
    // lent to the factor) and its trailing block, the update matrix.
    const auto panelValues = static_cast<std::int64_t>(columns * order);
    m_panels[at] = WorkBuffer(isPlanned ? WorkspacePool::classSize(panelValues) : 0, "streamed factorisation: panel",
                              &m_meter, m_pool.get());
    m_panels[at].ensure(panelValues, m_budget);
    m_factor.m_panelPointer[at] = m_panels[at].data();
    const auto updateValues = static_cast<std::int64_t>(rowCount * rowCount);
    WorkBuffer update(isPlanned ? WorkspacePool::classSize(updateValues) : 0, "factorisation: update matrix", &m_meter,
                      m_pool.get());
    update.ensure(updateValues, m_budget);
    const DenseMatrixView panel{m_panels[at].data(), order, columns, std::max<std::size_t>(order, 1)};
    const DenseMatrixView contribution{update.data(), rowCount, rowCount, std::max<std::size_t>(rowCount, 1)};
    assembleFront(layout, m_matrix, children, FrontStorage::split(panel, contribution), assemblyPool,
                  m_options.cancellation);
    const LdltOutcome outcome = kernels.factorFrontWithoutPivoting(panel, contribution);
    if (!outcome.isComplete)
    {
        m_status[at] = Status::ZeroPivot;
        m_zeroPivotAt[at] = layout.firstColumn + static_cast<std::int32_t>(outcome.eliminatedCount);
        releasePanel(s);
        return {};
    }
    for (std::size_t k = 0; k < columns; ++k)
        m_pivots[static_cast<std::size_t>(layout.firstColumn) + k] = panel(k, k);
    m_status[at] = Status::Done;
    updateOrder = static_cast<std::int32_t>(rowCount);
    return update;
}

void StreamedFactorisation::releasePanel(std::int32_t s) noexcept
{
    const auto at = static_cast<std::size_t>(s);
    m_panels[at].release();
    m_factor.m_panelPointer[at] = nullptr;
}

void StreamedFactorisation::finishUnit(std::size_t unit, UnitConsumer* consumer, std::span<const std::uint8_t> isKept)
{
    const SolvePlan::Unit& planned = plan().units()[unit];
    bool isComplete = true;
    for (std::int32_t s = planned.firstSupernode; s <= planned.lastSupernode; ++s)
        isComplete = isComplete && m_status[static_cast<std::size_t>(s)] == Status::Done;
    if (consumer != nullptr && isComplete)
        consumer->consumeUnit(unit);
    for (std::int32_t s = planned.firstSupernode; s <= planned.lastSupernode; ++s)
        if (isKept.empty() || isKept[static_cast<std::size_t>(s)] == 0)
            releasePanel(s);
}

void StreamedFactorisation::runSubtree(std::int32_t first, std::int32_t last, bool isParallel, bool handsToSlots,
                                       UnitConsumer* consumer, std::span<const std::uint8_t> isKept)
{
    // In a task of the subtrees: the kernels on this thread only (the subtrees are the parallel pieces), the working
    // storage covered by the plan unless the task runs beyond the planned concurrency (MultifrontalDriver). Serially:
    // the kernels' threads.
    std::unique_ptr<RunningSubtree> running;
    if (isParallel)
        running = std::make_unique<RunningSubtree>(m_runningSubtrees);
    const bool isPlanned = !isParallel || running->countAtStart() <= m_plannedConcurrency;
    DenseKernelOptions serialOptions = m_kernels.options();
    serialOptions.pool = nullptr;
    const DenseKernels serial(m_kernels.kind(), serialOptions);
    const DenseKernels& kernels = isParallel ? serial : m_kernels;
    std::vector<StackEntry> entries; // bounded by the depth of the range
    std::vector<UpdateMatrix> children;
    for (std::int32_t s = first; s <= last; ++s)
    {
        if (m_options.cancellation != nullptr)
            m_options.cancellation->throwIfCancelled();
        const auto at = static_cast<std::size_t>(s);
        const auto childSupernodes = m_schedule.children(s);
        OOS_ASSERT(entries.size() >= childSupernodes.size());
        const std::size_t firstChildEntry = entries.size() - childSupernodes.size();
        bool areChildrenDone = true;
        for (const std::int32_t child : childSupernodes)
            areChildrenDone = areChildrenDone && m_status[static_cast<std::size_t>(child)] == Status::Done;
        StackEntry entry;
        if (areChildrenDone)
        {
            children.clear();
            for (std::size_t k = firstChildEntry; k < entries.size(); ++k)
                children.push_back({entries[k].values.data(),
                                    entries[k].order,
                                    {},
                                    m_symbolic.rows(childSupernodes[k - firstChildEntry])});
            entry.values = eliminate(s, children, isPlanned, kernels, nullptr, entry.order);
        }
        else
            m_status[at] = Status::Skipped;
        entries.resize(firstChildEntry);
        // An update matrix whose parent is beyond the range: a subtree root's goes to its pending slot (its parent is
        // a top supernode, or the trailing one), the others are dropped (a region's forest).
        const std::int32_t parent = m_symbolic.supernodes().parent(s);
        if (parent == Supernodes::ROOT || parent > last)
        {
            const std::int32_t slot = m_schedule.slotOf(s);
            if (handsToSlots && slot != FactorSchedule::NO_SLOT && entry.order > 0)
            {
                PendingUpdate& pending = m_pending[static_cast<std::size_t>(slot)];
                pending.values = std::move(entry.values);
                pending.order = entry.order;
            }
        }
        else
            entries.push_back(std::move(entry));
        if (m_options.progress != nullptr && !m_hasRunAll)
            m_options.progress->advance(static_cast<std::uint64_t>(m_symbolic.columnCount(s)));
        // With a consumer the panels wait for the end of their unit; without one they go at once unless kept.
        const std::int32_t unit = m_unitEndingAt[at];
        if (consumer != nullptr)
        {
            if (unit != NO_UNIT)
                finishUnit(static_cast<std::size_t>(unit), consumer, isKept);
        }
        else if (isKept.empty() || isKept[at] == 0)
            releasePanel(s);
    }
    OOS_ASSERT(entries.empty());
}

void StreamedFactorisation::runTop(std::int32_t s, UnitConsumer* consumer, std::span<const std::uint8_t> isKept)
{
    if (m_options.cancellation != nullptr)
        m_options.cancellation->throwIfCancelled();
    const auto at = static_cast<std::size_t>(s);
    const auto childSupernodes = m_schedule.children(s);
    PendingUpdate& own = m_pending[static_cast<std::size_t>(m_schedule.slotOf(s))];
    bool areChildrenDone = true;
    for (const std::int32_t child : childSupernodes)
        areChildrenDone = areChildrenDone && m_status[static_cast<std::size_t>(child)] == Status::Done;
    if (!areChildrenDone)
    {
        m_status[at] = Status::Skipped;
        for (const std::int32_t child : childSupernodes)
            m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))].values.release();
        return;
    }
    std::vector<UpdateMatrix> children;
    for (const std::int32_t child : childSupernodes)
    {
        const PendingUpdate& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))];
        children.push_back({pending.values.data(), pending.order, {}, m_symbolic.rows(child)});
    }
    std::int32_t order = 0;
    WorkBuffer update = eliminate(s, children, true, m_kernels, m_kernels.options().pool, order);
    for (const std::int32_t child : childSupernodes)
        m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))].values.release();
    if (m_status[at] == Status::Done)
    {
        own.values = std::move(update);
        own.order = order;
    }
    if (m_options.progress != nullptr && !m_hasRunAll)
        m_options.progress->advance(static_cast<std::uint64_t>(m_symbolic.columnCount(s)));
    const std::int32_t unit = m_unitEndingAt[at];
    if (consumer != nullptr && unit != NO_UNIT)
        finishUnit(static_cast<std::size_t>(unit), consumer, isKept);
    else if (isKept.empty() || isKept[at] == 0)
        releasePanel(s);
}

void StreamedFactorisation::assembleTrailing(DenseMatrix& schurComplement)
{
    const std::int32_t trailing = m_schedule.excludedTrailing();
    OOS_ASSERT(trailing >= 0);
    const auto childSupernodes = m_schedule.children(trailing);
    std::vector<UpdateMatrix> children;
    for (const std::int32_t child : childSupernodes)
    {
        const PendingUpdate& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))];
        children.push_back({pending.values.data(), pending.order, {}, m_symbolic.rows(child)});
    }
    FrontLayout layout;
    layout.supernode = trailing;
    layout.firstColumn = m_symbolic.firstColumn(trailing);
    layout.columnCount = m_symbolic.columnCount(trailing);
    layout.rows = m_symbolic.rows(trailing);
    OOS_ASSERT(layout.rows.empty());
    const DenseMatrixView front = schurComplement.view();
    assembleFront(layout, m_matrix, children, FrontStorage::whole(front, front.rows));
    for (const std::int32_t child : childSupernodes)
        m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))].values.release();
    // Both triangles: the Schur complement is handed over as a full symmetric matrix (MultifrontalDriver).
    for (std::size_t j = 0; j < front.columns; ++j)
        for (std::size_t i = j + 1; i < front.rows; ++i)
            front(j, i) = front(i, j);
}

void StreamedFactorisation::run(UnitConsumer* consumer, std::span<const std::uint8_t> isKept,
                                DenseMatrix* schurComplement)
{
    if (schurComplement != nullptr && m_schedule.excludedTrailing() < 0)
        throw std::invalid_argument(
            "StreamedFactorisation::run: K* is assembled by a partial factorisation; make it "
            "with excludesTrailing");
    if (supernodeCount() == 0)
    {
        if (schurComplement != nullptr)
            assembleTrailing(*schurComplement);
        if (!m_hasRunAll)
            finishFirstRun();
        return;
    }
    runWindow(0, supernodeCount() - 1, consumer, isKept, schurComplement);
}

void StreamedFactorisation::runSubtreeOf(std::int32_t root, std::span<const std::uint8_t> isKept)
{
    if (root < 0 || root >= supernodeCount())
        throw std::invalid_argument(
            base::concatenate({"StreamedFactorisation::runSubtreeOf: expected one of the ",
                               std::to_string(supernodeCount()), " supernodes, found ", std::to_string(root)}));
    runWindow(subtreeFirst(root), root, nullptr, isKept, nullptr);
}

void StreamedFactorisation::runWindow(std::int32_t first, std::int32_t last, UnitConsumer* consumer,
                                      std::span<const std::uint8_t> isKept, DenseMatrix* schurComplement)
{
    const bool isWhole = first == 0 && last == supernodeCount() - 1;
    // Serially when the plan is not by tree or the window lies inside one subtree of the cut.
    const std::vector<FactorSubtree>& cut = m_schedule.subtrees();
    bool isSerial = !plan().isByTree();
    for (const FactorSubtree& subtree : cut)
        isSerial = isSerial || (subtree.first <= first && last <= subtree.last && !isWhole);
    // The panels kept, and the largest panel made in a subtree task and on the calling thread.
    std::int64_t keptValues = 0;
    std::int64_t largestInSubtrees = 0;
    std::int64_t largestOnCaller = 0;
    for (std::int32_t s = first; s <= last; ++s)
    {
        const std::int64_t values = WorkspacePool::classSize(panelValueCount(s));
        const bool isInTask = !isSerial && m_isTop[static_cast<std::size_t>(s)] == 0;
        std::int64_t& largest = isInTask ? largestInSubtrees : largestOnCaller;
        largest = std::max(largest, values);
        if (!isKept.empty() && isKept[static_cast<std::size_t>(s)] != 0)
            keptValues += values;
    }
    // The panels resident at once beside those kept: a unit of each running subtree or one top unit (a run that hands
    // its units over); one panel of each running subtree or one on the calling thread (a region).
    const auto concurrency = static_cast<std::int64_t>(isSerial ? 1 : m_plannedConcurrency);
    const std::int64_t resident = consumer != nullptr
                                      ? std::max(concurrency * m_largestSubtreeUnitValues, m_largestTopValues)
                                      : std::max(concurrency * largestInSubtrees, largestOnCaller);
    base::MemoryReservation reservation = reserveRun(resident + keptValues);
    for (std::int32_t s = first; s <= last; ++s)
        m_status[static_cast<std::size_t>(s)] = Status::Pending;
    try
    {
        if (isSerial)
            runSubtree(first, last, false, schurComplement != nullptr, consumer, isKept);
        else
        {
            base::TaskGraph graph;
            for (const FactorSubtree& subtree : cut)
                if (first <= subtree.first && subtree.last <= last)
                    graph.add([this, &subtree, consumer, isKept]
                              { runSubtree(subtree.first, subtree.last, true, true, consumer, isKept); });
            graph.run(m_kernels.options().pool, m_options.cancellation);
            if (consumer != nullptr)
                consumer->afterSubtrees();
            for (const std::int32_t s : m_schedule.topSupernodes())
                if (s != m_schedule.excludedTrailing() && first <= s && s <= last)
                    runTop(s, consumer, isKept);
        }
        throwIfSingular();
        if (schurComplement != nullptr)
            assembleTrailing(*schurComplement);
    }
    catch (...)
    {
        for (PendingUpdate& pending : m_pending)
            pending.values.release();
        for (std::int32_t s = first; s <= last; ++s)
            releasePanel(s);
        m_pool.reset();
        throw;
    }
    for (PendingUpdate& pending : m_pending)
        pending.values.release();
    m_workspaceUsage.boundValues = std::max(m_workspaceUsage.boundValues, m_schedule.workspaceValues() + resident);
    m_workspaceUsage.peakValues = std::max(m_workspaceUsage.peakValues, m_meter.peak());
    m_workspaceUsage.heldPeakValues = std::max(m_workspaceUsage.heldPeakValues, m_pool->heldPeak());
    if (isWhole && !m_hasRunAll)
        finishFirstRun();
    if (keptValues == 0)
    {
        m_pool.reset();
        return;
    }
    m_keptReservation = std::move(reservation);
}

void StreamedFactorisation::releaseKept() noexcept
{
    for (std::int32_t s = 0; s < supernodeCount(); ++s)
        if (m_factor.m_panelPointer[static_cast<std::size_t>(s)] != nullptr)
            releasePanel(s);
    m_pool.reset();
    m_keptReservation.release();
}

void StreamedFactorisation::throwIfSingular() const
{
    // The zero pivots in the order of the supernodes (MultifrontalDriver::finish): the factor is unusable.
    std::vector<base::GridComponent> zeroPivotDofs;
    std::string zeroPivotNames;
    std::size_t zeroPivotCount = 0;
    for (std::int32_t s = 0; s < supernodeCount(); ++s)
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

void StreamedFactorisation::finishFirstRun()
{
    // The ratio test of MultifrontalDriver::finish in positive definite mode, panel by panel in elimination order, on
    // the pivots kept (equation firstColumn + k is the k-th pivot of a supernode).
    std::vector<PivotIssue> issues; // at most RECORDED_ISSUE_COUNT, all counted in issueCount
    std::int64_t issueCount = 0;
    double largestRatio = 0.0;
    std::int32_t equationOfLargest = -1;
    const double maxRatio = m_options.pivotPolicy.maxRatio;
    for (std::int32_t s = 0; s < supernodeCount(); ++s)
        for (std::int32_t k = 0; k < m_symbolic.columnCount(s); ++k)
        {
            const std::int32_t equation = m_symbolic.firstColumn(s) + k;
            const double pivot = m_pivots[static_cast<std::size_t>(equation)];
            const double diagonal = matrixDiagonal(m_matrix, equation);
            const double ratio = diagonal / pivot;
            if (equationOfLargest < 0 || ratio > largestRatio)
            {
                largestRatio = ratio;
                equationOfLargest = equation;
            }
            if (ratio > maxRatio || ratio < 0.0)
            {
                ++issueCount;
                if (issues.size() < PivotDiagnostics::RECORDED_ISSUE_COUNT)
                    issues.push_back({equation, diagonal, pivot, ratio,
                                      ratio < 0.0 ? PivotIssueKind::NegativePivot : PivotIssueKind::RatioAboveMaximum});
            }
        }
    m_diagnostics = PivotDiagnostics(std::move(issues), issueCount, largestRatio, equationOfLargest, maxRatio);
    m_hasRunAll = true;
    m_diagnostics.enforce(m_options.pivotPolicy, m_options.equationNamer);
}

} // namespace oos::linalg
