#include "MultifrontalDriver.hpp"

#include "SolvePlan.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/threading/TaskGraph.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

constexpr std::size_t VALUE_BYTES = sizeof(double);

std::size_t toSize(std::int64_t value) noexcept
{
    return static_cast<std::size_t>(value);
}

std::int64_t squared(std::int64_t value) noexcept
{
    return value * value;
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

// Copies the update matrix left in the trailing block of an eliminated front (from position `first`) into `target`
// (order × order, leading dimension order, lower triangle).
void copyUpdate(DenseMatrixView front, std::size_t first, double* target)
{
    const std::size_t order = front.rows - first;
    for (std::size_t j = 0; j < order; ++j)
        std::copy_n(&front(first + j, first + j), order - j, target + j + j * order);
}

// Copies the first `eliminated` columns of an eliminated front (order rows, leading dimension order) into a panel and
// zeroes their entries above the diagonal: the front's upper triangle is never assembled (it holds what a reused
// buffer held before), and a panel is the same bits whichever buffer its front was in.
void storePanel(DenseMatrixView front, std::size_t eliminated, double* panel)
{
    const std::size_t order = front.rows;
    std::copy_n(front.data, order * eliminated, panel);
    for (std::size_t j = 1; j < eliminated; ++j)
        std::fill_n(panel + j * order, j, 0.0);
}

// The caller's kernel options with the factorisation's cancellation token and kernel workspace (unless the options
// carry their own already).
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

std::size_t threadCountOf(const DenseKernels& kernels) noexcept
{
    return kernels.options().pool == nullptr ? 1 : kernels.options().pool->threadCount();
}

// Counts a running subtree task for as long as it runs.
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

    // The number of subtree tasks running when this one started, itself included.
    [[nodiscard]] std::size_t countAtStart() const noexcept { return m_count; }

private:
    std::atomic<std::size_t>& m_running;
    std::size_t m_count;
};

} // namespace

MultifrontalDriver::MultifrontalDriver(const SymbolicFactor& symbolic, FrontSource& source, const DenseKernels& kernels,
                                       const FactorOptions& options, base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_source(&source),
      m_kernels(kernels.kind(), withFactorisation(kernels.options(), options.cancellation, m_kernelWorkspace, budget)),
      m_options(options),
      m_budget(budget),
      m_schedule(symbolic, false, threadCountOf(kernels), options.pivoting, budget),
      m_plannedConcurrency(threadCountOf(kernels)),
      m_pool(m_schedule.workspaceValues())
{
    if (options.pivoting != PivotingMode::Indefinite)
        throw std::invalid_argument(
            "supernodal factorisation: a front source is factorised in indefinite mode only (its positive definite "
            "factorisation is StreamedSolve's); give PivotingMode::Indefinite");
    const auto count = static_cast<std::size_t>(symbolic.supernodeCount());
    m_status = base::BudgetedBuffer<Status>(budget, count, "factorisation: supernode status");
    m_zeroPivotAt = base::BudgetedBuffer<std::int32_t>(budget, count, "factorisation: zero pivots");
    m_delayedOut = base::BudgetedBuffer<std::int32_t>(budget, count, "factorisation: delayed pivots");
    const auto slots = static_cast<std::size_t>(m_schedule.slotCount());
    m_pendingReservation = budget.reserve(static_cast<std::uint64_t>(slots) * sizeof(PendingUpdate),
                                          "factorisation: pending update matrices");
    m_pending.resize(slots);
}

MultifrontalDriver::MultifrontalDriver(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                       const DenseKernels& kernels, const FactorOptions& options, bool excludesTrailing,
                                       base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_matrix(&matrix),
      m_kernels(kernels.kind(), withFactorisation(kernels.options(), options.cancellation, m_kernelWorkspace, budget)),
      m_options(options),
      m_budget(budget),
      m_schedule(symbolic, excludesTrailing, threadCountOf(kernels), options.pivoting, budget),
      m_plannedConcurrency(threadCountOf(kernels)),
      m_pool(m_schedule.workspaceValues())
{
    if (matrix.dimension() != symbolic.dofCount())
        throw std::invalid_argument(base::concatenate(
            {"supernodal factorisation: the matrix has ", std::to_string(matrix.dimension()),
             " equations and the symbolic factor ", std::to_string(symbolic.dofCount()),
             "; factorise the matrix assembled on the graph and in the order of its symbolic analysis"}));
    const auto count = static_cast<std::size_t>(symbolic.supernodeCount());
    m_status = base::BudgetedBuffer<Status>(budget, count, "factorisation: supernode status");
    m_zeroPivotAt = base::BudgetedBuffer<std::int32_t>(budget, count, "factorisation: zero pivots");
    m_delayedOut = base::BudgetedBuffer<std::int32_t>(budget, count, "factorisation: delayed pivots");
    const auto slots = static_cast<std::size_t>(m_schedule.slotCount());
    m_pendingReservation = budget.reserve(static_cast<std::uint64_t>(slots) * sizeof(PendingUpdate),
                                          "factorisation: pending update matrices");
    m_pending.resize(slots);
}

void MultifrontalDriver::allocateFactor(SupernodalFactor& factor)
{
    const std::int32_t trailing = m_schedule.excludedTrailing();
    const std::int32_t count = trailing >= 0 ? m_symbolic.supernodeCount() - 1 : m_symbolic.supernodeCount();
    const auto total = static_cast<std::size_t>(count);
    factor.m_dimension = m_symbolic.dofCount();
    factor.m_eliminatedCount =
        trailing >= 0 ? m_symbolic.dofCount() - m_symbolic.trailingDofCount() : m_symbolic.dofCount();
    factor.m_supernodeCount = count;
    factor.m_pivoting = m_options.pivoting;
    factor.m_kernel = m_kernels.kind();

    if (m_spool != nullptr)
    {
        m_inertiaOf = base::BudgetedBuffer<Inertia>(m_budget, total, "factorisation out of core: inertia");
        m_twoByTwoOf = base::BudgetedBuffer<std::int64_t>(m_budget, total, "factorisation out of core: 2x2 pivots");
    }
    factor.m_panelOffset = base::BudgetedBuffer<std::int64_t>(m_budget, total + 1, "factor: panel offsets");
    factor.m_rowOffset = base::BudgetedBuffer<std::int64_t>(m_budget, total + 1, "factor: row offsets");
    factor.m_firstColumn = base::BudgetedBuffer<std::int32_t>(m_budget, total, "factor: first columns");
    factor.m_eliminated = base::BudgetedBuffer<std::int32_t>(m_budget, total, "factor: eliminated columns");
    factor.m_rowCount = base::BudgetedBuffer<std::int32_t>(m_budget, total, "factor: panel rows");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        const std::int64_t c = m_symbolic.columnCount(s);
        const std::int64_t r = m_symbolic.rowCount(s);
        factor.m_panelOffset[at + 1] = factor.m_panelOffset[at] + c * (c + r);
        factor.m_rowOffset[at + 1] = factor.m_rowOffset[at] + c + r;
        factor.m_firstColumn[at] = m_symbolic.firstColumn(s);
        factor.m_eliminated[at] = static_cast<std::int32_t>(c);
        factor.m_rowCount[at] = static_cast<std::int32_t>(c + r);
    }
    // Every panel value and row that is read is written by eliminate() first (a panel's c·(c + r) values and its rows
    // in positive definite mode; its eliminated columns and its rows in indefinite mode, the only ones a FactorPanel
    // shows), so neither is cleared: each page is first written by the task that assembles its front (a subtree's
    // thread, the pool's threads for a large top front's ranges), never by a serial fill here.
    const std::size_t valueCount = toSize(factor.m_panelOffset[total]);
    const std::size_t rowCount = toSize(factor.m_rowOffset[total]);
    FactorStorage* storage = m_options.storage;
    if (m_spool != nullptr)
    {
        // Out of core: the rows in memory, the values on the spool; a panel is read through a pointer while it is
        // resident (OutOfCoreFactor), null otherwise.
        factor.m_panelRows =
            base::BudgetedBuffer<std::int32_t>(m_budget, rowCount, "factor: rows", base::FOR_OVERWRITE);
        factor.m_panelPointer = base::BudgetedBuffer<double*>(m_budget, total, "factor out of core: resident panels");
    }
    else if (storage != nullptr && storage->m_values.size() == valueCount && storage->m_rows.size() == rowCount)
    {
        // PERFORMANCE: the panels of an earlier factor of this structure: their pages are mapped,
        // so the assembly writes them without a page fault per 4 kB.
        // Measured (three factorisations, one thread): a later factorisation of the
        // shell of 100 000 DOFs 0.46 → 0.39-0.42 s, its page faults 124 000 → 51 000-64 000.
        factor.m_panelValues = base::BudgetedBuffer<double>(m_budget, std::move(storage->m_values), "factor: panels");
        factor.m_panelRows = base::BudgetedBuffer<std::int32_t>(m_budget, std::move(storage->m_rows), "factor: rows");
    }
    else
    {
        factor.m_panelValues =
            base::BudgetedBuffer<double>(m_budget, valueCount, "factor: panels", base::FOR_OVERWRITE);
        factor.m_panelRows =
            base::BudgetedBuffer<std::int32_t>(m_budget, rowCount, "factor: rows", base::FOR_OVERWRITE);
    }
    factor.m_allocatedPanelValueCount = m_spool != nullptr ? 0 : factor.m_panelOffset[total];
    if (m_options.pivoting == PivotingMode::Indefinite)
    {
        factor.m_subdiagonal =
            base::BudgetedBuffer<double>(m_budget, toSize(factor.m_dimension), "factor: 2x2 pivot subdiagonal");
        factor.m_grownTableReservation =
            m_budget.reserve(static_cast<std::uint64_t>(total) * sizeof(void*), "factor: grown panel table");
        factor.m_grownPanels.resize(total);
    }
}

FrontLayout MultifrontalDriver::layoutOf(std::int32_t supernode, std::span<const UpdateMatrix> children) const
{
    FrontLayout layout;
    layout.supernode = supernode;
    layout.firstColumn = m_symbolic.firstColumn(supernode);
    layout.columnCount = m_symbolic.columnCount(supernode);
    for (const UpdateMatrix& child : children)
        layout.delayedEquations.insert(layout.delayedEquations.end(), child.delayedEquations.begin(),
                                       child.delayedEquations.end());
    layout.rows = m_symbolic.rows(supernode);
    return layout;
}

bool MultifrontalDriver::areChildrenDone(std::int32_t supernode) const
{
    const auto children = m_schedule.children(supernode);
    return std::all_of(children.begin(), children.end(),
                       [&](std::int32_t child) { return m_status[static_cast<std::size_t>(child)] == Status::Done; });
}

MultifrontalDriver::AssembledFront MultifrontalDriver::assemble(const FrontLayout& layout,
                                                                std::span<const UpdateMatrix> children,
                                                                SupernodalFactor& factor, WorkBuffer& front,
                                                                bool isPlanned, base::ThreadPool* pool)
{
    const auto at = static_cast<std::size_t>(layout.supernode);
    const auto order = static_cast<std::size_t>(layout.order());
    AssembledFront assembled;
    if (m_options.pivoting == PivotingMode::PositiveDefinite)
    {
        // PERFORMANCE: the front in place. Its panel (the c columns over the c + r rows, leading dimension c + r) is
        // the factor's panel of s, and its trailing block (r × r) is the update matrix it hands to its parent: no
        // front buffer, no copy of the panel nor of the update matrix (the elimination gives the bits it gives on the
        // whole front, DenseKernels::factorFrontWithoutPivoting).
        // Measured (one thread, with the extend-add by runs): the shell of
        // 100 000 DOFs 0.52 -> 0.50 s, the 3D grid of 50 000 DOFs 1.64 -> 1.45 s.
        const auto columns = static_cast<std::size_t>(layout.columnCount);
        const std::size_t rowCount = order - columns;
        const auto updateValues = static_cast<std::int64_t>(rowCount * rowCount);
        assembled.update = WorkBuffer(isPlanned ? WorkspacePool::classSize(updateValues) : 0,
                                      "factorisation: update matrix", &m_meter, &m_pool);
        assembled.update.ensure(updateValues, m_budget);
        const DenseMatrixView panel{factor.m_panelValues.data() + factor.m_panelOffset[at], order, columns,
                                    std::max<std::size_t>(order, 1)};
        const DenseMatrixView contribution{assembled.update.data(), rowCount, rowCount,
                                           std::max<std::size_t>(rowCount, 1)};
        assembled.storage = FrontStorage::split(panel, contribution);
    }
    else
    {
        front.ensure(squared(static_cast<std::int64_t>(order)), m_budget);
        const DenseMatrixView whole{front.data(), order, order, std::max<std::size_t>(order, 1)};
        assembled.storage = FrontStorage::whole(whole, static_cast<std::size_t>(layout.candidateCount()));
    }
    if (m_source == nullptr)
    {
        assembleFront(layout, *m_matrix, children, assembled.storage, pool, m_options.cancellation);
        return assembled;
    }
    // A front source's terms on the front's columns and rows (its delayed columns hold none: an item placed at s has
    // its first equation among s's columns, which follow every descendant's), before the children's: the order of a
    // matrix's entries (assembleFront).
    FrontTerms terms;
    terms.supernode = layout.supernode;
    terms.firstColumn = layout.firstColumn;
    terms.columnCount = layout.columnCount;
    terms.delayedCount = layout.delayedCount();
    terms.rows = layout.rows;
    terms.panel = assembled.storage.panel;
    terms.contribution = assembled.storage.contribution;
    terms.contributionBlockColumns = assembled.storage.contributionBlockColumns;
    terms.rightHandColumns = DenseMatrixView{nullptr, static_cast<std::size_t>(layout.columnCount), 0, 1};
    terms.rightHandRows = DenseMatrixView{nullptr, layout.rows.size(), 0, 1};
    assembleFrontInPhases(
        layout, children, assembled.storage, [&] { m_source->addTerms(terms); }, pool, m_options.cancellation);
    return assembled;
}

MultifrontalDriver::EliminationResult MultifrontalDriver::eliminate(const FrontLayout& layout, AssembledFront& front,
                                                                    const DenseKernels& kernels,
                                                                    SupernodalFactor& factor, bool isPlanned,
                                                                    ScratchSpool::Writer* writer)
{
    if (m_options.pivoting == PivotingMode::Indefinite)
    {
        const DenseMatrixView& panel = front.storage.panel;
        return eliminatePivoted(layout, DenseMatrixView{panel.data, panel.rows, panel.rows, panel.leadingDimension},
                                kernels, factor, isPlanned, writer);
    }
    const auto at = static_cast<std::size_t>(layout.supernode);
    const auto columns = static_cast<std::size_t>(layout.columnCount);
    EliminationResult result;
    const LdltOutcome outcome = kernels.factorFrontWithoutPivoting(front.storage.panel, front.storage.contribution);
    if (!outcome.isComplete)
    {
        m_status[at] = Status::ZeroPivot;
        m_zeroPivotAt[at] = layout.firstColumn + static_cast<std::int32_t>(outcome.eliminatedCount);
        return result;
    }
    std::int32_t* rows = factor.m_panelRows.data() + factor.m_rowOffset[at];
    for (std::size_t k = 0; k < columns; ++k)
        rows[k] = layout.firstColumn + static_cast<std::int32_t>(k);
    std::copy(layout.rows.begin(), layout.rows.end(), rows + columns);
    result.update = std::move(front.update);
    result.updateOrder = static_cast<std::int32_t>(layout.rows.size());
    m_status[at] = Status::Done;
    return result;
}

MultifrontalDriver::EliminationResult MultifrontalDriver::eliminatePivoted(const FrontLayout& layout,
                                                                           DenseMatrixView front,
                                                                           const DenseKernels& kernels,
                                                                           SupernodalFactor& factor, bool isPlanned,
                                                                           ScratchSpool::Writer* writer)
{
    const std::int32_t s = layout.supernode;
    const auto at = static_cast<std::size_t>(s);
    const auto order = static_cast<std::size_t>(layout.order());
    const auto candidates = static_cast<std::size_t>(layout.candidateCount());
    const auto columns = static_cast<std::size_t>(layout.columnCount);
    EliminationResult result;

    // Bunch-Kaufman among the candidates, the others delayed to the parent.
    const PivotedLdlt pivoting = kernels.factorWithPivoting(front, candidates);
    const std::size_t eliminated = pivoting.eliminatedCount;
    if (m_symbolic.supernodes().parent(s) == Supernodes::ROOT && eliminated < candidates)
        throw base::InternalError(
            base::concatenate({"supernodal factorisation: the root front ", std::to_string(s), " delayed ",
                               std::to_string(candidates - eliminated), " pivots, which a root front cannot do"}));
    // The equation of each front position before the interchanges.
    std::vector<std::int32_t> equations(order);
    for (std::size_t k = 0; k < columns; ++k)
        equations[k] = layout.firstColumn + static_cast<std::int32_t>(k);
    std::size_t next = columns;
    for (const std::int32_t equation : layout.delayedEquations)
        equations[next++] = equation;
    std::copy(layout.rows.begin(), layout.rows.end(), equations.begin() + static_cast<std::ptrdiff_t>(candidates));
    std::vector<std::int32_t> panelRows(order);
    for (std::size_t k = 0; k < candidates; ++k)
        panelRows[k] = equations[pivoting.permutation[k]];
    std::copy(equations.begin() + static_cast<std::ptrdiff_t>(candidates), equations.end(),
              panelRows.begin() + static_cast<std::ptrdiff_t>(candidates));

    if (writer != nullptr)
    {
        // Out of core: the panel as storePanel makes it, one record on the spool; its rows and subdiagonal
        // kept as an in-core panel's (the solve plan and the sweeps read them).
        std::vector<double> record(order * eliminated); // bounded by a front
        storePanel(front, eliminated, record.data());
        writer->append(record.data(), record.size() * sizeof(double), m_records[at]);
        m_inertiaOf[at] = pivoting.inertia;
        m_twoByTwoOf[at] = static_cast<std::int64_t>(pivoting.twoByTwoBlockCount);
        if (layout.delayedCount() == 0)
        {
            std::copy(panelRows.begin(), panelRows.end(), factor.m_panelRows.data() + factor.m_rowOffset[at]);
            std::copy_n(pivoting.subdiagonal.data(), eliminated,
                        factor.m_subdiagonal.data() + factor.m_firstColumn[at]);
        }
        else
        {
            auto grown = std::make_unique<SupernodalFactor::GrownPanel>();
            grown->rows = panelRows;
            grown->subdiagonal.assign(pivoting.subdiagonal.begin(),
                                      pivoting.subdiagonal.begin() + static_cast<std::ptrdiff_t>(eliminated));
            grown->eliminatedCount = static_cast<std::int32_t>(eliminated);
            factor.m_grownPanels[at] = std::move(grown);
        }
    }
    else if (layout.delayedCount() == 0)
    {
        // The panel fits the symbolic shape: e <= c columns over the c + r rows.
        storePanel(front, eliminated, factor.m_panelValues.data() + factor.m_panelOffset[at]);
        std::copy(panelRows.begin(), panelRows.end(), factor.m_panelRows.data() + factor.m_rowOffset[at]);
        std::copy_n(pivoting.subdiagonal.data(), eliminated, factor.m_subdiagonal.data() + factor.m_firstColumn[at]);
    }
    else
    {
        auto grown = std::make_unique<SupernodalFactor::GrownPanel>();
        grown->values = base::BudgetedBuffer<double>(m_budget, order * eliminated, "factor: panels grown by delays");
        storePanel(front, eliminated, grown->values.data());
        grown->rows = panelRows;
        grown->subdiagonal.assign(pivoting.subdiagonal.begin(),
                                  pivoting.subdiagonal.begin() + static_cast<std::ptrdiff_t>(eliminated));
        grown->eliminatedCount = static_cast<std::int32_t>(eliminated);
        factor.m_grownPanels[at] = std::move(grown);
    }
    factor.m_eliminated[at] = static_cast<std::int32_t>(eliminated);
    factor.m_rowCount[at] = static_cast<std::int32_t>(order);
    m_delayedOut[at] = static_cast<std::int32_t>(candidates - eliminated);
    result.updateOrder = static_cast<std::int32_t>(order - eliminated);
    result.delayedEquations.assign(panelRows.begin() + static_cast<std::ptrdiff_t>(eliminated),
                                   panelRows.begin() + static_cast<std::ptrdiff_t>(candidates));
    // The update matrix out of the front, into its own buffer (planned at the symbolic r²; delays grow it).
    result.update = WorkBuffer(isPlanned ? WorkspacePool::classSize(squared(m_symbolic.rowCount(s))) : 0,
                               "factorisation: update matrix", &m_meter, &m_pool);
    result.update.ensure(squared(result.updateOrder), m_budget);
    copyUpdate(front, eliminated, result.update.data());
    m_status[at] = Status::Done;
    return result;
}

void MultifrontalDriver::runSubtree(const FactorSubtree& subtree, SupernodalFactor& factor)
{
    // The bound of the working storage covers the fronts and stacks of m_plannedConcurrency subtrees at once; a subtree
    // that starts beyond them (a thread outside the pool helping this batch) reserves its own in the budget.
    const RunningSubtree running(m_runningSubtrees);
    const bool isBeyondPlan = running.countAtStart() > m_plannedConcurrency;
    if (isBeyondPlan)
        ++m_subtreesBeyondPlan;
    // The kernels on this thread only: the subtrees are the parallel pieces here (the bits are the same either way).
    DenseKernelOptions serialOptions = m_kernels.options();
    serialOptions.pool = nullptr;
    const DenseKernels kernels(m_kernels.kind(), serialOptions);
    // The front buffer of the indefinite mode, reused by the fronts of the subtree (a positive definite front needs
    // none: assemble()).
    WorkBuffer front(isBeyondPlan ? 0 : subtree.frontValues, "factorisation: front", &m_meter, &m_pool);
    // Out of core: the subtree's panels in blocks of its own (contiguous whatever the other subtrees write).
    std::optional<ScratchSpool::Writer> writer;
    if (m_spool != nullptr)
        writer.emplace(*m_spool);
    std::vector<StackEntry> entries; // bounded by the depth of the subtree
    std::vector<UpdateMatrix> children;

    for (std::int32_t s = subtree.first; s <= subtree.last; ++s)
    {
        if (m_options.cancellation != nullptr)
            m_options.cancellation->throwIfCancelled();
        const auto at = static_cast<std::size_t>(s);
        // The children of s are the top entries of the stack, in increasing order (postorder).
        const std::size_t childCount = m_schedule.children(s).size();
        OOS_ASSERT(entries.size() >= childCount);
        const std::size_t firstChildEntry = entries.size() - childCount;
        EliminationResult elimination;
        if (areChildrenDone(s))
        {
            children.clear();
            for (std::size_t k = firstChildEntry; k < entries.size(); ++k)
            {
                const std::int32_t child = m_schedule.children(s)[k - firstChildEntry];
                children.push_back(
                    {entries[k].values.data(), entries[k].order, entries[k].delayedEquations, m_symbolic.rows(child)});
            }
            const FrontLayout layout = layoutOf(s, children);
            AssembledFront assembled = assemble(layout, children, factor, front, !isBeyondPlan, nullptr);
            // The children's update matrices are summed into the front: their values go now (the layout keeps a
            // copy of their delayed equations).
            for (std::size_t k = firstChildEntry; k < entries.size(); ++k)
                entries[k].values.release();
            elimination = eliminate(layout, assembled, kernels, factor, !isBeyondPlan, writer ? &*writer : nullptr);
        }
        else
            m_status[at] = Status::Skipped;

        // Pop the children, push the update matrix of s (empty when it failed or was skipped).
        entries.resize(firstChildEntry);
        StackEntry entry;
        if (m_status[at] == Status::Done)
        {
            entry.values = std::move(elimination.update);
            entry.order = elimination.updateOrder;
            entry.delayedEquations = std::move(elimination.delayedEquations);
        }
        entries.push_back(std::move(entry));
        if (m_options.progress != nullptr)
            m_options.progress->advance(static_cast<std::uint64_t>(m_symbolic.columnCount(s)));
    }

    if (writer)
        writer->flush();
    // Hand the update matrix of the root to its parent's task.
    OOS_ASSERT(entries.size() == 1);
    PendingUpdate& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(subtree.last))];
    if (m_status[static_cast<std::size_t>(subtree.last)] == Status::Done && entries[0].order > 0)
    {
        pending.values = std::move(entries[0].values);
        pending.order = entries[0].order;
        pending.delayedEquations = std::move(entries[0].delayedEquations);
    }
}

void MultifrontalDriver::runTopSupernode(std::int32_t supernode, SupernodalFactor& factor)
{
    if (m_options.cancellation != nullptr)
        m_options.cancellation->throwIfCancelled();
    const auto at = static_cast<std::size_t>(supernode);
    const auto childSupernodes = m_schedule.children(supernode);
    PendingUpdate& own = m_pending[static_cast<std::size_t>(m_schedule.slotOf(supernode))];
    if (!areChildrenDone(supernode))
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
        children.push_back({pending.values.data(), pending.order, pending.delayedEquations, m_symbolic.rows(child)});
    }
    const FrontLayout layout = layoutOf(supernode, children);
    const std::int64_t plannedOrder = m_symbolic.columnCount(supernode) + m_symbolic.rowCount(supernode);
    // The indefinite mode's front (a positive definite front needs none).
    WorkBuffer front(WorkspacePool::classSize(squared(plannedOrder)), "factorisation: front", &m_meter, &m_pool);
    AssembledFront assembled = assemble(layout, children, factor, front, true, m_kernels.options().pool);
    for (const std::int32_t child : childSupernodes)
    {
        PendingUpdate& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))];
        pending.values.release();
        pending.delayedEquations.clear();
    }

    std::optional<ScratchSpool::Writer> writer;
    if (m_spool != nullptr)
        writer.emplace(*m_spool);
    EliminationResult elimination = eliminate(layout, assembled, m_kernels, factor, true, writer ? &*writer : nullptr);
    if (writer)
        writer->flush();
    if (m_status[at] == Status::Done)
    {
        own.values = std::move(elimination.update);
        own.order = elimination.updateOrder;
        own.delayedEquations = std::move(elimination.delayedEquations);
    }
    if (m_options.progress != nullptr)
        m_options.progress->advance(static_cast<std::uint64_t>(m_symbolic.columnCount(supernode)));
}

void MultifrontalDriver::assembleTrailing(std::int32_t supernode, DenseMatrix* schurComplement)
{
    const auto at = static_cast<std::size_t>(supernode);
    const auto childSupernodes = m_schedule.children(supernode);
    if (!areChildrenDone(supernode) || schurComplement == nullptr)
    {
        // Without a destination (an interior factor recomputed for recovery, K* kept) nothing is assembled.
        for (const std::int32_t child : childSupernodes)
            m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))].values.release();
        m_status[at] = schurComplement == nullptr && areChildrenDone(supernode) ? Status::Done : Status::Skipped;
        return;
    }
    std::vector<UpdateMatrix> children;
    for (const std::int32_t child : childSupernodes)
    {
        const PendingUpdate& pending = m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))];
        children.push_back({pending.values.data(), pending.order, pending.delayedEquations, m_symbolic.rows(child)});
    }
    const FrontLayout layout = layoutOf(supernode, children);
    // A partial factorisation is positive definite: nothing is delayed into the trailing set, whose equations are
    // the caller's boundary.
    OOS_ASSERT(layout.delayedCount() == 0 && layout.rows.empty());
    const DenseMatrixView front = schurComplement->view();
    assembleFront(layout, *m_matrix, children, FrontStorage::whole(front, front.rows));
    for (const std::int32_t child : childSupernodes)
        m_pending[static_cast<std::size_t>(m_schedule.slotOf(child))].values.release();
    // Both triangles: the Schur complement is handed over as a full symmetric matrix.
    for (std::size_t j = 0; j < front.columns; ++j)
        for (std::size_t i = j + 1; i < front.rows; ++i)
            front(j, i) = front(i, j);
    m_status[at] = Status::Done;
}

void MultifrontalDriver::run(SupernodalFactor& factor, DenseMatrix* schurComplement)
{
    m_schurComplement = schurComplement;
    allocateFactor(factor);
    // The working storage, reserved before any numeric work (refused here when it does not fit).
    const base::MemoryReservation workspace =
        m_budget.reserve(static_cast<std::uint64_t>(m_schedule.workspaceValues()) * VALUE_BYTES,
                         "factorisation: fronts and update matrices");

    base::TaskGraph graph;
    std::vector<base::TaskGraph::TaskId> taskOfSlot(static_cast<std::size_t>(m_schedule.slotCount()));
    // The subtrees the most work first (FactorSchedule::subtreesByWork: the largest do not start last).
    for (const std::size_t index : m_schedule.subtreesByWork())
    {
        const FactorSubtree& subtree = m_schedule.subtrees()[index];
        taskOfSlot[static_cast<std::size_t>(m_schedule.slotOf(subtree.last))] =
            graph.add([this, &subtree, &factor] { runSubtree(subtree, factor); });
    }
    for (const std::int32_t s : m_schedule.topSupernodes())
    {
        std::vector<base::TaskGraph::TaskId> dependencies;
        for (const std::int32_t child : m_schedule.children(s))
            dependencies.push_back(taskOfSlot[static_cast<std::size_t>(m_schedule.slotOf(child))]);
        const bool isTrailing = s == m_schedule.excludedTrailing();
        taskOfSlot[static_cast<std::size_t>(m_schedule.slotOf(s))] = graph.add(
            [this, s, isTrailing, &factor]
            {
                if (isTrailing)
                    assembleTrailing(s, m_schurComplement);
                else
                    runTopSupernode(s, factor);
            },
            dependencies);
    }
    graph.run(m_kernels.options().pool, m_options.cancellation);
    factor.m_workspaceUsage = {m_schedule.workspaceValues(), m_meter.peak(), m_subtreesBeyondPlan.load(),
                               m_pool.heldPeak()};
    for (const FactorSubtree& subtree : m_schedule.subtrees())
        factor.m_subtrees.push_back({subtree.first, subtree.last});
    finish(factor);
}

void MultifrontalDriver::finish(SupernodalFactor& factor)
{
    // Zero pivots first: the factor is unusable. They are listed in the order of the supernodes (the set does not
    // depend on the threads: the supernodes skipped are the ancestors of the failed ones).
    // At most RECORDED_ISSUE_COUNT DOFs are carried (a bound that does not grow with the model), all are counted.
    std::vector<base::GridComponent> zeroPivotDofs;
    std::string zeroPivotNames;
    std::size_t zeroPivotCount = 0;
    for (std::int32_t s = 0; s < m_symbolic.supernodeCount(); ++s)
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
    if (zeroPivotCount > PivotDiagnostics::LISTED_ISSUE_COUNT)
        zeroPivotNames.append("  and ")
            .append(std::to_string(zeroPivotCount - PivotDiagnostics::LISTED_ISSUE_COUNT))
            .append(" more\n");
    if (zeroPivotCount > 0)
        throw base::SingularityError(
            base::concatenate(
                {"the stiffness matrix is singular: ", std::to_string(zeroPivotCount),
                 " zero (or not finite) pivots met in the factorisation, a mechanism or an unconstrained "
                 "DOF at:\n",
                 zeroPivotNames,
                 "constrain these DOFs (single-point constraints) or connect them to the structure; with a matrix "
                 "that is meant to be indefinite, factorise it in indefinite mode"}),
            std::move(zeroPivotDofs));

    // The inertia, the pivot counts and the ratio test, panel by panel in elimination order.
    Inertia inertia;
    std::int64_t twoByTwo = 0;
    std::int64_t delayed = 0;
    std::int32_t largestPanelRows = 0;
    std::vector<PivotIssue> issues; // at most RECORDED_ISSUE_COUNT, all counted in issueCount
    std::int64_t issueCount = 0;
    double largestRatio = 0.0;
    std::int32_t equationOfLargest = -1;
    const double maxRatio = m_options.pivotPolicy.maxRatio;
    for (std::int32_t s = 0; s < factor.supernodeCount(); ++s)
    {
        const FactorPanel panel = factor.panel(s);
        delayed += m_delayedOut[static_cast<std::size_t>(s)];
        largestPanelRows = std::max(largestPanelRows, static_cast<std::int32_t>(panel.rows.size()));
        if (m_spool != nullptr)
        {
            // Out of core: the inertia taken as the panel was written (the kernels' count of its D, the same signs).
            const Inertia& own = m_inertiaOf[static_cast<std::size_t>(s)];
            inertia.positive += own.positive;
            inertia.negative += own.negative;
            inertia.zero += own.zero;
            twoByTwo += m_twoByTwoOf[static_cast<std::size_t>(s)];
            continue;
        }
        const BlockDiagonalView d = panel.pivots();
        std::size_t k = 0;
        while (k < d.size)
        {
            const double pivot = d.diagonal[k * d.diagonalStride];
            if (d.subdiagonal != nullptr && k + 1 < d.size && d.subdiagonal[k] != 0.0)
            {
                // A 2 × 2 block: the signs of its eigenvalues from its determinant (as inertiaOf in dense).
                const double second = d.diagonal[(k + 1) * d.diagonalStride];
                const double determinant = pivot * second - d.subdiagonal[k] * d.subdiagonal[k];
                if (determinant < 0.0)
                {
                    ++inertia.positive;
                    ++inertia.negative;
                }
                else if (determinant > 0.0)
                {
                    (pivot > 0.0 ? inertia.positive : inertia.negative) += 2;
                }
                else
                {
                    ++inertia.zero;
                    const double trace = pivot + second;
                    ++(trace > 0.0 ? inertia.positive : trace < 0.0 ? inertia.negative : inertia.zero);
                }
                ++twoByTwo;
                k += 2;
                continue;
            }
            ++(pivot > 0.0 ? inertia.positive : pivot < 0.0 ? inertia.negative : inertia.zero);
            if (m_options.pivoting == PivotingMode::PositiveDefinite)
            {
                // The ratio a_jj / d_j that PARAM MAXRATIO bounds.
                const std::int32_t equation = panel.rows[k];
                const double diagonal = matrixDiagonal(*m_matrix, equation);
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
                        issues.push_back(
                            {equation, diagonal, pivot, ratio,
                             ratio < 0.0 ? PivotIssueKind::NegativePivot : PivotIssueKind::RatioAboveMaximum});
                }
            }
            k += 1;
        }
    }
    factor.m_inertia = inertia;
    factor.m_twoByTwoPivotCount = twoByTwo;
    factor.m_delayedPivotCount = delayed;
    factor.m_largestPanelRowCount = largestPanelRows;
    std::int64_t allocated =
        m_spool != nullptr ? 0 : factor.m_panelOffset[static_cast<std::size_t>(factor.supernodeCount())];
    for (const auto& grown : factor.m_grownPanels)
        if (grown)
            allocated += static_cast<std::int64_t>(grown->values.size());
    factor.m_allocatedPanelValueCount = allocated;
    factor.m_diagnostics = PivotDiagnostics(std::move(issues), issueCount, largestRatio, equationOfLargest, maxRatio);
    if (m_options.pivoting == PivotingMode::PositiveDefinite)
        factor.m_diagnostics.enforce(m_options.pivotPolicy, m_options.equationNamer);

    // The plan of the solves, from the final panels; the parent of a supernode below an excluded trailing one (a
    // partial factorisation) is none.
    std::vector<std::int32_t> parents(static_cast<std::size_t>(factor.supernodeCount())); // one per supernode
    for (std::int32_t s = 0; s < factor.supernodeCount(); ++s)
    {
        const std::int32_t parent = m_symbolic.supernodes().parent(s);
        parents[static_cast<std::size_t>(s)] =
            parent == Supernodes::ROOT || parent >= factor.supernodeCount() ? SolvePlan::NO_PARENT : parent;
    }
    // A plan kept from an earlier factorisation of the same symbolic factor is that plan when the pivots are those of
    // the symbolic factor (FactorOptions::keptSolvePlan); its fit is checked all the same.
    const bool isSymbolicShape = m_options.pivoting == PivotingMode::PositiveDefinite && delayed == 0;
    if (isSymbolicShape && m_options.keptSolvePlan != nullptr &&
        m_options.keptSolvePlan->object().isFor(factor, parents))
    {
        factor.m_keptSolvePlan = m_options.keptSolvePlan;
        factor.m_solvePlan = factor.m_keptSolvePlan->lend(m_budget, "solve plan (kept)");
        return;
    }
    if (isSymbolicShape && m_options.keepsSolvePlan)
    {
        factor.m_keptSolvePlan = base::KeptObject<SolvePlan>::build([&](base::MemoryBudget& planBudget)
                                                                    { return SolvePlan(factor, parents, planBudget); });
        factor.m_solvePlan = factor.m_keptSolvePlan->lend(m_budget, "solve plan (kept)");
        return;
    }
    factor.m_solvePlan = std::make_shared<const SolvePlan>(factor, parents, m_budget);
}

} // namespace oos::linalg
