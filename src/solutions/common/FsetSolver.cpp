#include "oos/solutions/common/FsetSolver.hpp"

#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

namespace oos::solutions
{

FsetSolver::FsetSolver(const linalg::BlockSparseMatrix& knn, const linalg::NodeGraph& graph,
                       const assembly::DofMap& dofMap, const assembly::DofSets& sets,
                       const linalg::DenseKernels& kernels, const FsetSolverOptions& options,
                       base::MemoryBudget& budget, base::ThreadPool* pool)
    : m_dofMap(&dofMap), m_kernels(&kernels)
{
    base::Stopwatch watch;
    std::shared_ptr<KeptOrdering> kept;
    if (options.structures != nullptr)
    {
        // The structure of an earlier analysis with the same key (StructureCache), or built and kept for the next.
        StructureOrdering taken = options.structures->ordering(graph, dofMap, sets, options.keptSet, options.ordering,
                                                               pool, options.cancellation);
        kept = std::move(taken.kept);
        m_reuse.isOrderingReused = taken.isReused;
        m_ordering = kept->ordering().lend(budget, "ordering and symbolic factor (kept)");
        const double seconds = watch.lap();
        m_timings.symbolic = taken.isReused ? 0.0 : m_ordering->timings().symbolic;
        m_timings.ordering = seconds - m_timings.symbolic;
    }
    else
    {
        m_ordering = std::make_shared<const FsetOrdering>(graph, dofMap, sets, options.keptSet, options.ordering,
                                                          budget, pool, options.cancellation);
        m_timings = m_ordering->timings();
        static_cast<void>(watch.lap());
    }
    if (m_ordering->equationCount() == 0)
        return;
    // The steps before the numeric factorisation are cancellable; their progress units (nodes, equations) differ from
    // the counter's (the columns eliminated), so only the factorisation advances it.
    const base::OperationControl control{options.cancellation, nullptr};
    const linalg::SymmetricCsc matrix =
        linalg::SymmetricCsc::fromBlocks(knn, m_ordering->selection(), budget, pool, control);
    m_timings.conversion = watch.lap();
    linalg::FactorOptions factorOptions;
    factorOptions.pivoting = linalg::PivotingMode::PositiveDefinite;
    factorOptions.pivotPolicy = options.pivotPolicy;
    factorOptions.equationNamer = [this](std::int32_t equation) { return nameOf(equation); };
    factorOptions.cancellation = options.cancellation;
    factorOptions.progress = options.progress;
    linalg::FactorStorage storage;
    if (kept != nullptr)
    {
        // The plan of the solves of the first factorisation of the kept ordering, or built and kept now; the panels'
        // storage of a freed factor of this shape.
        factorOptions.keptSolvePlan = kept->plan();
        factorOptions.keepsSolvePlan = factorOptions.keptSolvePlan == nullptr;
        storage = kept->storageSlot().take(m_ordering->symbolic());
        factorOptions.storage = &storage;
    }
    const bool hasStorage = !storage.empty();
    m_factor = linalg::SupernodalFactor::factorise(m_ordering->symbolic(), matrix, kernels, factorOptions, budget);
    if (kept != nullptr)
    {
        m_reuse.isPlanReused =
            factorOptions.keptSolvePlan != nullptr && m_factor.keptSolvePlan() == factorOptions.keptSolvePlan;
        if (!m_reuse.isPlanReused && m_factor.keptSolvePlan() != nullptr)
            kept->offerPlan(m_factor.keptSolvePlan());
        options.structures->countPlan(m_reuse.isPlanReused);
        m_reuse.isStorageReused = hasStorage && storage.empty();
        options.structures->countStorage(m_reuse.isStorageReused);
        m_kept = std::move(kept);
    }
    m_timings.numeric = watch.lap();
}

FsetSolver::~FsetSolver()
{
    // The panels' memory goes to the cache's slot for the next factorisation of this shape, its reservation
    // given back to the budget.
    if (m_kept != nullptr && m_factor.supernodeCount() > 0)
        m_kept->storageSlot().offer(m_factor.releaseStorage());
}

void FsetSolver::solve(linalg::DenseMatrixView rightHandSides, const base::CancellationToken* cancellation) const
{
    if (equationCount() == 0)
        return;
    solve(rightHandSides, *m_kernels, cancellation);
}

void FsetSolver::solve(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
                       const base::CancellationToken* cancellation) const
{
    if (equationCount() == 0)
        return;
    linalg::FactorSolve(m_factor, kernels).solve(rightHandSides, cancellation);
}

base::GridComponent FsetSolver::nameOf(std::int32_t equation) const
{
    return equationName(*m_dofMap, dofOfEquation(equation));
}

base::GridComponent equationName(const assembly::DofMap& dofMap, base::DofIndex dof)
{
    const assembly::DofLabel label = dofMap.labelOf(dof);
    return {dofMap.pointIdOf(dof), label.kind == input::PointKind::Scalar ? 0 : static_cast<int>(label.component)};
}

} // namespace oos::solutions
