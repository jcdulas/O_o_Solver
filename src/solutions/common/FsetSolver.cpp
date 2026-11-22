#include "oos/solutions/common/FsetSolver.hpp"

#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <cstddef>

namespace oos::solutions
{

namespace
{

// The diagonal term of a stiffness matrix at a g-set DOF (0 when the node of the DOF has no block).
double diagonalOf(const linalg::BlockSparseMatrix& matrix, const assembly::DofMap& dofMap, base::DofIndex dof)
{
    const linalg::NodeDof at = dofMap.nodeDofOf(dof);
    const linalg::SparsePattern& pattern = matrix.pattern();
    const std::int64_t block = pattern.findBlock(at.node, at.node);
    if (block < 0)
        return 0.0;
    // The diagonal block of a node is square and column-major: term (c, c) is at c·(count + 1).
    const auto count = static_cast<std::size_t>(pattern.nodeDofCount(at.node));
    return matrix.blockValues(block)[static_cast<std::size_t>(at.component) * (count + 1)];
}

// A pivot below the diagonal of Knn over this ratio has lost twelve of the sixteen digits of a double to the
// elimination: it is not known from zero, whatever the stiffness of the elements of its DOF.
constexpr double PIVOT_PRECISION_RATIO = 1.0e12;

// How many times MaxDiagonalRatio the ratio of an entity DOF may be (the class comment of FsetSolver says why).
// Measured on a cantilever plate bent out of its plane (length 6, width 0.2, thickness 0.1), 6 × 1 quadrilaterals of
// the mixed hybrid formulation of degree 4 cut in two, four, eight, twelve, sixteen and twenty-four each way: the
// largest ratio, at a function of an edge across the plate, is 6e6, 3e7, 1e8, 2e8, 5e8 then 1e9 (four times more
// each time the elements are cut in two), and the tip deflection is exact to 4e-5 up to 1e8, then off by 9e-4, 4e-3
// and 1e-2: the loss grows as the square of the ratio. Ten times MaxDiagonalRatio (1e8 at its default) keeps it
// below 2e-4.
constexpr double ENTITY_RATIO_ALLOWANCE = 10.0;

// The reference diagonal of the ratio test (linalg::FactorOptions::referenceDiagonal), one value per equation of
// `selection`: at an entity DOF, the diagonal of the stiffness before the relations are eliminated (`kgg`: the
// stiffness of the elements of the DOF; that of `knn` where it is the smaller) over ENTITY_RATIO_ALLOWANCE (the class
// comment of FsetSolver says why); no reference at the other DOFs.
//
// The reference is not taken below the diagonal of Knn times maxRatio / PIVOT_PRECISION_RATIO, so that the test,
// reference / pivot > maxRatio, also fails a pivot below the diagonal of Knn over PIVOT_PRECISION_RATIO: under a
// very stiff relation the round-off of the elimination (a fraction of the diagonal of Knn) may exceed the stiffness
// of the elements over maxRatio, and a mechanism would pass the test against that stiffness alone.
base::BudgetedBuffer<double> entityReferenceDiagonal(const linalg::BlockSparseMatrix& kgg,
                                                     const linalg::BlockSparseMatrix& knn,
                                                     const assembly::DofMap& dofMap,
                                                     const linalg::DofSelection& selection, double maxRatio,
                                                     base::MemoryBudget& budget)
{
    const auto equations = static_cast<std::size_t>(selection.keptDofCount());
    base::BudgetedBuffer<double> reference(budget, equations, "ratio test reference diagonal");
    const double precisionShare = maxRatio / PIVOT_PRECISION_RATIO;
    for (std::size_t equation = 0; equation < equations; ++equation)
    {
        const auto dof = static_cast<base::DofIndex>(selection.dofOfEquation(static_cast<std::int32_t>(equation)));
        reference[equation] = linalg::NO_REFERENCE_DIAGONAL;
        if (dof < dofMap.firstEntityDof())
            continue;
        const double reduced = diagonalOf(knn, dofMap, dof);
        const double ofElements = std::min(diagonalOf(kgg, dofMap, dof), reduced);
        const double candidate = std::max(ofElements / ENTITY_RATIO_ALLOWANCE, precisionShare * reduced);
        if (candidate < reduced)
            reference[equation] = candidate;
    }
    return reference;
}

} // namespace

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
    // The reference of the ratio test at the entity DOFs; none (the test as it is) for a model without entity DOFs.
    base::BudgetedBuffer<double> referenceDiagonal;
    if (dofMap.hasEntities())
        referenceDiagonal = entityReferenceDiagonal(
            options.unreducedStiffness != nullptr ? *options.unreducedStiffness : knn, knn, dofMap,
            m_ordering->selection(), options.pivotPolicy.maxRatio, budget);
    m_timings.conversion = watch.lap();
    linalg::FactorOptions factorOptions;
    factorOptions.pivoting = linalg::PivotingMode::PositiveDefinite;
    factorOptions.pivotPolicy = options.pivotPolicy;
    factorOptions.referenceDiagonal = referenceDiagonal.span();
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
    if (label.kind == input::PointKind::Entity)
        return {0, static_cast<int>(label.component) + 1, dofMap.describe(dof)};
    return {dofMap.pointIdOf(dof), label.kind == input::PointKind::Scalar ? 0 : static_cast<int>(label.component), {}};
}

} // namespace oos::solutions
