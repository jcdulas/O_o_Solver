#include "oos/solutions/sol101/StreamedStatics.hpp"

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/linalg/dense/IterativeRefinement.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/BasisFixedRows.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"
#include "oos/solutions/sol101/StreamedFront.hpp"
#include "oos/solutions/sol101/StreamedLoads.hpp"
#include "oos/solutions/sol101/StreamedOutputs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::solutions
{
namespace
{

constexpr std::uint64_t VALUE_BYTES = sizeof(double);
// The memory plan's margin over its parts (the allocations a part does not count: the blocks of the element facets,
// the kernels' packing, the vectors a front's terms make), measured: the budget's peak against the estimate on the
// tests and the ladder.
constexpr double PLAN_MARGIN = 1.05;
constexpr std::uint64_t PLAN_MARGIN_BYTES = std::uint64_t{64} << 20;
// A model of few elements has as few of those allocations: the margin at most PLAN_MARGIN_BASE_BYTES and
// PLAN_MARGIN_ELEMENT_BYTES an element (a run of elements' blocks on each thread).
constexpr std::uint64_t PLAN_MARGIN_BASE_BYTES = std::uint64_t{4} << 20;
constexpr std::uint64_t PLAN_MARGIN_ELEMENT_BYTES = std::uint64_t{16} << 10;
// The refinement of a relieved solution: the solutions it holds whole (the solution, K·û, the correction),
// and the share of the memory left at most they take (beside the solve's own, which the plan sizes after).
constexpr std::uint64_t REFINED_SOLUTIONS = 3;
constexpr std::uint64_t REFINEMENT_BUDGET_DIVISOR = 4;

base::Diagnostic problem(base::Severity severity, std::string code, std::string card, base::EntityId id,
                         std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = std::move(card);
    diagnostic.id = id;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The rigid-body motions of a g-set DOF about the basic origin (RigidBodyModes' convention).
std::array<double, 6> rigidRowOf(const assembly::DofMap& dofMap, base::DofIndex dof)
{
    std::array<double, 6> row{};
    const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dof);
    if (nodeDof.node >= dofMap.gridCount())
        return row;
    const base::Mat3 toCd = dofMap.cdToBasic(nodeDof.node).transposed();
    const base::Vec3 x = dofMap.positionInBasic(nodeDof.node);
    const auto c = static_cast<std::size_t>(nodeDof.component);
    for (std::size_t mode = 0; mode < 6; ++mode)
    {
        base::Vec3 translation{};
        base::Vec3 rotation{};
        if (mode < 3)
            translation[mode] = 1.0;
        else
        {
            rotation[mode - 3] = 1.0;
            translation = base::cross(rotation, x);
        }
        row[mode] = c < 3 ? (toCd * translation)[c] : (toCd * rotation)[c - 3];
    }
    return row;
}

// x = A⁻¹·b for a 6 × 6 symmetric positive definite A, by Cholesky (Golub and Van Loan, Algorithm 4.2.1); nothing
// when A is not positive definite (a support DOF whose motion moves no mass).
// The solution of a streamed solve held whole (for the refinement of a relieved solution): each
// supernode's columns written into `into` (equations × vectors).
class StoredSolution final : public linalg::SolutionSink
{
public:
    StoredSolution(const linalg::SymbolicFactor& symbolic, linalg::DenseMatrixView into)
        : m_symbolic(&symbolic), m_into(into)
    {
    }

    void take(std::int32_t supernode, linalg::ConstDenseMatrixView columnValues, linalg::ConstDenseMatrixView) override
    {
        const auto first = static_cast<std::size_t>(m_symbolic->firstColumn(supernode));
        for (std::size_t b = 0; b < columnValues.columns; ++b)
            for (std::size_t p = 0; p < columnValues.rows; ++p)
                m_into(first + p, b) = columnValues(p, b);
    }

private:
    const linalg::SymbolicFactor* m_symbolic;
    linalg::DenseMatrixView m_into;
};

// A solution held whole handed to a sink as the downward pass hands it (top-down, each supernode with the values of
// its rows R(s)).
void handOver(const linalg::SymbolicFactor& symbolic, linalg::ConstDenseMatrixView solution, linalg::SolutionSink& sink,
              base::MemoryBudget& budget)
{
    const std::size_t columns = solution.columns;
    for (std::int32_t s = symbolic.supernodeCount() - 1; s >= 0; --s)
    {
        const auto first = static_cast<std::size_t>(symbolic.firstColumn(s));
        const auto count = static_cast<std::size_t>(symbolic.columnCount(s));
        const std::span<const std::int32_t> rows = symbolic.rows(s);
        linalg::DenseMatrix columnValues(budget, count, columns, "streamed handed columns");
        linalg::DenseMatrix rowValues(budget, rows.size(), columns, "streamed handed rows");
        for (std::size_t b = 0; b < columns; ++b)
        {
            for (std::size_t p = 0; p < count; ++p)
                columnValues(p, b) = solution(first + p, b);
            for (std::size_t r = 0; r < rows.size(); ++r)
                rowValues(r, b) = solution(static_cast<std::size_t>(rows[r]), b);
        }
        sink.take(s, columnValues.view(), rowValues.view());
    }
}

std::optional<std::array<double, 6>> solveSix(const base::MatN<6>& a, std::array<double, 6> b)
{
    base::MatN<6> l{};
    for (std::size_t j = 0; j < 6; ++j)
    {
        double diagonal = a(j, j);
        for (std::size_t k = 0; k < j; ++k)
            diagonal -= l(j, k) * l(j, k);
        if (!(diagonal > 0.0))
            return std::nullopt;
        l(j, j) = std::sqrt(diagonal);
        for (std::size_t i = j + 1; i < 6; ++i)
        {
            double value = a(i, j);
            for (std::size_t k = 0; k < j; ++k)
                value -= l(i, k) * l(j, k);
            l(i, j) = value / l(j, j);
        }
    }
    for (std::size_t i = 0; i < 6; ++i)
    {
        for (std::size_t k = 0; k < i; ++k)
            b[i] -= l(i, k) * b[k];
        b[i] /= l(i, i);
    }
    for (std::size_t i = 6; i-- > 0;)
    {
        for (std::size_t k = i + 1; k < 6; ++k)
            b[i] -= l(k, i) * b[k];
        b[i] /= l(i, i);
    }
    return b;
}

// The plan's time of a candidate in units of the upward pass's work on the threads: the upward pass's work, the region
// runs' on the threads times PARALLEL_RUN_SLOWDOWN (a run's window holds fewer subtrees than the whole tree: threads
// wait more), and the work made in one postorder on the calling thread (the order Postorder, a region inside one
// subtree of the cut) times SERIAL_WORK_SLOWDOWN (its small fronts on one thread, its large ones on the tasks), the
// ratios measured on six threads, interpolated from 1 on one thread and kept beyond six (more threads make the
// threaded parts relatively faster still: the choice then leans the same way).
// Measured (the barrel of three frames at 1/128, 1/64 and 1/32, six threads): the postorder 2.9 to 3.4 times slower a
// unit of work than the parallel order's upward pass, the runs on the threads 1.3 to 1.8 times.
constexpr double SERIAL_WORK_SLOWDOWN = 3.0;
constexpr double PARALLEL_RUN_SLOWDOWN = 1.5;

double estimatedTime(const linalg::RegionSweepCost& cost, std::size_t threads) noexcept
{
    constexpr double SLOWDOWN_THREADS = 6.0;
    const double reach = std::min(static_cast<double>(std::max<std::size_t>(threads, 1)) - 1.0, SLOWDOWN_THREADS - 1.0);
    const auto at = [&](double ratio) { return 1.0 + (ratio - 1.0) * reach / (SLOWDOWN_THREADS - 1.0); };
    const double serialWork = cost.serialUpwardWork + cost.serialRunWork;
    const double threadedRunWork = cost.recomputedWork - cost.serialRunWork;
    return (cost.upwardWork - cost.serialUpwardWork) + at(PARALLEL_RUN_SLOWDOWN) * threadedRunWork +
           at(SERIAL_WORK_SLOWDOWN) * serialWork;
}

// The region bounds a plan tries, largest first: every panel kept, then halves down to one panel.
std::vector<std::int64_t> candidateBounds(const linalg::FrontStreamEstimate& whole)
{
    std::vector<std::int64_t> bounds; // bounded by the halvings
    std::int64_t bound = std::max<std::int64_t>(whole.regions.topValues, 1);
    bounds.push_back(std::numeric_limits<std::int64_t>::max() / 4);
    while (bound > std::max<std::int64_t>(whole.residentValues, 1))
    {
        bounds.push_back(bound);
        bound /= 2;
    }
    bounds.push_back(1);
    return bounds;
}

// The connection elements (for the dissection aligned on the components): they join components, as the
// relations do (which are no elements); every other element of two nodes or more is structural and joins its nodes
// into one component.
bool isConnectionCard(std::string_view card) noexcept
{
    constexpr std::string_view CONNECTIONS[] = {
        "Bushing",    "Bushing1D",  "Bushing2D",        "Fastener", "Weld",    "SpringDofProperty",
        "SpringDofInline", "SpringScalarProperty", "SpringScalarInline",       "DamperDofProperty",  "DamperDofInline", "DamperScalarProperty",
        "DamperScalarInline", "Gap2",     "ViscousDamper2", "PlotLine2", "PointMassMatrix",  "PointMassInertia",
        "MassDofProperty",   "MassDofInline",   "MassScalarProperty",         "MassScalarInline",    "GeneralMatrix"};
    return std::find(std::begin(CONNECTIONS), std::end(CONNECTIONS), card) != std::end(CONNECTIONS);
}

// The component of each node of the model (dense IDs from 0 by first node): the connected sets of the structural
// elements' nodes (a union-find over each element's nodes); a node of no structural element is a component of its
// own.
base::BudgetedBuffer<std::int32_t> componentsOfNodes(const AnalysisModel& model, std::int32_t& componentCount,
                                                     base::MemoryBudget& budget)
{
    const auto nodes = static_cast<std::size_t>(model.dofMap().nodeCount());
    base::BudgetedBuffer<std::int32_t> parent(budget, nodes, "dissection: components", base::FOR_OVERWRITE);
    for (std::size_t n = 0; n < nodes; ++n)
        parent[n] = static_cast<std::int32_t>(n);
    const auto find = [&](std::int32_t node)
    {
        while (parent[static_cast<std::size_t>(node)] != node)
        {
            const std::int32_t up = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(node)])];
            parent[static_cast<std::size_t>(node)] = up;
            node = up;
        }
        return node;
    };
    const assembly::ElementConnectivity& connectivity = model.connectivity();
    const linalg::NodeGroupList groups = connectivity.groups();
    const std::span<const std::int32_t> groupIds = connectivity.groupElementIds();
    std::size_t group = 0;
    for (const assembly::AssemblyBatch& batch : model.elementSet().batches())
    {
        const bool isConnection = isConnectionCard(batch.family->card());
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
        {
            if (group >= groupIds.size() || groupIds[group] != batch.view.elementId(row))
                continue;
            const std::span<const std::int32_t> members = groups.group(group++);
            if (isConnection || members.empty())
                continue;
            const std::int32_t first = find(members.front());
            for (const std::int32_t node : members.subspan(1))
            {
                const std::int32_t root = find(node);
                if (root != first)
                    parent[static_cast<std::size_t>(std::max(root, first))] = std::min(root, first);
            }
        }
    }
    base::BudgetedBuffer<std::int32_t> component(budget, nodes, "dissection: components", base::FOR_OVERWRITE);
    componentCount = 0;
    for (std::size_t n = 0; n < nodes; ++n)
    {
        const std::int32_t root = find(static_cast<std::int32_t>(n));
        component[n] =
            root == static_cast<std::int32_t>(n) ? componentCount++ : component[static_cast<std::size_t>(root)];
    }
    return component;
}

// The separators of the top levels of the elimination tree (level 0 the root's): each separator a chain of supernodes
// with one child each, the levels below it those of its chain's last supernode's children.
std::vector<std::int64_t> separatorLevels(const linalg::SymbolicFactor& symbolic, std::size_t levels)
{
    std::vector<std::int64_t> dofs(levels, 0);
    const std::int32_t count = symbolic.supernodeCount();
    std::vector<std::vector<std::int32_t>> children(static_cast<std::size_t>(count)); // bounded by the supernodes
    std::vector<std::int32_t> roots;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int32_t p = symbolic.supernodes().parent(s);
        if (p == linalg::Supernodes::ROOT)
            roots.push_back(s);
        else
            children[static_cast<std::size_t>(p)].push_back(s);
    }
    std::vector<std::pair<std::int32_t, std::size_t>> stack; // (separator's top, level)
    for (const std::int32_t root : roots)
        stack.emplace_back(root, 0);
    while (!stack.empty())
    {
        auto [s, level] = stack.back();
        stack.pop_back();
        if (level >= levels)
            continue;
        for (;;)
        {
            dofs[level] += symbolic.columnCount(s);
            if (children[static_cast<std::size_t>(s)].size() != 1)
                break;
            s = children[static_cast<std::size_t>(s)].front();
        }
        for (const std::int32_t child : children[static_cast<std::size_t>(s)])
            stack.emplace_back(child, level + 1);
    }
    return dofs;
}

} // namespace

std::string_view nameOf(StreamedDissection dissection) noexcept
{
    switch (dissection)
    {
    case StreamedDissection::Free:
        return "free";
    case StreamedDissection::Components:
        return "components";
    case StreamedDissection::Hybrid:
        return "hybrid";
    }
    return "free";
}

std::string_view nameOf(StreamedRetention retention) noexcept
{
    switch (retention)
    {
    case StreamedRetention::Auto:
        return "auto";
    case StreamedRetention::None:
        return "none";
    case StreamedRetention::Given:
        return "given";
    }
    return "auto";
}

StreamedStatics::GroupResult StreamedStatics::solveGroup(const GroupInputs& inputs, const StreamedOptions& options,
                                                         base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    base::Stopwatch watch;
    const AnalysisModel& model = *inputs.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const StaticSubcase& first = inputs.subcases[inputs.cases.front()];
    const output::OutputPlan& plan = inputs.plans[inputs.cases.front()];
    GroupResult result;
    StreamedSetRecord& record = result.record;
    record.constraintSet = inputs.constraintSet;
    record.dissection = std::string(nameOf(options.dissection));

    // What the path refuses, before any element matrix.
    const char* otherPaths = "solve it on the full path (not streamed) or the condensed path";
    if (inputs.selection.contact)
        sink.report(problem(base::Severity::Error, "streamed-unsupported", "ContactSelection", first.label,
                            base::concatenate({"expected no glued contact on the full path streamed; ", otherPaths})));
    if (inputs.relief == assembly::InertiaReliefMode::Automatic)
        sink.report(
            problem(base::Severity::Error, "streamed-unsupported", "PARAM InertiaRelief", first.label,
                    base::concatenate({"expected PARAM InertiaRelief 0 or -1 on the full path streamed (-2 removes the "
                                       "rigid motion with the whole mass); ",
                                       otherPaths})));
    const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
    for (const std::size_t c : inputs.cases)
    {
        const output::OutputPlan& other = inputs.plans[c];
        const output::OutputRequestPlan& a = other.of(output::OutputKind::Displacement);
        const output::OutputRequestPlan& b = plan.of(output::OutputKind::Displacement);
        if (a.isRequested != b.isRequested || a.isAll != b.isAll || a.members != b.members ||
            !output::elementSelectionOf(other).selectsSameAs(selection))
            sink.report(problem(base::Severity::Error, "streamed-unsupported", "SubcaseStart", inputs.subcases[c].label,
                                base::concatenate({"expected the subcases of one constraint set to ask for the same "
                                                   "outputs on the full path streamed (its regions recover every case "
                                                   "at once); ",
                                                   otherPaths})));
        for (const output::OutputKind kind :
             {output::OutputKind::SpcForces, output::OutputKind::MpcForces, output::OutputKind::AppliedLoads,
              output::OutputKind::GridPointForces, output::OutputKind::StrainEnergy})
            if (other.of(kind).isRequested)
                sink.report(problem(
                    base::Severity::Warning, "streamed-output-not-produced", std::string(output::requestNameOf(kind)),
                    inputs.subcases[c].label,
                    base::concatenate({"the full path streamed gives the displacements and the element tables, not ",
                                       output::requestNameOf(kind),
                                       " (a product with the stiffness it never assembles); ", otherPaths})));
    }
    stopOnErrors(sink, "the full path streamed");

    // The constraints, the sets and the singularities (the n-set diagonal blocks summed element by element).
    std::optional<StreamedFront::NodeDiagonals> diagonals;
    ConstrainedSystemOptions systemOptions;
    systemOptions.spc = inputs.spc;
    systemOptions.supportDofs.assign(inputs.supports.begin(), inputs.supports.end());
    systemOptions.diagonalBlocks = [&](const assembly::RigidElimination& gm, const assembly::DofSets&)
    {
        diagonals.emplace(StreamedFront::nodeDiagonals(model, gm, budget, inputs.pool, inputs.cancellation));
        return std::move(diagonals->blocks);
    };
    const ConstrainedSystem system(model, inputs.selection, budget, inputs.pool, inputs.cancellation, sink,
                                   systemOptions);
    result.singularityTable = system.singularityTable();
    const assembly::DofSets& solveSets = system.solveSets();
    const std::vector<base::DofIndex>& rSet = system.supportDofs();
    record.autoSpcCount = solveSets.count(assembly::ExclusiveSet::AutoSpc);
    record.constraintSeconds = watch.lap();

    // Inertia relief of a free body on rigid support motions, solved on conditioned supports.
    const bool hasRelief = inputs.relief == assembly::InertiaReliefMode::OnSupports && !rSet.empty();
    std::optional<std::array<double, 36>> supportCoefficients;
    std::unique_ptr<assembly::DofSets> conditionedSets;
    if (hasRelief)
    {
        const auto refuseRelief = [&](const std::string& why)
        {
            sink.report(problem(base::Severity::Error, "streamed-unsupported", "FreeBodySupport", first.label,
                                base::concatenate({"the full path streamed solves inertia relief on the rigid motions "
                                                   "of a body free apart from six FreeBodySupport DOFs, found ",
                                                   why, "; ", otherPaths})));
            stopOnErrors(sink, "the full path streamed");
        };
        if (rSet.size() != assembly::MAX_SUPPORT_DOFS)
            refuseRelief(base::concatenate({std::to_string(rSet.size()), " FreeBodySupport DOFs"}));
        // Free apart from its supports: no element resists a rigid motion, no held DOF has stiffness.
        double supportStiffness = 0.0;
        for (const base::DofIndex dof : rSet)
            supportStiffness =
                std::max(supportStiffness,
                         std::abs(static_cast<double>(diagonals->diagonalOfDof[static_cast<std::size_t>(dof)])));
        bool isHeld = false;
        for (base::DofIndex dof = 0; dof < dofMap.gSetSize() && !isHeld; ++dof)
        {
            const assembly::ExclusiveSet set = solveSets.exclusiveSetOf(dof);
            isHeld = (set == assembly::ExclusiveSet::Sb || set == assembly::ExclusiveSet::Sg) &&
                     diagonals->diagonalOfDof[static_cast<std::size_t>(dof)] != 0.0F;
        }
        if (isHeld)
            refuseRelief("a single-point constraint on a DOF with stiffness");
        // The round-off of K·Φ scales with the largest stiffness of the model (fasteners 1e8 times the skins' make it
        // 1e8 times a support's), not with the supports' own.
        double largestStiffness = supportStiffness;
        for (const float diagonal : diagonals->diagonalOfDof.span())
            largestStiffness = std::max(largestStiffness, std::abs(static_cast<double>(diagonal)));
        if (diagonals->largestRigidForce > assembly::RIGID_SUPPORT_TOLERANCE * largestStiffness)
            refuseRelief(base::concatenate({"an element that resists a rigid motion (|K·Φ| ",
                                            std::to_string(diagonals->largestRigidForce), " against a support's ",
                                            std::to_string(supportStiffness), ")"}));
        std::array<double, 36> atSupports{};
        for (std::size_t j = 0; j < rSet.size(); ++j)
        {
            const std::array<double, 6> row = rigidRowOf(dofMap, rSet[j]);
            std::copy(row.begin(), row.end(), atSupports.begin() + static_cast<std::ptrdiff_t>(6 * j));
        }
        supportCoefficients = assembly::rigidSupportCoefficients(atSupports, rSet.size());
        if (!supportCoefficients)
            refuseRelief("FreeBodySupport DOFs that do not fix every rigid motion");
        base::BudgetedBuffer<std::uint8_t> isCandidate(budget, static_cast<std::size_t>(dofMap.gSetSize()),
                                                       "streamed relief support candidates");
        for (base::DofIndex dof = 0; dof < dofMap.gSetSize(); ++dof)
        {
            const assembly::ExclusiveSet set = solveSets.exclusiveSetOf(dof);
            isCandidate[static_cast<std::size_t>(dof)] =
                set == assembly::ExclusiveSet::B || set == assembly::ExclusiveSet::R ? 1 : 0;
        }
        const std::vector<base::DofIndex> solveSupports =
            assembly::chooseConditionedSupports(dofMap, isCandidate.span());
        if (solveSupports.size() != assembly::MAX_SUPPORT_DOFS)
            refuseRelief("a body whose points do not fix every rigid motion (a model on a line)");
        conditionedSets = std::make_unique<assembly::DofSets>(solveSets.withSupports(solveSupports));
        record.isRelieved = true;
    }
    const assembly::DofSets& factorSets = conditionedSets ? *conditionedSets : solveSets;

    // The load basis of the set (the full path's).
    StaticLoadCases::Inputs caseInputs;
    caseInputs.snapshot = &model.snapshot();
    caseInputs.points = &model.index().points();
    caseInputs.dofMap = &dofMap;
    caseInputs.loads = inputs.loads;
    std::vector<const StaticSubcase*> setCases; // bounded by the set
    for (const std::size_t c : inputs.cases)
    {
        caseInputs.cases.push_back(&inputs.subcases[c]);
        caseInputs.spcs.push_back(inputs.spc);
        setCases.push_back(&inputs.subcases[c]);
    }
    caseInputs.tolerance = inputs.tolerance;
    caseInputs.pool = inputs.pool;
    caseInputs.kernel = inputs.kernel;
    caseInputs.checksInertialLoads = false;
    const auto ownedCases = std::make_shared<const StaticLoadCases>(std::move(caseInputs), budget, sink);
    const StaticLoadCases& loadCases = *ownedCases;
    record.basisVectors = static_cast<std::int64_t>(loadCases.basisSize());
    record.cases = static_cast<std::int64_t>(inputs.cases.size());

    // The refinement of the relieved solution of a free body (linalg::IterativeRefinement): each
    // step another streamed solve of the residual, the solution held whole meanwhile (it, K·û and the correction:
    // equations × vectors each) and handed to the outputs once refined, when they take at most a
    // REFINEMENT_BUDGET_DIVISOR-th of the memory left; otherwise solved once, the FreeBodySupport points last in the
    // ordering so that the gauge is known first.
    const std::size_t columns = loadCases.basisSize();
    const std::size_t threads = inputs.pool != nullptr ? inputs.pool->threadCount() : 1;
    const std::uint64_t refinementBytes =
        hasRelief ? REFINED_SOLUTIONS * static_cast<std::uint64_t>(factorSets.count(assembly::DofSet::L)) *
                            static_cast<std::uint64_t>(columns) * VALUE_BYTES +
                        StreamedFront::refinementWorkBytes(threads, columns)
                  : 0;
    const bool refines = hasRelief && refinementBytes <= budget.available() / REFINEMENT_BUDGET_DIVISOR;
    record.isRefined = refines;
    if (hasRelief && !refines)
        sink.report(problem(base::Severity::Information, "streamed-relief-unrefined", "FreeBodySupport", first.label,
                            base::concatenate({"the full path streamed solves the free body once, without the "
                                               "refinement of its relieved solution (it would hold ",
                                               std::to_string(refinementBytes), " bytes of solutions, above a ",
                                               std::to_string(REFINEMENT_BUDGET_DIVISOR),
                                               "th of the memory left); give more memory or fewer cases per "
                                               "constraint set for the refined solution, or use the full path"})));

    // The ordering: unrefined, the FreeBodySupport points last (their displacements first in the downward pass, for the
    // gauge).
    std::vector<std::int32_t> lastNodes; // bounded by the supports
    for (const base::DofIndex dof : rSet)
    {
        const std::int32_t node = dofMap.nodeDofOf(dof).node;
        if (std::find(lastNodes.begin(), lastNodes.end(), node) == lastNodes.end())
            lastNodes.push_back(node);
    }
    // The supernodes no wider than the update matrices' blocks (AmalgamationThresholds::maxColumns): the panel a front
    // makes beside the updates waiting stays small at the top of the tree.
    linalg::AmalgamationThresholds thresholds;
    thresholds.maxColumns = linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS;
    // The dissection (StreamedDissection): free, or its regions of several components cut between them.
    std::int32_t componentCount = 0;
    const base::BudgetedBuffer<std::int32_t> components = options.dissection == StreamedDissection::Free
                                                              ? base::BudgetedBuffer<std::int32_t>()
                                                              : componentsOfNodes(model, componentCount, budget);
    const linalg::ComponentCuts cuts =
        options.dissection == StreamedDissection::Components ? linalg::ComponentCuts::Components
        : options.dissection == StreamedDissection::Hybrid   ? linalg::ComponentCuts::Hybrid
                                                             : linalg::ComponentCuts::Free;
    record.componentCount = componentCount;
    const FsetOrdering ordering(
        system.graph(), dofMap, factorSets, assembly::DofSet::L, linalg::OrderingMethod::Automatic, budget, inputs.pool,
        inputs.cancellation,
        hasRelief && !refines ? std::span<const std::int32_t>(lastNodes) : std::span<const std::int32_t>(), thresholds,
        cuts, components.span());
    const linalg::SymbolicFactor& symbolic = ordering.symbolic();
    const linalg::DofSelection& equations = ordering.selection();
    record.equations = ordering.equationCount();
    record.supernodes = symbolic.supernodeCount();
    record.factorValues = symbolic.panelValueCount();
    for (std::int32_t s = 0; s < symbolic.supernodeCount(); ++s)
        record.largestFrontOrder =
            std::max<std::int64_t>(record.largestFrontOrder, symbolic.columnCount(s) + symbolic.rowCount(s));
    // The top separator: the root supernode and its chain of single children.
    for (std::int32_t s = symbolic.supernodeCount() - 1; s >= 0;)
    {
        record.rootSeparatorDofs += symbolic.columnCount(s);
        std::int32_t child = -1;
        std::int32_t children = 0;
        for (std::int32_t k = s - 1; k >= 0 && children < 2; --k)
            if (symbolic.supernodes().parent(k) == s)
            {
                child = k;
                ++children;
            }
        s = children == 1 ? child : -1;
    }
    record.separatorLevelDofs = separatorLevels(symbolic, SEPARATOR_LEVELS);
    record.orderingSeconds = watch.lap();

    // The placement and the loads at the fronts.

    base::BudgetedBuffer<std::uint8_t> isOutputNode(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                    "streamed output points");
    const output::OutputRequestPlan& displacement = plan.of(output::OutputKind::Displacement);
    if (displacement.isRequested)
        for (std::int32_t node = 0; node < dofMap.nodeCount(); ++node)
        {
            const base::EntityId id =
                node < dofMap.gridCount() ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount());
            isOutputNode[static_cast<std::size_t>(node)] =
                displacement.isAll || std::binary_search(displacement.members.begin(), displacement.members.end(), id)
                    ? 1
                    : 0;
        }
    StreamedElements::Inputs placementInputs;
    placementInputs.model = &model;
    placementInputs.gm = &system.gm();
    placementInputs.selection = &equations;
    placementInputs.symbolic = &symbolic;
    placementInputs.isOutputNode = isOutputNode.span();
    const StreamedElements elements(placementInputs, budget);
    record.elementsPlaced = elements.elementCount();
    StreamedLoads::Inputs loadInputs;
    loadInputs.cases = &loadCases;
    loadInputs.loads = inputs.loads;
    loadInputs.elements = &elements;
    loadInputs.subcases = setCases;
    StreamedLoads loads(loadInputs, budget, sink);
    stopOnErrors(sink, "the full path streamed");
    record.placementSeconds = watch.lap();

    const assembly::NonStructuralMassShares shares(model.elementSet());
    StreamedFront::Inputs frontInputs;
    frontInputs.model = &model;
    frontInputs.elements = &elements;
    frontInputs.loads = &loads;
    frontInputs.loadBuilder = inputs.loads;
    frontInputs.massOptions = assembly::MassAssemblyOptions::fromParams(model.params(), first.label, sink);
    frontInputs.shares = &shares;
    frontInputs.diagonalOfDof = diagonals->diagonalOfDof.span();
    frontInputs.gm = &system.gm();
    frontInputs.reliefSets = &solveSets;
    StreamedFront front(frontInputs);
    base::MatN<6> rigidBodyMass{}; // Φᵀ·M·Φ (the refinement's gauge)
    if (hasRelief)
    {
        // w = (Rᵀ·M·R)⁻¹·Rᵀ·P (R the rows of reliefRowOf, the rigid-body motions Φ): the relieved loads P − M·R·w,
        // those of D = R·C (ü = C⁻¹·w) made without C, so that they do not depend on the FreeBodySupport DOFs in
        // floating point either (as the full path's): the FreeBodySupport DOFs are the gauge of the displacements only.
        const StreamedFront::ReliefSums sums = front.reliefSums(inputs.pool, budget, inputs.cancellation);
        rigidBodyMass = sums.rigidBodyMass;
        std::vector<std::array<double, 6>> amplitudes(loadCases.basisSize()); // per vector
        for (std::size_t v = 0; v < loadCases.basisSize(); ++v)
        {
            const std::optional<std::array<double, 6>> accelerations = solveSix(sums.rigidBodyMass, sums.resultants[v]);
            if (!accelerations)
                throw base::SolverError(
                    "the full path streamed: the mass of the free body does not move with its six FreeBodySupport DOFs "
                    "(DᵀMD is singular); give the body mass, or solve it on the full path");
            amplitudes[v] = *accelerations;
        }
        loads.setRelief(std::move(amplitudes));
        record.reliefSeconds = watch.lap();
    }

    // The load basis with declared quantities: the fixed rows' points, their relations and the vectors'
    // loads there as the solve takes them (relief included), and the nodes whose elements give grid point forces.
    std::shared_ptr<KeptFixedRows> fixedRows;
    base::BudgetedBuffer<std::uint8_t> isForceNode;
    if (inputs.keepsLoadBasis && inputs.basisFixedRows.isRequested)
    {
        if (inputs.basisFixedRows.isAll)
            throw base::SolverError(
                "the full path streamed keeps the grid point forces of the declared cuts' points alone, not of every "
                "point (its regions are gone after the solve); declare the cuts, or make "
                "the basis on the full or condensed path with every point's grid point forces");
        fixedRows =
            BasisFixedRows::plan(inputs.basisFixedRows, dofMap, system.gm(), system.isConstrained(), columns, budget);
        fixedRows->loads = front.loadsAt(fixedRows->dofs.span(), budget);
        isForceNode = base::BudgetedBuffer<std::uint8_t>(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                         "streamed basis force nodes");
        for (const std::int32_t node : fixedRows->nodes.span())
            isForceNode[static_cast<std::size_t>(node)] = 1;
        for (const std::int32_t node : fixedRows->mNodes.span())
            isForceNode[static_cast<std::size_t>(node)] = 1;
    }

    // The memory plan: the largest region bound whose solve fits beside what the budget holds and the outputs.
    // The chunk at most the whole g set's displacements (a small model's outputs one chunk).
    const std::int64_t chunkValues =
        options.chunkValues > 0
            ? options.chunkValues
            : std::min(StreamedOutputs::chunkValuesFor(DEFAULT_CHUNK_BYTES, columns,
                                                       inputs.keepsLoadBasis ? columns : inputs.cases.size()),
                       std::max<std::int64_t>(dofMap.gSetSize(), 6) *
                           std::max<std::int64_t>(static_cast<std::int64_t>(columns), 1));
    // The load basis: its vectors are the chunk's cases; it keeps their declared quantities and, at the
    // fixed rows' points, their grid point forces (about BASIS_ROWS_PER_POINT element rows a point, each held as a
    // contribution while the chunks come and as a table row after).
    constexpr std::uint64_t BASIS_ROWS_PER_POINT = 6;
    const std::size_t chunkCases = inputs.keepsLoadBasis ? columns : inputs.cases.size();
    const std::uint64_t basisBytes =
        !inputs.keepsLoadBasis
            ? 0
            : static_cast<std::uint64_t>(columns) *
                  ((inputs.basisQuantities != nullptr ? inputs.basisQuantities->quantityCount() : 0) * sizeof(double) +
                   (fixedRows ? fixedRows->nodes.size() : 0) * BASIS_ROWS_PER_POINT *
                       (sizeof(output::StiffnessRecovery::PointContribution) + 3 * sizeof(std::int32_t) +
                        6 * sizeof(double)));
    const std::uint64_t outputBytes = StreamedOutputs::chunkBytes(chunkValues, columns, chunkCases) +
                                      3 * static_cast<std::uint64_t>(dofMap.nodeCount()) * sizeof(std::int32_t) +
                                      inputs.resultBytes + basisBytes;
    const std::uint64_t heldBefore = budget.used();
    for (const base::MemoryConsumer& consumer : budget.largestConsumers(HELD_CONSUMERS))
        record.heldConsumers.emplace_back(consumer.label, consumer.bytes);
    const std::uint64_t marginBytes =
        std::min(PLAN_MARGIN_BYTES, PLAN_MARGIN_BASE_BYTES + PLAN_MARGIN_ELEMENT_BYTES *
                                                                 static_cast<std::uint64_t>(elements.elementCount()));
    const linalg::FrontStreamEstimate whole =
        linalg::StreamedSolve::estimate(symbolic, threads, columns, std::numeric_limits<std::int64_t>::max() / 4);
    const auto bytesWith = [&](const linalg::FrontStreamEstimate& estimate)
    {
        return static_cast<std::uint64_t>(PLAN_MARGIN * static_cast<double>(estimate.bytes() + outputBytes +
                                                                            (refines ? refinementBytes : 0))) +
               marginBytes;
    };
    // The partial factor retention: at a bound that fits, the regions retained within what the plan leaves beside it
    // (left / PLAN_MARGIN, capped by StreamedOptions::retainedBytes), the estimate made again with them; the arrays of
    // the retained slots come out of the same memory, so the budget is lowered by any excess until it fits (each step
    // retains less; a few steps at most, then none).
    constexpr int RETENTION_FIT_STEPS = 4;
    const bool retains = options.retention != StreamedRetention::None &&
                         !(options.retention == StreamedRetention::Given && options.retainedBytes == 0);
    record.retention = std::string(nameOf(retains ? options.retention : StreamedRetention::None));
    const auto withRetention = [&](const linalg::FrontStreamEstimate& plain, std::int64_t bound,
                                   linalg::FrontStreamOrder tried, std::uint64_t& retentionBudget)
    {
        retentionBudget = 0;
        const std::uint64_t needed = bytesWith(plain);
        if (!retains || plain.regions.regionRunCount == 0 || needed >= budget.available())
            return plain;
        std::uint64_t retainedBytes =
            static_cast<std::uint64_t>(static_cast<double>(budget.available() - needed) / PLAN_MARGIN);
        if (options.retention == StreamedRetention::Given)
            retainedBytes = std::min(retainedBytes, options.retainedBytes);
        retentionBudget = retainedBytes;
        for (int step = 0; step < RETENTION_FIT_STEPS && retainedBytes >= VALUE_BYTES; ++step)
        {
            const linalg::FrontStreamEstimate retained = linalg::StreamedSolve::estimate(
                symbolic, threads, columns, bound, tried, linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS,
                static_cast<std::int64_t>(retainedBytes / VALUE_BYTES));
            const std::uint64_t retainedNeed = bytesWith(retained);
            if (retainedNeed <= budget.available())
                return retained;
            const std::uint64_t excess = retainedNeed - budget.available();
            retainedBytes -= std::min(
                retainedBytes, static_cast<std::uint64_t>(static_cast<double>(excess) * PLAN_MARGIN) + VALUE_BYTES);
        }
        return plain;
    };
    // The orders tried: the subtrees on the threads (the fastest), then one postorder (the least working storage);
    // for each, the bounds from every panel kept down to one panel a region, each that fits with its retention. The
    // least estimated time is taken (estimatedTime: the factorisation and the regions' work again, the work in
    // one postorder slower; taking the parallel order at any bound that fits would choose it even where its small
    // bounds recompute many factorisations and the postorder at the same budget few); ties: the parallel
    // order, the larger bound.
    std::optional<linalg::FrontStreamEstimate> chosen;
    std::optional<linalg::FrontStreamEstimate> least;
    double chosenTime = 0.0;
    linalg::FrontStreamOrder order = linalg::FrontStreamOrder::Parallel;
    for (const linalg::FrontStreamOrder tried :
         {linalg::FrontStreamOrder::Parallel, linalg::FrontStreamOrder::Postorder})
    {
        for (const std::int64_t bound : candidateBounds(whole))
        {
            if (options.regionBytes && static_cast<std::uint64_t>(bound) * VALUE_BYTES > *options.regionBytes &&
                bound != 1)
                continue;
            const linalg::FrontStreamEstimate estimate =
                linalg::StreamedSolve::estimate(symbolic, threads, columns, bound, tried);
            least = estimate;
            if (bytesWith(estimate) > budget.available())
                continue;
            // Nothing beats the factorisation once on the threads.
            if (chosen && chosen->regions.recomputedWork == 0.0 && chosen->regions.serialUpwardWork == 0.0)
                continue;
            std::uint64_t retentionBudget = 0;
            const linalg::FrontStreamEstimate candidate = withRetention(estimate, bound, tried, retentionBudget);
            const double time = estimatedTime(candidate.regions, threads);
            if (!chosen || time < chosenTime)
            {
                chosen = candidate;
                chosenTime = time;
                order = tried;
                record.retentionBudgetBytes = retentionBudget;
            }
        }
        if (tried == linalg::FrontStreamOrder::Parallel)
            record.leastParallelPeakBytes = heldBefore + bytesWith(*least);
    }
    record.heldBeforeBytes = heldBefore;
    record.flops = whole.flopCount;
    record.leastPeakBytes = heldBefore + bytesWith(*least);
    record.leastWorkspaceBytes = static_cast<std::uint64_t>(least->workspaceValues) * VALUE_BYTES;
    record.leastPeakStackBytes = static_cast<std::uint64_t>(least->peakStackValues) * VALUE_BYTES;
    record.leastPeakMadeBytes = static_cast<std::uint64_t>(least->peakMadeValues) * VALUE_BYTES;
    if (least->peakSupernode >= 0)
    {
        record.leastPeakColumns = symbolic.columnCount(least->peakSupernode);
        record.leastPeakRows = symbolic.rowCount(least->peakSupernode);
    }
    // The plan alone (the pre-flight check, the automatic path choice) is made at the budget as a run makes it; when
    // nothing fits it records the least (its peak above the budget says so) instead of refusing.
    if (!chosen && options.isMemoryPlanOnly)
    {
        chosen = least;
        order = linalg::FrontStreamOrder::Postorder;
    }
    record.order = order == linalg::FrontStreamOrder::Parallel ? "parallel" : "postorder";
    if (!chosen)
    {
        const std::uint64_t needed = bytesWith(*least);
        std::vector<base::MemoryConsumer> consumers{
            {"the fronts and update matrices with their right-hand sides", least->workspaceValues * VALUE_BYTES},
            {"the panels of the least region", static_cast<std::uint64_t>(least->keptValues) * VALUE_BYTES},
            {"the solutions of the downward pass", static_cast<std::uint64_t>(least->solutionValues) * VALUE_BYTES},
            {"the outputs (a chunk and the result tables)", outputBytes},
            {"the refinement of the relieved solution (its solutions held whole)", refines ? refinementBytes : 0},
            {"the per-supernode arrays", least->arrayBytes},
            {"held before the solve (the model's equations, ordering, placement, loads)", heldBefore}};
        std::sort(consumers.begin(), consumers.end(),
                  [](const base::MemoryConsumer& a, const base::MemoryConsumer& b) { return a.bytes > b.bytes; });
        throw base::MemoryBudgetError(
            base::concatenate({"the full path streamed needs at least ", std::to_string(heldBefore + needed),
                               " bytes (its least region bound), the budget has ", std::to_string(budget.limit()),
                               "; give more memory, fewer cases per constraint set or outputs, or a smaller model"}),
            base::MemoryShortfall{needed, budget.available(), budget.limit(), std::move(consumers)});
    }
    record.regionValues = chosen->regions.regionValues;
    record.regionRuns = chosen->regions.regionRunCount;
    record.recomputedFlops = chosen->regions.recomputedFlops;
    record.retainedValues = chosen->retainedValues;
    record.retainedRegions = chosen->regions.retainedRunCount;
    record.retainedFlops = chosen->regions.retainedFlops;
    record.recomputedWork = chosen->regions.recomputedWork;
    record.upwardWork = chosen->regions.upwardWork;
    record.serialWork = chosen->regions.serialUpwardWork + chosen->regions.serialRunWork;
    record.estimatedWork = estimatedTime(chosen->regions, threads);
    record.retainedWork = chosen->regions.retainedWork;
    for (std::size_t step = 1; step <= REGION_LADDER_STEPS; ++step)
    {
        const std::int64_t bound = std::max<std::int64_t>(record.factorValues >> step, 1);
        const linalg::RegionSweepCost cost =
            linalg::StreamedSolve::estimate(symbolic, threads, columns, bound, order).regions;
        record.regionLadder.push_back({bound, cost.keptValues, cost.regionRunCount, cost.recomputedFlops});
    }
    // The retention ladder at the chosen bound: the values below the top are the regions' (a partition of them).
    {
        const linalg::FrontStreamEstimate plain =
            linalg::StreamedSolve::estimate(symbolic, threads, columns, chosen->regions.regionValues, order);
        const linalg::FrontStreamEstimate all = linalg::StreamedSolve::estimate(
            symbolic, threads, columns, chosen->regions.regionValues, order,
            linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS, std::numeric_limits<std::int64_t>::max() / 4);
        record.belowTopValues = all.regions.retainedValues;
        for (std::size_t step = 0; step < RETENTION_LADDER_STEPS; ++step)
        {
            // 0, 1/8, 1/4, 1/2, all.
            const std::int64_t budgetValues =
                step == 0 ? 0 : record.belowTopValues >> (RETENTION_LADDER_STEPS - 1 - step);
            const linalg::RegionSweepCost cost =
                step == 0
                    ? plain.regions
                    : linalg::StreamedSolve::estimate(symbolic, threads, columns, chosen->regions.regionValues, order,
                                                      linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS, budgetValues)
                          .regions;
            record.retentionLadder.push_back(
                {budgetValues, cost.retainedValues, cost.retainedRunCount, cost.recomputedFlops, cost.recomputedWork});
        }
    }
    record.keptValues = chosen->keptValues;
    record.workspaceBytes = static_cast<std::uint64_t>(chosen->workspaceValues) * VALUE_BYTES;
    record.packingBytes = static_cast<std::uint64_t>(chosen->packingValues) * VALUE_BYTES;
    record.keptBytes = static_cast<std::uint64_t>(chosen->keptValues) * VALUE_BYTES;
    record.retainedBytes = static_cast<std::uint64_t>(chosen->retainedValues) * VALUE_BYTES;
    record.solutionBytes = static_cast<std::uint64_t>(chosen->solutionValues) * VALUE_BYTES;
    record.arrayBytes = chosen->arrayBytes;
    record.outputBytes = outputBytes;
    record.estimatedBytes = bytesWith(*chosen);
    record.estimatedPeakBytes = heldBefore + record.estimatedBytes;
    record.pressureRows = loads.pressureRowCount();
    record.planSeconds = watch.lap();
    if (options.isMemoryPlanOnly)
    {
        result.nodal.resize(inputs.cases.size());
        result.element.resize(inputs.cases.size());
        return result;
    }

    // The envelopes' rows: the whole tables of the plan at rest (EnvelopeAccumulator::defineRows), so that the regions'
    // rows take their places.
    std::vector<base::EntityId> labels; // bounded by the set
    for (const std::size_t c : inputs.cases)
        labels.push_back(inputs.subcases[c].label);
    if (inputs.envelopes != nullptr && !inputs.envelopes->isEmpty())
    {
        base::BudgetedBuffer<double> zeros(budget, static_cast<std::size_t>(dofMap.gSetSize()),
                                           "streamed tables at rest");
        if (displacement.isRequested && inputs.envelopes->takes("displacements"))
            inputs.envelopes->defineRows(inputs.nodalTables->build("displacements", labels.front(), zeros.span(),
                                                                   budget, displacement.isAll, displacement.members));
        if (selection.stress.isRequested || selection.strain.isRequested || selection.forces.isRequested)
        {
            std::vector<output::ElementResultTable> atRest;
            inputs.elementTables->build(labels.front(), zeros.span(), {}, budget, atRest, inputs.pool,
                                        inputs.cancellation, nullptr, &selection);
            for (const output::ElementResultTable& table : atRest)
                inputs.envelopes->defineRows(table);
        }
    }

    // The solve, the outputs region by region.
    StreamedOutputs::Inputs outputInputs;
    outputInputs.elements = &elements;
    outputInputs.builder = inputs.elementTables;
    outputInputs.coefficients = &loadCases.coefficients();
    outputInputs.caseLabels = labels;
    // The load basis: the vectors as the cases (α the identity), numbered as the units.
    linalg::DenseMatrix identity(budget, inputs.keepsLoadBasis ? columns : 0, inputs.keepsLoadBasis ? columns : 0,
                                 "streamed load basis identity");
    std::vector<base::EntityId> unitLabels; // bounded by the basis
    if (inputs.keepsLoadBasis)
    {
        for (std::size_t b = 0; b < columns; ++b)
        {
            identity(b, b) = 1.0;
            unitLabels.push_back(inputs.firstUnit + static_cast<base::EntityId>(b));
        }
        outputInputs.coefficients = &identity;
        outputInputs.caseLabels = unitLabels;
        outputInputs.basisQuantities = inputs.basisQuantities;
        outputInputs.basisSelection = inputs.basisSelection;
        outputInputs.isForceNode = isForceNode.span();
        outputInputs.fixedRows = fixedRows.get();
    }
    outputInputs.plan = &plan;
    outputInputs.envelopes = inputs.envelopes;
    outputInputs.keepsCaseTables = inputs.keepsCaseTables;
    if (hasRelief)
    {
        outputInputs.supports = rSet;
        outputInputs.supportCoefficients = supportCoefficients;
        outputInputs.gaugeRowOf = [&front](base::DofIndex dof) { return front.reliefRowOf(dof); };
    }
    outputInputs.chunkValues = chunkValues;
    outputInputs.pool = inputs.pool;
    outputInputs.cancellation = inputs.cancellation;
    StreamedOutputs outputs(outputInputs, budget);
    linalg::FactorOptions factorOptions;
    factorOptions.pivoting = linalg::PivotingMode::PositiveDefinite;
    factorOptions.pivotPolicy = inputs.pivotPolicy;
    factorOptions.equationNamer = [&](std::int32_t equation)
    { return equationName(dofMap, equations.dofOfEquation(equation)); };
    factorOptions.cancellation = inputs.cancellation;
    linalg::StreamedSolveResult solved;
    if (!refines)
        solved = linalg::StreamedSolve::solve(symbolic, front, outputs, columns, *inputs.kernels, factorOptions, budget,
                                              chosen->regions.regionValues, order,
                                              linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS, chosen->retainedValues);
    else
    {
        // The refined solution (linalg::IterativeRefinement): the first solve held whole, then each step the
        // residual P'_l − K·û (û the solution in the gauge M-orthogonal to the rigid motions, K·û element by element)
        // solved by another streamed solve on a front source that subtracts K·û, the correction added; then the
        // solution handed to the outputs supernode by supernode, top-down, the FreeBodySupport DOFs' values given
        // first.
        const auto equationCount = static_cast<std::size_t>(symbolic.dofCount());
        linalg::DenseMatrix solution(budget, equationCount, columns, "streamed refined displacements");
        linalg::DenseMatrix stiffnessProducts(budget, equationCount, columns, "streamed refinement K u");
        linalg::DenseMatrix correction(budget, equationCount, columns, "streamed refinement correction");
        StoredSolution firstStore(symbolic, solution.view());
        solved = linalg::StreamedSolve::solve(symbolic, front, firstStore, columns, *inputs.kernels, factorOptions,
                                              budget, chosen->regions.regionValues, order,
                                              linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS, chosen->retainedValues);
        linalg::IterativeRefinement refinement;
        while (true)
        {
            const std::vector<std::array<double, 6>> sums =
                front.rigidMotionSums(solution.view(), inputs.pool, inputs.cancellation);
            std::vector<std::array<double, 6>> gauge(columns); // per vector: (Φᵀ·M·Φ)⁻¹·Φᵀ·M·u
            for (std::size_t b = 0; b < columns; ++b)
                gauge[b] = solveSix(rigidBodyMass, sums[b]).value_or(std::array<double, 6>{});
            front.multiplyStiffness(solution.view(), gauge, stiffnessProducts.view(), inputs.pool, inputs.cancellation);
            StreamedFront::Inputs residualInputs = frontInputs;
            residualInputs.subtractedLoads = stiffnessProducts.view();
            StreamedFront residualFront(residualInputs);
            StoredSolution store(symbolic, correction.view());
            static_cast<void>(linalg::StreamedSolve::solve(symbolic, residualFront, store, columns, *inputs.kernels,
                                                           factorOptions, budget, chosen->regions.regionValues, order,
                                                           linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS,
                                                           chosen->retainedValues));
            const linalg::IterativeRefinement::Step step =
                refinement.record(linalg::IterativeRefinement::largestRelative(correction.view(), solution.view()));
            if (step.isAccepted)
                for (std::size_t b = 0; b < columns; ++b)
                    for (std::size_t e = 0; e < equationCount; ++e)
                        solution(e, b) += correction(e, b);
            if (!step.continues)
                break;
        }
        record.refinementCorrections = refinement.corrections();
        linalg::DenseMatrix atSupports(budget, rSet.size(), columns, "streamed refined SUPORT displacements");
        for (std::size_t j = 0; j < rSet.size(); ++j)
        {
            const std::int32_t equation = equations.equationOfDof(rSet[j]);
            for (std::size_t b = 0; b < columns && equation >= 0; ++b)
                atSupports(j, b) = solution(static_cast<std::size_t>(equation), b);
        }
        outputs.presetSupportValues(atSupports.view());
        handOver(symbolic, solution.view(), outputs, budget);
    }
    outputs.finish();
    result.diagnostics = solved.diagnostics;
    record.upwardSeconds = solved.regions.upwardSeconds;
    record.regionSeconds = solved.regions.regionSeconds;
    record.downwardSeconds = solved.regions.downwardSeconds;
    record.taskModel = solved.regions.taskModel;
    record.largestPivotRatio = solved.diagnostics.largestRatio();
    record.pivotIssueCount = solved.diagnostics.issueCount();
    record.chunks = outputs.chunkCount();
    record.largestChunkValues = outputs.largestChunkValues();
    record.measuredPeakBytes = budget.peak();
    reportKeptPivots(solved.diagnostics, factorOptions.equationNamer, first.label, sink);
    if (inputs.keepsLoadBasis)
    {
        result.basis = outputs.takeBasis();
        result.loadCases = ownedCases;
        result.fixedRows = fixedRows;
        result.nodal.resize(inputs.cases.size());
        result.element.resize(inputs.cases.size());
    }
    else
        for (std::size_t c = 0; c < inputs.cases.size(); ++c)
        {
            result.nodal.push_back(std::move(outputs.nodalTables(c)));
            result.element.push_back(std::move(outputs.elementTables(c)));
        }
    record.solveSeconds = watch.lap();
    return result;
}

} // namespace oos::solutions
