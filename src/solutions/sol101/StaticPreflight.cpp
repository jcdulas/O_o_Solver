#include "oos/solutions/sol101/StaticPreflight.hpp"

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/MsetCouplings.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/envelopes/EnvelopeAccumulator.hpp"
#include "oos/output/volume/OutputVolumeEstimate.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <bit>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>

namespace oos::solutions
{

namespace
{

// The bytes of one double value.
constexpr std::uint64_t VALUE_BYTES = sizeof(double);
// Kgg, Knn and the scalar copy of Knn the factorisation reads (SymmetricCsc) are held together (StaticSolution step 4).
constexpr std::uint64_t STIFFNESS_COPIES = 3;

// The couplings the elimination of the m set adds to the element connectivity: the rigid elements,
// the constraints of the fasteners' piercing points and the MPC equations of the first solved subcase, as the full
// path's Knn holds them (ConstrainedSystem: MsetCouplings). Without them an assembly joined by RigidInterpolation,
// RigidBodyStar, Fastener or MPC is, for the ordering, a set of separate parts: its factor was estimated 2.4 times
// too small and its flops 26 times on the 3 000-plate assembly of 3.1 M DOFs (9.37 GB of panels needed where 3.9 GB
// were estimated). Nothing when the relations cannot be built (a problem of the model the run reports: the element
// connectivity alone is then the estimate).
std::unique_ptr<assembly::MsetCouplings> relationCouplings(const input::ModelSnapshot& snapshot,
                                                           const assembly::DofMap& dofMap,
                                                           const assembly::ElementConnectivity& connectivity,
                                                           std::optional<base::EntityId> mpcSet,
                                                           base::MemoryBudget& budget)
{
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* piercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    if (rigidElements == nullptr && piercings == nullptr && !mpcSet)
        return nullptr;
    base::DiagnosticSink sink; // the relations' problems are the run's to report
    try
    {
        const assembly::MpcEquations mpc(constraints, dofMap, mpcSet, budget, sink);
        const assembly::FastenerConstraints fasteners(piercings, dofMap, budget, sink);
        const assembly::RigidElimination gm(rigidElements, &mpc, dofMap, budget, sink, &fasteners);
        if (sink.hasErrors() || gm.rowCount() == 0)
            return nullptr;
        return std::make_unique<assembly::MsetCouplings>(connectivity.groups(), gm, dofMap, budget);
    }
    catch (const std::invalid_argument&)
    {
        return nullptr;
    }
}

// The factorisation estimate of the full path: the ordering and the symbolic analysis of the element connectivity
// with the couplings of the m set's elimination (relationCouplings), every node with the DOFs an element's stiffness
// has (ElementConnectivity::stiffenedComponents: a grid point of solids alone its three translations, its rotations
// held by AUTOSPC in the analysis; with all six the factor of a SolidHexa block was estimated four
// times its entries, 208 GB at 512 000 SolidHexa where the analysis's is 33 GB), and every DOF of a point a relation
// reaches (no constraint removed, the dependent DOFs kept: an upper bound of the f set).
void estimateFactor(const input::ModelSnapshot& snapshot, const assembly::ElementSet& set,
                    const assembly::DofMap& dofMap, std::optional<base::EntityId> mpcSet, const StaticOptions& options,
                    base::MemoryBudget& budget, StaticPreflightReport& report)
{
    const base::CancellationToken* cancellation = options.cancellation;
    const assembly::ElementConnectivity connectivity(set, dofMap, budget, options.pool, cancellation);
    const std::unique_ptr<assembly::MsetCouplings> couplings =
        relationCouplings(snapshot, dofMap, connectivity, mpcSet, budget);
    std::vector<linalg::NodeGroupList> groupLists{connectivity.groups()};
    if (couplings)
        groupLists.push_back(couplings->groups());
    const linalg::NodeGraph graph(dofMap.nodeCount(), groupLists, budget, options.pool);
    const base::BudgetedBuffer<std::uint8_t> dofCounts = dofMap.nodeDofCounts(budget);
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    base::BudgetedBuffer<std::int32_t> weights(budget, nodeCount, "pre-flight node weights");
    base::BudgetedBuffer<base::Vec3> coordinates(budget, nodeCount, "pre-flight node positions");
    // The points the relations reach keep all of their DOFs (a relation may give a rotation its stiffness).
    base::BudgetedBuffer<std::uint8_t> isRelated(budget, nodeCount, "pre-flight related points");
    if (couplings)
        for (const std::int32_t node : couplings->groups().nodes)
            isRelated[static_cast<std::size_t>(node)] = 1;
    const std::span<const std::uint8_t> stiffened = connectivity.stiffenedComponents();
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        const auto components = static_cast<std::int32_t>(std::popcount(static_cast<unsigned>(stiffened[node])));
        // At least one DOF: a point without stiffness (fully held by AUTOSPC) is kept in the graph.
        weights[node] = isRelated[node] != 0 || components == 0 ? dofCounts[node] : components;
        // A scalar point has no position; the origin is enough for an estimate.
        const auto row = static_cast<std::int32_t>(node);
        coordinates[node] = row < dofMap.gridCount() ? dofMap.positionInBasic(row) : base::Vec3{};
    }
    const linalg::SparsePattern pattern(graph, dofCounts.span(), budget);
    report.stiffnessBytes = STIFFNESS_COPIES * VALUE_BYTES * static_cast<std::uint64_t>(pattern.valueCount());
    if (nodeCount == 0)
        return;
    const base::OperationControl control{cancellation, nullptr};
    const linalg::NodePermutation order =
        linalg::ConstrainedOrdering().order(graph, weights.span(), coordinates.span(), {},
                                            linalg::OrderingMethod::Automatic, budget, options.pool, control);
    const linalg::SymbolicFactor symbolic = linalg::SymbolicFactor::analyse(
        graph, weights.span(), order, 0, linalg::AmalgamationThresholds{}, budget, control);
    report.factorEquationCount = symbolic.dofCount();
    report.factorEntryCount = symbolic.factorEntryCount();
    report.factorFlopCount = symbolic.flopCount();
    report.factorMemoryBytes = symbolic.memoryPeakBytes();
}

} // namespace

StaticPreflightReport StaticPreflight::run(base::MemoryBudget& budget) const
{
    const input::ModelSnapshot& snapshot = *m_snapshot;
    StaticPreflightReport report;
    base::DiagnosticSink sink;

    // StaticSolution step 1, every problem kept.
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    report.support = checkSupport(snapshot, registry, m_options.analysis);
    const elements::SnapshotIndex index(snapshot, budget);
    const assembly::ElementSet elementSet(index, registry);
    elementSet.check(sink);
    // A buckling analysis also checks the differential stiffness of every element (as BucklingSolution: the types
    // without it, the offset elements under ModelParams OFFDEF ELMOFF).
    if (m_options.analysis == AnalysisKind::Buckling)
        for (const std::size_t b : elementSet.batchesWith(elements::Facet::Stiffness))
        {
            const assembly::AssemblyBatch& batch = elementSet.batches()[b];
            batch.family->checkDifferentialStiffness(batch.view, sink);
        }
    for (const assembly::AssemblyBatch& batch : elementSet.batches())
        report.elementCount += batch.view.rowCount();
    const std::vector<StaticSubcase> subcases = staticSubcasesOf(snapshot, m_options.subcases, sink);
    report.subcaseCount = static_cast<std::int64_t>(subcases.size());
    std::vector<output::OutputPlan> plans; // bounded by the subcases
    for (const StaticSubcase& subcase : subcases)
        plans.push_back(output::OutputPlan::forSubcase(snapshot, subcase.tableId, sink));
    // The glued contact of each subcase: its pairs listed, touching contact refused; a model whose contact tables
    // lack the columns glued contact needs is a problem here (the check never throws for the model).
    try
    {
        for (const StaticSubcase& subcase : subcases)
            for (assembly::GluedPair& pair : assembly::selectGluedPairs(snapshot, subcase.contact, subcase.label, sink))
                report.gluedPairs.push_back({subcase.label, std::move(pair)});
    }
    catch (const base::ContractError& error)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "contact-contract";
        diagnostic.card = "ContactSelection";
        diagnostic.message = error.what();
        sink.report(std::move(diagnostic));
    }
    report.problems = sink.diagnostics();

    const assembly::DofMap dofMap(snapshot.nodes(), index.points());
    report.gSetSize = dofMap.gSetSize();
    if (!report.support.isComplete() || sink.hasErrors())
        return report;

    // StaticSolution step 2: the output volume, with the same rules (an envelope may replace its kind's tables).
    output::OutputVolumeEstimate volume(elementSet, dofMap);
    const output::EnvelopeAccumulator envelopes(m_options.envelopes);
    const bool hasMset =
        snapshot.hasTable(input::TableRole::RigidElements) || snapshot.hasTable(input::TableRole::FastenerPiercings) ||
        (snapshot.hasTable(input::TableRole::Spc) && snapshot.constraints().mpcReader().rowCount() > 0) ||
        !report.gluedPairs.empty();
    for (const output::OutputPlan& plan : plans)
        volume.addCase(plan, hasMset,
                       [&](const std::string& kind) { return m_options.keepsCaseTables || !envelopes.takes(kind); });
    report.outputBytes = volume.totalBytes();
    for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
    {
        const auto kind = static_cast<output::OutputKind>(k);
        if (volume.bytesOf(kind) > 0)
            report.outputBytesByKind.emplace_back(std::string(output::tableKindOf(kind)), volume.bytesOf(kind));
    }

    if (m_isFactorEstimated)
    {
        std::optional<base::EntityId> mpcSet;
        for (const StaticSubcase& subcase : subcases)
            if (!subcase.isCombination)
            {
                mpcSet = subcase.mpc;
                break;
            }
        estimateFactor(snapshot, elementSet, dofMap, mpcSet, m_options, budget, report);
    }
    report.hasEstimates = true;
    return report;
}

} // namespace oos::solutions
