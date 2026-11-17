#include "oos/solutions/common/AnalysisModel.hpp"

#include "oos/base/threading/Stopwatch.hpp"
#include "oos/solutions/common/RunChecks.hpp"

#include <stdexcept>

namespace oos::solutions
{

namespace
{

// The mixed hybrid formulation has the consistent mass only: a lumped form of its hierarchical and hybrid unknowns
// has no meaning (docs/mixed-hybrid-formulation-spec.md section 8). An analysis that uses the mass while CoupledMass
// asks for the lumped one (its default) is told so once.
void reportConsistentMass(const input::ModelSnapshot& snapshot, AnalysisKind analysis, base::DiagnosticSink& sink)
{
    if (analysis != AnalysisKind::Modes && analysis != AnalysisKind::StaticAeroelastic)
        return;
    const input::ParamTable* params = snapshot.hasTable(input::TableRole::Params) ? &snapshot.params() : nullptr;
    if (params != nullptr)
    {
        if (const std::optional<input::ParamValue> value = params->find("CoupledMass"))
        {
            const double coupled = value->kind == input::ParamValue::Kind::Integer ? value->integer : value->real;
            if (value->kind != input::ParamValue::Kind::Text && coupled > 0.0)
                return;
        }
    }
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Information;
    diagnostic.code = "consistent-mass";
    diagnostic.card = "Parameter CoupledMass";
    diagnostic.field = "CoupledMass";
    diagnostic.message =
        "the elements of MixedHybridFormulation use their consistent mass whatever CoupledMass asks: a lumped mass of "
        "their unknowns on edges, faces and interiors has no meaning";
    sink.report(std::move(diagnostic));
}

} // namespace

AnalysisModel::AnalysisModel(const input::ModelSnapshot& snapshot, AnalysisKind analysis, base::MemoryBudget& budget,
                             base::ThreadPool* pool, const base::CancellationToken* cancellation,
                             base::DiagnosticSink& sink, const AnalysisModelOptions& options)
    : m_snapshot(&snapshot), m_registry(elements::ElementRegistry::forModel(snapshot))
{
    base::Stopwatch watch;
    reportSupport(snapshot, m_registry, analysis, sink);
    if (options.index != nullptr && &options.index->snapshot() != &snapshot)
        throw std::invalid_argument(
            "AnalysisModel: the index given indexes another snapshot; give the index of a model of the same snapshot");
    m_index =
        options.index != nullptr ? options.index : std::make_shared<const elements::SnapshotIndex>(snapshot, budget);
    // A formulation with unknowns on edges, faces and interiors: the entities of this kind of analysis, which the
    // element views carry and the DOF map numbers.
    m_layoutKind =
        analysis == AnalysisKind::Modes ? elements::EntityLayout::Kind::Modal : elements::EntityLayout::Kind::Static;
    m_formulation = elements::formulationModelOf(*m_index, m_registry);
    const elements::EntityLayout* entities = nullptr;
    if (m_formulation != nullptr)
    {
        m_formulation->reportProblems(sink);
        entities = &m_formulation->layout(m_layoutKind);
        reportConsistentMass(snapshot, analysis, sink);
    }
    m_elementSet = std::make_unique<assembly::ElementSet>(*m_index, m_registry, entities);
    m_elementSet->check(sink, pool);
    if (options.checkCases)
        options.checkCases(sink);
    stopOnErrors(sink, "the assembly");
    m_timings.checks = watch.lap();
    m_dofMap = std::make_shared<assembly::DofMap>(snapshot.nodes(), m_index->points(), m_formulation, m_layoutKind);
    if (options.replacedElements)
    {
        std::vector<base::EntityId> replaced = options.replacedElements(*m_elementSet, *m_dofMap);
        if (!replaced.empty())
        {
            m_replaced = std::make_unique<reduction::ReplacedElements>(m_registry, std::move(replaced));
            m_elementSet = std::make_unique<assembly::ElementSet>(*m_index, m_replaced->registry(), entities);
        }
    }
    if (options.checkBeforeConnectivity)
        options.checkBeforeConnectivity(*m_elementSet, *m_dofMap);
    assembly::ElementMatrixStore* store = nullptr;
    if (options.keptStiffnessBudgetFraction > 0.0)
    {
        m_keptStiffness = assembly::ElementMatrixStore(
            budget,
            static_cast<std::uint64_t>(options.keptStiffnessBudgetFraction * static_cast<double>(budget.available())));
        store = &m_keptStiffness;
    }
    m_connectivity =
        std::make_unique<assembly::ElementConnectivity>(*m_elementSet, *m_dofMap, budget, pool, cancellation, store);
    m_nodeDofCounts = m_dofMap->nodeDofCounts(budget);
    m_params = snapshot.hasTable(input::TableRole::Params) ? &snapshot.params() : nullptr;
    m_timings.connectivity = watch.lap();
}

} // namespace oos::solutions
