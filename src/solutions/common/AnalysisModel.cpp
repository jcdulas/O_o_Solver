#include "oos/solutions/common/AnalysisModel.hpp"

#include "oos/base/threading/Stopwatch.hpp"
#include "oos/solutions/common/RunChecks.hpp"

namespace oos::solutions
{

AnalysisModel::AnalysisModel(const input::ModelSnapshot& snapshot, AnalysisKind analysis, base::MemoryBudget& budget,
                             base::ThreadPool* pool, const base::CancellationToken* cancellation,
                             base::DiagnosticSink& sink, const AnalysisModelOptions& options)
    : m_snapshot(&snapshot), m_registry(elements::ElementRegistry::forModel(snapshot))
{
    base::Stopwatch watch;
    reportSupport(snapshot, m_registry, analysis, sink);
    m_index = std::make_shared<elements::SnapshotIndex>(snapshot, budget);
    m_elementSet = std::make_unique<assembly::ElementSet>(*m_index, m_registry);
    m_elementSet->check(sink, pool);
    if (options.checkCases)
        options.checkCases(sink);
    stopOnErrors(sink, "the assembly");
    m_timings.checks = watch.lap();
    m_dofMap = std::make_shared<assembly::DofMap>(snapshot.nodes(), m_index->points());
    if (options.replacedElements)
    {
        std::vector<base::EntityId> replaced = options.replacedElements(*m_elementSet, *m_dofMap);
        if (!replaced.empty())
        {
            m_replaced = std::make_unique<reduction::ReplacedElements>(m_registry, std::move(replaced));
            m_elementSet = std::make_unique<assembly::ElementSet>(*m_index, m_replaced->registry());
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
