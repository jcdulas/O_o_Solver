#include "oos/solutions/common/AnalysisModel.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/elements/shells/ShellDirectors.hpp"
#include "oos/solutions/common/RunChecks.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

// The cards of the point masses: they hold no rotation.
[[nodiscard]] bool isMassCard(std::string_view card)
{
    return card.starts_with("PointMass") || card.starts_with("MassDof") || card.starts_with("MassScalar");
}

// The grid point rows something other than a shell with directors (elements::DIRECTOR_SHELL_CARDS: ShellQuad4,
// ShellTria3, ShellQuad8, ShellTria6) or a mass is attached to: another element, a rigid element, a multipoint or a
// single-point constraint. Nothing when a batch of elements does not give the rows
// of its grid points (the notice below is then not made).
[[nodiscard]] std::optional<std::vector<std::uint8_t>> otherwiseHeldGridPoints(const elements::SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    std::vector<std::uint8_t> isHeld(static_cast<std::size_t>(snapshot.nodes().rowCount()), 0);
    const auto holdId = [&](std::int32_t id)
    {
        if (const std::optional<std::int32_t> row = index.points().findGrid(id))
            isHeld[static_cast<std::size_t>(*row)] = 1;
    };
    for (const input::ElementBatch& batch : snapshot.elementBatches())
    {
        if (elements::isDirectorShellCard(batch.card()) || isMassCard(batch.card()))
            continue;
        if (!batch.hasNodes())
            return std::nullopt;
        const input::ConnectivityView nodes = batch.nodes();
        for (std::int64_t row = 0; row < nodes.size(); ++row)
            for (const std::int32_t id : nodes[row])
                holdId(id);
    }
    if (snapshot.hasTable(input::TableRole::RigidElements))
        for (const std::int32_t id : snapshot.rigidElements().nodeIds())
            holdId(id);
    if (snapshot.hasTable(input::TableRole::Spc))
        for (const std::int32_t id : snapshot.constraints().spcNodeIds())
            holdId(id);
    if (snapshot.hasTable(input::TableRole::Mpc))
        for (const std::int32_t id : snapshot.constraints().mpcNodeIds())
            holdId(id);
    return isHeld;
}

// The rotation of a grid point about the normal of the shells with directors on one smooth side is held by their
// drilling term alone (elements::ShellDirectors), a stabilisation. A PointMassInertia with an inertia or an
// offset on such a grid point, which nothing else holds, turns about that normal in a mode of its own whose frequency
// depends on the mesh. A modal analysis is told so once, with the masses concerned.
void reportInertiaOnShellNormal(const elements::SnapshotIndex& index, AnalysisKind analysis,
                                base::DiagnosticSink& sink)
{
    constexpr std::size_t NAMED_MASSES = 5;
    if (analysis != AnalysisKind::Modes)
        return;
    const input::ModelSnapshot& snapshot = index.snapshot();
    const input::ElementBatch* masses = snapshot.findElementBatch("PointMassInertia");
    if (masses == nullptr || masses->rowCount() == 0)
        return;
    const elements::ShellDirectors& directors = elements::shellDirectors(index);
    if (!(directors.drillingFactor() > 0.0))
        return; // the automatic constraint of singular DOFs removes the rotation: the inertia about it has no effect
    const std::optional<std::vector<std::uint8_t>> isHeld = otherwiseHeldGridPoints(index);
    if (!isHeld)
        return;
    const std::span<const std::int32_t> ids = masses->ids();
    const std::span<const std::int32_t> nodes = masses->reader().values<std::int32_t>("node");
    const base::FixedListView<double> offsets = masses->reader().fixedLists<double>("offset", base::NullPolicy::Allow);
    const base::FixedListView<double> inertias =
        masses->reader().fixedLists<double>("inertia", base::NullPolicy::Allow);
    const auto hasValue = [](std::span<const double> values)
    { return std::any_of(values.begin(), values.end(), [](double v) { return std::isfinite(v) && v != 0.0; }); };
    std::size_t count = 0;
    std::string named;
    for (std::int64_t row = 0; row < masses->rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        if (!hasValue(offsets[row]) && !hasValue(inertias[row]))
            continue;
        const std::optional<std::int32_t> gridRow = index.points().findGrid(nodes[r]);
        if (!gridRow || (*isHeld)[static_cast<std::size_t>(*gridRow)] != 0 || !directors.hasOneSmoothSide(*gridRow))
            continue;
        if (count < NAMED_MASSES)
            named += (count == 0 ? "" : ", ") + std::to_string(ids[r]);
        ++count;
    }
    if (count == 0)
        return;
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Information;
    diagnostic.code = "inertia-on-shell-normal";
    diagnostic.card = "PointMassInertia";
    diagnostic.field = "inertia";
    diagnostic.message = base::concatenate(
        {std::to_string(count), " PointMassInertia (", named, count > NAMED_MASSES ? ", ..." : "",
         ") put an inertia on a grid point that ShellQuad4, ShellTria3, ShellQuad8 or ShellTria6 alone hold: the "
         "rotation about the normal of these shells has their drilling stabilisation for only stiffness (PARAM "
         "ShellDrillingStiffness), so expect a "
         "mode of each mass turning about that normal, at a frequency that depends on the mesh; attach the mass to the "
         "neighbouring grid points with a rigid or an interpolation element, or leave out its inertia about the "
         "normal"});
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
    else
    {
        reportInertiaOnShellNormal(*m_index, analysis, sink);
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
