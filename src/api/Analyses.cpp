#include "oos/api/Analyses.hpp"

#include "AdaptiveShellStatics.hpp"
#include "oos/api/Threading.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/platform/MemoryInfo.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/FormulationModel.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"
#include "oos/reduction/external/ExternalUnit.hpp"
#include "oos/reduction/units/UnitSet.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/CondensationPlanner.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"
#include "oos/solutions/sol105/BucklingSolution.hpp"
#include "oos/solutions/superelements/SuperelementSolution.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string_view>

namespace oos::api
{

namespace
{

constexpr const char* PROGRAM_VERSION = "0.2.4";

solutions::AnalysisKind kindOf(Analysis analysis)
{
    switch (analysis)
    {
    case Analysis::Statics:
        return solutions::AnalysisKind::Statics;
    case Analysis::Modes:
        return solutions::AnalysisKind::Modes;
    case Analysis::Buckling:
        return solutions::AnalysisKind::Buckling;
    case Analysis::Aeroelastic:
        return solutions::AnalysisKind::StaticAeroelastic;
    }
    return solutions::AnalysisKind::Statics;
}

// The budget of the run: what the caller gives, or the memory left (in WebAssembly, what the module's memory may still
// grow to).
std::uint64_t budgetBytes(const AnalysisSettings& settings)
{
    if (settings.memoryBytes > 0)
        return settings.memoryBytes;
    const std::uint64_t available = base::MemoryInfo::query().availableBytes;
    return available > 0 ? available : base::MemoryBudget::UNLIMITED;
}

// The tables of O_o_Model a condensation plan reads: the components with their elements, the points of each
// component and the connections between them.
constexpr std::array<input::TableRole, 4> CONDENSATION_TABLES{
    input::TableRole::Components, input::TableRole::ComponentElements, input::TableRole::ConnectionNodes,
    input::TableRole::ConnectionClusters};

// The condensation plan of the settings for `analysis`, made on the snapshot (its own analysis model, released once the
// plan is made: the plan borrows nothing); null for the full path. Normal modes and buckling reduce their units by
// Craig-Bampton; `retained` are points the analysis keeps on the boundaries of their units beside the settings' (the
// structural points of the splines of a trim). Its diagnostics go to `diagnostics`; the index of the snapshot its model
// made to `index`, for the analysis to take (StaticOptions::snapshotIndex).
std::shared_ptr<const reduction::CondensationPlan> condensationPlanOf(
    const input::ModelSnapshot& snapshot, solutions::AnalysisKind analysis, const CondensationSettings& settings,
    std::span<const base::EntityId> retained, base::MemoryBudget& budget, base::ThreadPool* pool,
    std::vector<base::Diagnostic>& diagnostics, std::shared_ptr<const elements::SnapshotIndex>& index)
{
    reduction::PlanRequest request;
    switch (settings.units)
    {
    case CondensedUnits::None:
        return nullptr;
    case CondensedUnits::All:
        request.mode = reduction::CondenseMode::All;
        break;
    case CondensedUnits::Recommended:
        request.mode = reduction::CondenseMode::All;
        request.keepsFullWhereRecommended = true;
        break;
    case CondensedUnits::Listed:
        request.mode = reduction::CondenseMode::Listed;
        request.condensedUnits = settings.condensedUnits;
        break;
    }
    for (const input::TableRole role : CONDENSATION_TABLES)
        if (!snapshot.hasTable(role))
            throw base::SolverError(base::concatenate(
                {"expected the component tables of the model for a condensed analysis, found no '",
                 input::tableName(role),
                 "'; read the model with its component tables (modelFromReader(oor, deck, { condensation: true }))"}));
    if (settings.modeCount < 0)
        throw base::SolverError(base::concatenate(
            {"expected a count of fixed-interface modes of at least 0, found ", std::to_string(settings.modeCount)}));
    if (analysis == solutions::AnalysisKind::Modes || analysis == solutions::AnalysisKind::Buckling)
    {
        request.reduction = reduction::ReductionKind::CraigBampton;
        request.modeCount = settings.modeCount;
    }
    request.fullUnits = settings.fullUnits;
    request.retainedNodes.assign(settings.retainedNodes.begin(), settings.retainedNodes.end());
    request.retainedNodes.insert(request.retainedNodes.end(), retained.begin(), retained.end());
    std::sort(request.retainedNodes.begin(), request.retainedNodes.end());
    request.retainedNodes.erase(std::unique(request.retainedNodes.begin(), request.retainedNodes.end()),
                                request.retainedNodes.end());
    base::DiagnosticSink sink;
    std::shared_ptr<const reduction::CondensationPlan> plan;
    {
        const solutions::CondensationPlanner planner(snapshot, budget, pool, sink, analysis);
        index = planner.analysisModel().sharedIndex();
        const reduction::UnitSet units = reduction::UnitSet::fromComponents(snapshot.units());
        try
        {
            plan = std::make_shared<const reduction::CondensationPlan>(planner.plan(units, request));
        }
        catch (const std::invalid_argument& error)
        {
            // A request naming a unit or a point the model does not have: a problem of the settings.
            throw base::SolverError(error.what());
        }
    }
    diagnostics = sink.diagnostics();
    return plan;
}

// The diagnostics of the plan's own analysis model that the run does not report again, added to the run's.
void addPlanDiagnostics(std::vector<base::Diagnostic>& diagnostics, std::vector<base::Diagnostic>& planDiagnostics)
{
    for (base::Diagnostic& diagnostic : planDiagnostics)
    {
        const bool isReported =
            std::any_of(diagnostics.begin(), diagnostics.end(), [&](const base::Diagnostic& other)
                        { return other.code == diagnostic.code && other.message == diagnostic.message; });
        if (!isReported)
            diagnostics.push_back(std::move(diagnostic));
    }
}

// What the plan decided, for the caller.
CondensationSummary summaryOf(const reduction::CondensationPlan& plan)
{
    CondensationSummary summary;
    for (const reduction::UnitPlan& unit : plan.units())
        (unit.decision == reduction::UnitDecision::Condensed ? summary.condensedUnits : summary.fullUnits)
            .push_back(unit.unitId);
    std::sort(summary.condensedUnits.begin(), summary.condensedUnits.end());
    std::sort(summary.fullUnits.begin(), summary.fullUnits.end());
    summary.interfaceDofs = plan.isFullPath() ? 0 : plan.interfaceEstimate().dofCount;
    summary.note = plan.note();
    return summary;
}

// The external unit of the settings, its matrices in the budget of the run.
std::shared_ptr<const reduction::ExternalUnit> externalUnitOf(const ExternalUnitSettings& settings,
                                                              base::MemoryBudget& budget)
{
    const std::size_t n = settings.labels.size();
    if (settings.stiffness.size() != n * n || (!settings.mass.empty() && settings.mass.size() != n * n))
        throw base::SolverError(base::concatenate({"expected the matrices of the external unit of zone ",
                                                   std::to_string(settings.zone.id), " of ", std::to_string(n), " x ",
                                                   std::to_string(n), " values (one row and column per label), found ",
                                                   std::to_string(settings.stiffness.size()), " stiffness and ",
                                                   std::to_string(settings.mass.size()), " mass values"}));
    reduction::ExternalUnitParts parts;
    parts.zoneId = settings.zone.id;
    parts.zoneName = settings.zone.name;
    parts.labels = base::BudgetedBuffer<reduction::UnitDofLabel>(budget, n, "external unit labels");
    std::copy(settings.labels.begin(), settings.labels.end(), parts.labels.data());
    parts.stiffness = linalg::DenseMatrix(budget, n, n, "external unit stiffness");
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            parts.stiffness(i, j) = settings.stiffness[j * n + i];
    if (!settings.mass.empty())
    {
        parts.mass = linalg::DenseMatrix(budget, n, n, "external unit mass");
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i)
                (*parts.mass)(i, j) = settings.mass[j * n + i];
    }
    parts.identity = "matrices given to the API";
    try
    {
        return std::make_shared<const reduction::ExternalUnit>(std::move(parts));
    }
    catch (const std::invalid_argument& error)
    {
        throw base::SolverError(error.what());
    }
}

AnalysisOutput runStatics(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                          base::MemoryBudget& budget, base::ThreadPool* pool)
{
    if (settings.isStreamed && settings.condensation.units != CondensedUnits::None)
        throw base::SolverError(
            "expected the full path or the condensed path, found both a condensation and the "
            "streamed path; leave one of settings.condensation and settings.streamed out");
    solutions::StaticOptions options;
    options.subcases = settings.subcases;
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    options.isStreamed = settings.isStreamed;
    std::vector<base::Diagnostic> planDiagnostics;
    options.condensation = condensationPlanOf(snapshot, solutions::AnalysisKind::Statics, settings.condensation, {},
                                              budget, pool, planDiagnostics, options.snapshotIndex);
    options.memoryMode = settings.condensation.memoryMode;
    options.cache = settings.condensation.cache.get();
    options.cacheModelTag = settings.condensation.cacheModelTag;
    for (const ExternalUnitSettings& unit : settings.externalUnits)
        options.externalUnits.push_back({unit.zone, externalUnitOf(unit, budget)});
    options.envelopes = settings.envelopes;
    options.keepsCaseTables = settings.keepsCaseTables;
    AnalysisOutput output;
    solutions::StaticResults results = runStaticSolution(snapshot, options, budget, output.selectedQuarticShells);
    output.diagnostics = std::move(results.diagnostics);
    // The plan's own analysis model reports what the run reports again: only its other diagnostics are added.
    addPlanDiagnostics(output.diagnostics, planDiagnostics);
    output.tables = std::move(results.arrays);
    output.envelopes = std::move(results.envelopes);
    if (options.condensation)
    {
        output.condensation = summaryOf(*options.condensation);
        if (results.condensed)
            output.condensation->isUnitByUnit =
                std::any_of(results.condensed->constraintSets.begin(), results.condensed->constraintSets.end(),
                            [](const solutions::CondensedSetRecord& set) { return set.isUnitByUnit; });
    }
    return output;
}

// The streamed path asked of an analysis that has none.
void refuseStreamed(const AnalysisSettings& settings, std::string_view analysis)
{
    if (!settings.externalUnits.empty())
        throw base::SolverError(base::concatenate({"expected no external unit for ", analysis,
                                                   ": external units are offered for statics only; leave "
                                                   "settings.externalUnits out"}));
    if (settings.isStreamed)
        throw base::SolverError(base::concatenate({"expected the full path or the condensed path for ", analysis,
                                                   ": the streamed path is offered for statics only; leave "
                                                   "settings.streamed out"}));
    if (!settings.envelopes.empty() || !settings.keepsCaseTables)
        throw base::SolverError(base::concatenate({"expected no envelope for ", analysis,
                                                   ": envelopes are offered for statics only; leave "
                                                   "settings.envelopes and settings.keepsCaseTables out"}));
}

AnalysisOutput runModes(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                        base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::ModesOptions options;
    options.subcases = settings.subcases;
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    std::vector<base::Diagnostic> planDiagnostics;
    options.condensation = condensationPlanOf(snapshot, solutions::AnalysisKind::Modes, settings.condensation, {},
                                              budget, pool, planDiagnostics, options.snapshotIndex);
    options.cache = settings.condensation.cache.get();
    options.cacheModelTag = settings.condensation.cacheModelTag;
    solutions::ModesResults results = solutions::ModesSolution(snapshot, options).run(budget);
    AnalysisOutput output;
    for (const solutions::ModesSubcaseResult& subcase : results.subcases)
        output.eigenvalues.push_back(subcase.eigenvalues);
    output.diagnostics = std::move(results.diagnostics);
    addPlanDiagnostics(output.diagnostics, planDiagnostics);
    output.tables = std::move(results.arrays);
    if (options.condensation)
        output.condensation = summaryOf(*options.condensation);
    return output;
}

AnalysisOutput runBuckling(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                           base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::BucklingOptions options;
    options.subcases = settings.subcases;
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    std::vector<base::Diagnostic> planDiagnostics;
    options.condensation = condensationPlanOf(snapshot, solutions::AnalysisKind::Buckling, settings.condensation, {},
                                              budget, pool, planDiagnostics, options.snapshotIndex);
    options.cache = settings.condensation.cache.get();
    options.cacheModelTag = settings.condensation.cacheModelTag;
    solutions::BucklingResults results = solutions::BucklingSolution(snapshot, options).run(budget);
    AnalysisOutput output;
    for (const solutions::BucklingSolve& solve : results.solves)
        output.eigenvalues.push_back(solve.eigenvalues);
    // The preload's warnings first, then those of the buckling part: the order in which the run made them.
    output.diagnostics = std::move(results.preload.diagnostics);
    output.diagnostics.insert(output.diagnostics.end(), results.diagnostics.begin(), results.diagnostics.end());
    addPlanDiagnostics(output.diagnostics, planDiagnostics);
    // The static tables of the preload and the buckling shapes are in the same arrays (BucklingSolution adds the shapes
    // to the preload's tables).
    output.tables = std::move(results.preload.arrays);
    if (options.condensation)
        output.condensation = summaryOf(*options.condensation);
    return output;
}

AnalysisOutput runTrim(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                       base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::TrimOptions options;
    options.subcases = settings.subcases;
    options.links = aero::trimLinks(snapshot);
    options.links.insert(options.links.end(), settings.links.begin(), settings.links.end());
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    std::vector<base::Diagnostic> planDiagnostics;
    if (settings.condensation.units != CondensedUnits::None)
    {
        // The structural points of the splines stay on the boundaries of their units: the aerodynamic loads act on
        // them and their motions give the slopes of the boxes.
        const std::vector<base::EntityId> retained = solutions::trimRetainedPoints(snapshot, budget);
        options.condensation =
            condensationPlanOf(snapshot, solutions::AnalysisKind::StaticAeroelastic, settings.condensation, retained,
                               budget, pool, planDiagnostics, options.snapshotIndex);
    }
    options.cache = settings.condensation.cache.get();
    options.cacheModelTag = settings.condensation.cacheModelTag;
    AnalysisOutput output;
    output.aeroelastic =
        std::make_shared<solutions::TrimResults>(solutions::TrimSolution(snapshot, options).run(budget));
    output.diagnostics = output.aeroelastic->diagnostics;
    addPlanDiagnostics(output.diagnostics, planDiagnostics);
    output.tables = std::move(output.aeroelastic->arrays);
    if (options.condensation)
        output.condensation = summaryOf(*options.condensation);
    return output;
}

} // namespace

std::string programName()
{
    return std::string("O_o_Solver ") + PROGRAM_VERSION;
}

SuperelementAnalysisOutput runSuperelements(const input::ModelSnapshot& residual,
                                            std::vector<SuperelementPartSettings> parts, Analysis analysis,
                                            const AnalysisSettings& settings)
{
    if (analysis != Analysis::Statics && analysis != Analysis::Modes)
        throw base::SolverError(
            "expected statics or normal modes for an analysis with superelements; run another "
            "analysis on the model without its superelements");
    if (settings.condensation.units != CondensedUnits::None || settings.isStreamed || !settings.externalUnits.empty())
        throw base::SolverError(
            "expected no condensation, streamed path or external unit with superelements; leave "
            "them out of the settings");
    const auto pool = acquireThreadPool(settings.threads);
    auto budget = std::make_shared<base::MemoryBudget>(budgetBytes(settings));
    solutions::SuperelementOptions options;
    options.analysis = kindOf(analysis);
    options.pool = pool.get();
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    for (SuperelementPartSettings& part : parts)
    {
        if (!part.snapshot)
            throw base::SolverError(base::concatenate(
                {"expected the model of superelement ", std::to_string(part.superelementId), ", found none"}));
        solutions::SuperelementPart given;
        given.superelementId = part.superelementId;
        given.snapshot = std::move(part.snapshot);
        for (const auto& [nodeId, residualNodeId] : part.boundary)
            given.boundary.push_back({nodeId, residualNodeId});
        std::sort(given.boundary.begin(), given.boundary.end(),
                  [](const solutions::SuperelementBoundaryPoint& a, const solutions::SuperelementBoundaryPoint& b)
                  { return a.nodeId < b.nodeId; });
        options.parts.push_back(std::move(given));
    }
    // Every subcase of the Case Control applies to every superelement.
    base::DiagnosticSink caseSink;
    for (const solutions::StaticSubcase& subcase : solutions::staticSubcasesOf(residual, settings.subcases, caseSink))
        options.subcases.push_back({subcase.label, {}});
    solutions::SuperelementResults results = solutions::SuperelementSolution(residual, options).run(*budget);
    SuperelementAnalysisOutput output;
    output.budget = std::move(budget);
    for (solutions::SuperelementOutput& entry : results.outputs)
        output.superelements.push_back({entry.superelementId, std::move(entry.arrays), std::move(entry.eigenvalues)});
    output.diagnostics = std::move(results.diagnostics);
    return output;
}

std::shared_ptr<reduction::CondensationCache> makeCondensationCache(std::uint64_t memoryBytes)
{
    reduction::CacheOptions options;
    options.solverVersion = programName();
    if (memoryBytes > 0)
        options.memoryMaxBytes = memoryBytes;
    return std::make_shared<reduction::CondensationCache>(std::move(options));
}

std::vector<ElementFormulations> elementFormulations()
{
    const elements::ElementRegistry registry = elements::ElementRegistry::standard();
    std::vector<ElementFormulations> list;
    for (const elements::ElementFamily* family : registry.families())
    {
        ElementFormulations entry;
        entry.card = std::string(family->card());
        for (const std::string_view name : registry.formulationsOf(family->card()))
            entry.formulations.emplace_back(name);
        list.push_back(std::move(entry));
    }
    return list;
}

namespace
{

// The problems that an element formulation with a model of its own finds before any assembly (the mixed hybrid
// formulation: its geometry declarations, the elements its reading of the mesh refuses), which stop an analysis as a
// refused card does: appended to `refusals`. The model of the formulation is built as an analysis builds it (the
// reading of the surface and the check of every element); the other formulations have none.
void addFormulationRefusals(const input::ModelSnapshot& snapshot, const elements::ElementRegistry& registry,
                            Analysis analysis, std::vector<base::Diagnostic>& refusals)
{
    if (!registry.isMixedHybrid())
        return;
    base::DiagnosticSink sink;
    try
    {
        base::MemoryBudget budget(budgetBytes(AnalysisSettings{}));
        const elements::SnapshotIndex index(snapshot, budget);
        const elements::FormulationModel* formulation = elements::formulationModelOf(index, registry);
        if (formulation == nullptr)
            return;
        formulation->reportProblems(sink);
        const elements::EntityLayout::Kind kind =
            analysis == Analysis::Modes ? elements::EntityLayout::Kind::Modal : elements::EntityLayout::Kind::Static;
        const assembly::ElementSet elements(index, registry, &formulation->layout(kind));
        elements.check(sink, nullptr);
    }
    catch (const std::exception& error)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "formulation-model";
        diagnostic.message = error.what();
        refusals.push_back(std::move(diagnostic));
    }
    for (const base::Diagnostic& diagnostic : sink.diagnostics())
        if (diagnostic.severity == base::Severity::Error)
            refusals.push_back(diagnostic);
}

} // namespace

SurfaceEdgeTable surfaceEdges(const input::ModelSnapshot& snapshot)
{
    SurfaceEdgeTable table;
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    if (!registry.isMixedHybrid())
        return table;
    base::MemoryBudget budget(budgetBytes(AnalysisSettings{}));
    const elements::SnapshotIndex index(snapshot, budget);
    const elements::FormulationModel* formulation = elements::formulationModelOf(index, registry);
    if (formulation == nullptr)
        return table;
    for (const elements::SurfaceEdgeReading& edge : formulation->surfaceEdges())
    {
        table.grid1.push_back(edge.grid1);
        table.grid2.push_back(edge.grid2);
        table.surface.emplace_back(edge.surface);
        table.state.emplace_back(edge.state);
        table.origin.emplace_back(edge.origin);
        table.angle.push_back(edge.angle);
        table.kink.push_back(edge.kink);
        table.rounding.push_back(edge.rounding);
    }
    return table;
}

static SupportCheck checkAnalysisSupport(const input::ModelSnapshot& snapshot, Analysis analysis, solutions::AnalysisKind kind)
{
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    base::SupportReport report = solutions::checkSupport(snapshot, registry, kind);
    // Diagnostics keep the neutral names used by the model and the tables.
    // With every card supported, the formulation's own problems: an analysis would stop on them too.
    if (report.refusals.empty())
        addFormulationRefusals(snapshot, registry, analysis, report.refusals);
    return {report.supported, report.refusals, report.ignored};
}

SupportCheck checkSupport(const input::ModelSnapshot& snapshot, Analysis analysis)
{
    return checkAnalysisSupport(snapshot, analysis, kindOf(analysis));
}

SupportCheck checkTrimPreloadedBuckling(const input::ModelSnapshot& snapshot)
{
    return checkAnalysisSupport(snapshot, Analysis::Buckling, solutions::AnalysisKind::TrimPreloadedBuckling);
}

AnalysisOutput runAnalysis(const input::ModelSnapshot& snapshot, Analysis analysis, const AnalysisSettings& settings)
{
    const auto pool = acquireThreadPool(settings.threads);
    auto budget = std::make_shared<base::MemoryBudget>(budgetBytes(settings));
    AnalysisOutput output;
    switch (analysis)
    {
    case Analysis::Statics:
        output = runStatics(snapshot, settings, *budget, pool.get());
        break;
    case Analysis::Modes:
        refuseStreamed(settings, "normal modes");
        output = runModes(snapshot, settings, *budget, pool.get());
        break;
    case Analysis::Buckling:
        refuseStreamed(settings, "buckling");
        output = runBuckling(snapshot, settings, *budget, pool.get());
        break;
    case Analysis::Aeroelastic:
        refuseStreamed(settings, "the aeroelastic trim");
        output = runTrim(snapshot, settings, *budget, pool.get());
        break;
    }
    output.budget = std::move(budget);
    return output;
}

} // namespace oos::api
