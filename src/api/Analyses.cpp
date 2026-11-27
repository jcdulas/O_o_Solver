#include "oos/api/Analyses.hpp"

#include "AdaptiveShellStatics.hpp"
#include "oos/api/Threading.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/base/platform/MemoryInfo.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/FormulationModel.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/Laminate.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/tables/PiercingFootprint.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
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
#include <cmath>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>

namespace oos::api
{

namespace
{

constexpr const char* PROGRAM_VERSION = "0.2.7";

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

// The eigenvalue solver of the settings for normal modes or buckling (`analysis` names it in the messages): refused
// with a condensation, whose reduced problem has its own solver.
solutions::EigenSolverOptions eigenSolverOf(const AnalysisSettings& settings, std::string_view analysis)
{
    solutions::EigenSolverOptions options;
    const EigenSolverSettings& chosen = settings.eigenSolver;
    switch (chosen.solver)
    {
    case EigenSolver::Lanczos:
        options.solver = solutions::EigenSolver::Lanczos;
        break;
    case EigenSolver::Substructuring:
        options.solver = solutions::EigenSolver::Substructuring;
        break;
    case EigenSolver::OutOfCore:
        options.solver = solutions::EigenSolver::OutOfCore;
        break;
    }
    if (chosen.solver == EigenSolver::Lanczos)
        return options;
    if (settings.condensation.units != CondensedUnits::None)
        throw base::SolverError(base::concatenate(
            {"expected the full path for the eigenvalue solver chosen for ", analysis,
             ", found a condensation too: the condensed path solves its reduced problem itself; leave one of "
             "settings.condensation and settings.eigenSolver out"}));
    if (!(chosen.cutoffFactor >= 0.0))
        throw base::SolverError(base::concatenate(
            {"expected a cut-off factor of the substructuring of 0 (the default) or more, found ",
             std::to_string(chosen.cutoffFactor)}));
    options.substructuringCutoffFactor = chosen.cutoffFactor;
    options.scratchFolder = chosen.scratchFolder;
    if (options.scratchFolder.empty() && chosen.solver == EigenSolver::OutOfCore)
        if (const char* folder = std::getenv("OOS_SCRATCH"); folder != nullptr)
            options.scratchFolder = folder;
    return options;
}

// The name of the eigenvalue solver that ran, for the report.
std::string eigenSolverName(solutions::AnalysisPath path)
{
    switch (path)
    {
    case solutions::AnalysisPath::Substructuring:
        return "substructuring";
    case solutions::AnalysisPath::OutOfCore:
        return "outOfCore";
    case solutions::AnalysisPath::Full:
        return "lanczos";
    default:
        return "condensed";
    }
}

// An eigenvalue solver asked of an analysis that solves no eigenvalue problem.
void refuseEigenSolver(const AnalysisSettings& settings, std::string_view analysis)
{
    if (settings.eigenSolver.solver != EigenSolver::Lanczos)
        throw base::SolverError(base::concatenate({"expected no eigenvalue solver for ", analysis,
                                                   ": it is chosen for normal modes and buckling only; leave "
                                                   "settings.eigenSolver out"}));
}

// The report of a static run: its sizes and stages (StaticTimings), the kernel family it was given.
RunReport staticReport(const solutions::StaticResults& results, linalg::KernelKind kernel)
{
    RunReport report;
    report.kernel = std::string(linalg::kernelName(kernel));
    report.dofs = results.gSetSize;
    report.solvedDofs = results.largestFsetSize;
    report.factorisations = static_cast<std::size_t>(results.constraintSetCount);
    const solutions::StaticTimings& t = results.timings;
    report.stages = {{"setup", t.setup},         {"assembly", t.assembly},
                     {"loadBasis", t.loadBasis}, {"factorisation", t.factorisation},
                     {"solve", t.solve},         {"recovery", t.recovery}};
    return report;
}

// The factorisations of the eigenvalue solves (one per Lanczos shift).
template <typename Solve>
std::size_t factorisationsOf(const std::vector<Solve>& solves)
{
    std::size_t count = 0;
    for (const Solve& solve : solves)
        count += solve.factorisationCount;
    return count;
}

RunReport modesReport(const solutions::ModesResults& results, linalg::KernelKind kernel)
{
    RunReport report;
    report.kernel = std::string(linalg::kernelName(kernel));
    report.dofs = results.gSetSize;
    report.solvedDofs = results.largestAnalysisSetSize;
    report.factorisations = factorisationsOf(results.solves);
    report.eigenSolver = eigenSolverName(results.path);
    const solutions::ModesTimings& t = results.timings;
    report.stages = {{"setup", t.setup}, {"assembly", t.assembly}, {"eigenvalues", t.eigenvalues},
                     {"recovery", t.recovery}};
    return report;
}

// A buckling run: the static preload's factorisations and those of the eigenvalue solves.
RunReport bucklingReport(const solutions::BucklingResults& results, linalg::KernelKind kernel)
{
    RunReport report;
    report.kernel = std::string(linalg::kernelName(kernel));
    report.dofs = results.preload.gSetSize;
    report.solvedDofs = std::max(results.preload.largestFsetSize, results.largestAnalysisSetSize);
    report.factorisations =
        static_cast<std::size_t>(results.preload.constraintSetCount) + factorisationsOf(results.solves);
    report.eigenSolver = eigenSolverName(results.path);
    const solutions::BucklingTimings& t = results.timings;
    report.stages = {{"preload", t.preload}, {"assembly", t.assembly}, {"eigenvalues", t.eigenvalues},
                     {"recovery", t.recovery}};
    return report;
}

// A trim run: the trim does not count its factorisations.
RunReport trimReport(const solutions::TrimResults& results, linalg::KernelKind kernel)
{
    RunReport report;
    report.kernel = std::string(linalg::kernelName(kernel));
    report.dofs = results.gSetSize;
    report.solvedDofs = results.largestLsetSize;
    const solutions::TrimTimings& t = results.timings;
    report.stages = {{"setup", t.setup},
                     {"assembly", t.assembly},
                     {"factorisation", t.factorisation},
                     {"flexibility", t.flexibility},
                     {"aerodynamics", t.aerodynamics},
                     {"trim", t.trim}};
    return report;
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
    output.report = staticReport(results, options.kernel);
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
    // First: a solver refused with a condensation is refused before the plan of the condensation is made.
    options.eigenSolver = eigenSolverOf(settings, "normal modes");
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
    output.report = modesReport(results, options.kernel);
    if (options.condensation)
        output.condensation = summaryOf(*options.condensation);
    return output;
}

AnalysisOutput runBuckling(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                           base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::BucklingOptions options;
    // First: a solver refused with a condensation is refused before the plan of the condensation is made.
    options.eigenSolver = eigenSolverOf(settings, "buckling");
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
    output.report = bucklingReport(results, options.kernel);
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
    output.report = trimReport(*output.aeroelastic, options.kernel);
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
    refuseEigenSolver(settings, "an analysis with superelements");
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

namespace
{

// The laminates of the report: the transverse shear of each laminate property the shells use. A model without a
// laminate property builds nothing. The lookups of the model are built again for it, in a budget of their own (the
// run's is held by its results); a model that cannot afford them gets no table rather than a failed run, since the
// analysis is done.
std::vector<LaminateShear> laminateShearOf(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings)
{
    std::vector<LaminateShear> rows;
    if (snapshot.findPropertyTable("CompProp") == nullptr && snapshot.findPropertyTable("CompPropGlobal") == nullptr)
        return rows;
    try
    {
        base::MemoryBudget budget(budgetBytes(settings));
        const elements::SnapshotIndex index(snapshot, budget);
        for (elements::LaminateShearSummary& summary : elements::summariseLaminateShear(index))
            rows.push_back({std::move(summary.card), summary.propId, summary.stiffness, summary.factor});
    }
    catch (const std::exception&)
    {
        rows.clear();
    }
    return rows;
}

// A shear factor of a laminate further than this from 5/6, relatively, is told to the user: below it the two
// conventions of the transverse shear give the same results to the digits a comparison reads.
constexpr double LAMINATE_FACTOR_NOTICE = 0.01;

// What a comparison with another program or a formula should know of the laminates before it reads a difference as
// an error: one information when the transverse shear stiffness of some laminate is not 5/6 of the sheared sum of
// its plies, which many programs and closed forms take for every stack. It counts these properties, names the
// furthest and the model parameter that takes 5/6. Nothing under that parameter (the factors are then 5/6).
void addLaminateShearNote(const std::vector<LaminateShear>& laminates, std::vector<base::Diagnostic>& diagnostics)
{
    std::size_t count = 0;
    const LaminateShear* furthest = nullptr;
    std::size_t furthestAxis = 0;
    double furthestGap = 0.0;
    for (const LaminateShear& laminate : laminates)
    {
        bool isTold = false;
        for (std::size_t axis = 0; axis < 2; ++axis)
        {
            if (!(laminate.factor[axis] > 0.0))
                continue; // rigid along the axis: no factor
            const double gap = std::abs(laminate.factor[axis] / elements::LAMINATE_SHEAR_FACTOR - 1.0);
            isTold = isTold || gap > LAMINATE_FACTOR_NOTICE;
            if (gap > furthestGap)
            {
                furthestGap = gap;
                furthest = &laminate;
                furthestAxis = axis;
            }
        }
        count += isTold ? 1 : 0;
    }
    if (count == 0)
        return;
    base::Diagnostic note;
    note.severity = base::Severity::Information;
    note.code = "laminate-shear-factor";
    note.card = furthest->card;
    note.id = furthest->propId;
    note.field = std::string(elements::LAMINATE_SHEAR_PARAMETER);
    note.message = base::concatenate(
        {std::to_string(count), " of the ", std::to_string(laminates.size()),
         " laminate properties of the shells have a transverse shear stiffness that is not 5/6 of the sum of their "
         "plies' shear moduli times their thicknesses (the furthest: ", furthest->card, " ",
         std::to_string(furthest->propId), ", factor ", std::to_string(furthest->factor[furthestAxis]), " along ",
         furthestAxis == 0 ? "x" : "y", "): the stiffness follows the shear stresses of the stack (Parameter ",
         elements::LAMINATE_SHEAR_PARAMETER, " = ", elements::LAMINATE_SHEAR_EQUILIBRIUM,
         ", the default); a program or a formula that takes the factor 5/6 for every stack has another transverse "
         "shear stiffness, in the ratio of the factors; to compare with one, give Parameter ",
         elements::LAMINATE_SHEAR_PARAMETER, " = ", elements::LAMINATE_SHEAR_FIVE_SIXTHS,
         " (the laminate rows of the report of the run give the stiffness and the factor of each property)"});
    diagnostics.push_back(std::move(note));
}

// One information when the shells of the model are computed in the classic formulation: the model can be put in the
// mixed hybrid one, which reaches the converged values on coarse meshes. The text is the same for every model, so
// that an interface that shows each message once shows it once.
void addFormulationNote(const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& diagnostics)
{
    const bool hasShells =
        std::any_of(input::PIERCED_SHELL_CARDS.begin(), input::PIERCED_SHELL_CARDS.end(), [&](std::string_view card) {
            const input::ElementBatch* batch = snapshot.findElementBatch(card);
            return batch != nullptr && batch->rowCount() > 0;
        });
    if (!hasShells || elements::ElementRegistry::forModel(snapshot).isMixedHybrid())
        return;
    base::Diagnostic note;
    note.severity = base::Severity::Information;
    note.code = "mixed-formulation-available";
    note.field = std::string(elements::MODEL_FORMULATION_PARAMETER);
    note.message = base::concatenate(
        {"the shells of the model are computed in ", elements::CLASSIC_FORMULATION,
         ": displacement elements, which approach the converged values as the mesh is refined; ",
         elements::MIXED_HYBRID_FORMULATION,
         " reaches these values on coarse meshes, at a higher cost per element; to compare the two, or to compare "
         "with another program at a given accuracy, choose it with Parameter ",
         elements::MODEL_FORMULATION_PARAMETER, " = ", elements::MIXED_HYBRID_FORMULATION,
         " (or the element formulation option of the interface) and the geometry declarations of the model"});
    diagnostics.push_back(std::move(note));
}

} // namespace

AnalysisOutput runAnalysis(const input::ModelSnapshot& snapshot, Analysis analysis, const AnalysisSettings& settings)
{
    base::Stopwatch watch;
    const auto pool = acquireThreadPool(settings.threads);
    auto budget = std::make_shared<base::MemoryBudget>(budgetBytes(settings));
    AnalysisOutput output;
    switch (analysis)
    {
    case Analysis::Statics:
        refuseEigenSolver(settings, "statics");
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
        refuseEigenSolver(settings, "the aeroelastic trim");
        output = runTrim(snapshot, settings, *budget, pool.get());
        break;
    }
    output.report.threads = pool ? pool->threadCount() : 1;
    output.report.peakBudgetBytes = budget->peak();
    output.report.laminateShear = laminateShearOf(snapshot, settings);
    addLaminateShearNote(output.report.laminateShear, output.diagnostics);
    addFormulationNote(snapshot, output.diagnostics);
    output.report.seconds = watch.lap();
    output.budget = std::move(budget);
    return output;
}

} // namespace oos::api
