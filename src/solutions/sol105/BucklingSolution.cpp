#include "oos/solutions/sol105/BucklingSolution.hpp"

#include "oos/assembly/assemble/DifferentialStiffnessAssembler.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/BucklingPencil.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/volume/OutputVolumeEstimate.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/EigenvectorScaling.hpp"
#include "oos/solutions/common/ElementEigenSystem.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/common/FullPathEigen.hpp"
#include "oos/solutions/common/ModeRecovery.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/CondensedStatics.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace oos::solutions
{

namespace
{

constexpr double TWO_PI = 6.283185307179586;
// As the static solution: half of the budget left at the recovery for the result tables when no limit is given.
constexpr double OUTPUT_BUDGET_FRACTION = 0.5;
// The column of subcasesTable that gives the static subcase selection (STATSUB) of a subcase.
constexpr const char* STATIC_SUBCASE_COLUMN = "staticSubcaseSelection";

class Stopwatch
{
public:
    double lap()
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - m_last).count();
        m_last = now;
        return seconds;
    }

private:
    std::chrono::steady_clock::time_point m_last = std::chrono::steady_clock::now();
};

base::Diagnostic subcaseProblem(base::EntityId subcase, base::Severity severity, std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = "SubcaseStart";
    diagnostic.id = subcase;
    diagnostic.message = std::move(message);
    return diagnostic;
}

base::ProgressCounter* beginStage(ProgressMonitor* monitor, const char* stage, std::uint64_t total)
{
    return monitor == nullptr ? nullptr : &monitor->beginStage(stage, total);
}

// One subcase of the deck with its METHOD and STATSUB (when the Case Control tables give them).
struct DeckSubcase
{
    StaticSubcase subcase;
    std::optional<std::int32_t> method;
    std::optional<std::int32_t> staticSubcase;
};

bool hasPreload(const StaticSubcase& subcase)
{
    return subcase.load.has_value() || subcase.hasInitialStrains();
}

std::optional<std::int32_t> staticSubcaseOf(const input::ModelSnapshot& snapshot, base::EntityId tableId)
{
    if (!snapshot.hasTable(input::TableRole::Subcases))
        return std::nullopt;
    const input::SubcaseTable& table = snapshot.subcases();
    const base::ContractReader& reader = table.reader();
    if (!reader.hasColumn(STATIC_SUBCASE_COLUMN))
        return std::nullopt;
    const std::int64_t row = table.findRow(tableId);
    if (row < 0 || !reader.validity(STATIC_SUBCASE_COLUMN).isValid(row))
        return std::nullopt;
    return reader.values<std::int32_t>(STATIC_SUBCASE_COLUMN, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

// The methods and the preload of a buckling subcase.
struct BucklingCase
{
    std::size_t deckIndex = 0;
    std::size_t preloadIndex = 0;
    EigenMethod method;
    output::OutputPlan plan;
};

// The eigenvalue table of the normalised buckling shapes: K-orthonormal vectors scaled by s, φᵀ·K·φ = s² and
// φᵀ·(−K_d)·φ = s²/λ (the "generalised mass" of a buckling shape).
void fillEigenvalueRows(BucklingSolve& solve, const std::vector<double>& eigenvalues)
{
    for (std::size_t mode = 0; mode < eigenvalues.size(); ++mode)
    {
        output::EigenvalueRow row;
        row.mode = static_cast<std::int32_t>(mode + 1);
        row.extractionOrder = row.mode;
        row.eigenvalue = eigenvalues[mode];
        row.radians = std::sqrt(std::abs(row.eigenvalue));
        row.cycles = row.radians / TWO_PI;
        row.generalizedStiffness = solve.scales[mode] * solve.scales[mode];
        row.generalizedMass = row.generalizedStiffness / row.eigenvalue;
        solve.eigenvalues.rows.push_back(row);
    }
}

base::Diagnostic incompleteProblem(base::EntityId subcase, const EigenMethod& method,
                                   const linalg::EigenSolution& solution)
{
    return subcaseProblem(
        subcase, base::Severity::Warning, "eigen-incomplete",
        base::concatenate(
            {"the Lanczos solver reached its limit of shifts before the request of ", method.card, " ",
             std::to_string(method.setId), " was met: ", std::to_string(solution.eigenvalues.size()), " roots found, ",
             std::to_string(solution.verifiedCount),
             " confirmed by the Sturm counts; narrow the range or lower LanczosEigenMethod.desiredModeCount"}));
}

// What the condensed buckling solves read and write.
struct CondensedBucklingContext
{
    const input::ModelSnapshot& snapshot;
    const AnalysisModel& model;
    const std::vector<DeckSubcase>& deck;
    const std::vector<BucklingCase>& cases;
    const linalg::DenseKernels& kernels;
    const BucklingOptions& options;
    base::MemoryBudget& budget;
    base::DiagnosticSink& sink;
    BucklingResults& results;
    Stopwatch& watch;
};

// The buckling subcases on the condensed path (Craig-Bampton for SOL 103 and SOL 105): the units
// condensed with their mass for the constraints of the buckling subcases, their fixed-interface modes (the plan's
// count, or the cut-off CB_CUTOFF_FACTOR times the BUCKLING_REFERENCE_MODE_COUNT-th vibration frequency of the
// statically condensed model, buckling asking for no frequency), each unit's differential stiffness from its preload
// displacements projected on its Craig-Bampton basis (never cached), the reduced buckling problem on the interface, the
// shapes recovered on the interface and in the recovered units. The local buckling inside a condensed unit is not seen:
// the units are listed first, in a warning.
void solveCondensed(const CondensedBucklingContext& context)
{
    const assembly::DofMap& dofMap = context.model.dofMap();
    const BucklingOptions& options = context.options;
    base::MemoryBudget& budget = context.budget;
    base::DiagnosticSink& sink = context.sink;
    BucklingResults& results = context.results;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    {
        std::string units;
        for (const std::int64_t unit : results.uncheckedUnits)
            units.append(units.empty() ? "" : ", ").append(std::to_string(unit));
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "local-buckling-not-checked";
        diagnostic.card = "UNIT";
        diagnostic.message = base::concatenate(
            {"the buckling analysis is condensed (Craig-Bampton): the local buckling of the interior of the condensed "
             "units ",
             units,
             " is not checked (their differential stiffness is reduced on their boundary and fixed-interface modes); "
             "keep a unit full (plan(full=[...])) where its local buckling matters"});
        sink.report(diagnostic);
    }

    // The units with their mass, condensed for the constraints of the buckling subcases (no load pattern).
    const input::ConstraintTable* constraints =
        context.snapshot.hasTable(input::TableRole::Spc) ? &context.snapshot.constraints() : nullptr;
    std::vector<StaticSubcase> subcases; // bounded by the buckling subcases
    std::vector<assembly::SpcSet> spcSets;
    std::vector<std::size_t> solved;
    std::vector<output::OutputPlan> plans;
    spcSets.reserve(context.cases.size());
    for (std::size_t c = 0; c < context.cases.size(); ++c)
    {
        StaticSubcase subcase = context.deck[context.cases[c].deckIndex].subcase;
        subcase.load.reset();
        subcase.dropInitialStrains();
        subcases.push_back(subcase);
        spcSets.emplace_back(constraints, context.snapshot.findLoads(), dofMap, subcase.spc, std::nullopt, budget,
                             sink);
        solved.push_back(c);
        plans.push_back(context.cases[c].plan);
    }
    stopOnErrors(sink, "the constraints");
    const assembly::LoadVectorBuilder loads(context.model.elementSet(), dofMap, budget);
    CondensedStatics::Inputs inputs;
    inputs.snapshot = &context.snapshot;
    inputs.model = &context.model;
    inputs.plan = options.condensation.get();
    inputs.loads = &loads;
    inputs.subcases = subcases;
    inputs.spcSets = spcSets;
    inputs.solved = solved;
    inputs.plans = plans;
    inputs.recoveredUnits = options.recoveredUnits;
    inputs.recoversDrivenUnits = options.recoversDrivenUnits;
    inputs.alsoRecoveredUnits = options.alsoRecoveredUnits;
    inputs.cache = options.cache;
    inputs.cacheModelTag = options.cacheModelTag;
    inputs.designStamps = options.designStamps;
    inputs.kernels = &context.kernels;
    inputs.pool = options.pool;
    inputs.cancellation = options.cancellation;
    inputs.stage = CondensedStage::Condensation;
    inputs.isCraigBampton = true;
    CondensedStatics condensed(inputs, budget, sink);
    std::vector<output::OutputPlan> condensedPlans; // bounded by the buckling subcases
    for (const output::OutputPlan& plan : plans)
        condensedPlans.push_back(condensed.outputPlanOf(plan));
    results.path = AnalysisPath::Condensed;
    results.timings.assembly += context.watch.lap();

    for (std::size_t c = 0; c < context.cases.size(); ++c)
    {
        if (options.cancellation != nullptr)
            options.cancellation->throwIfCancelled();
        const BucklingCase& item = context.cases[c];
        const StaticSubcase& subcase = context.deck[item.deckIndex].subcase;
        const StaticSubcase& preloadSubcase = context.deck[item.preloadIndex].subcase;
        const auto kept =
            std::find_if(results.preload.keptDisplacements.begin(), results.preload.keptDisplacements.end(),
                         [&](const KeptDisplacements& entry) { return entry.subcase == preloadSubcase.label; });
        if (kept == results.preload.keptDisplacements.end())
            throw std::logic_error("BucklingSolution: the static solution did not keep the preload displacements");

        base::ProgressCounter* progress = beginStage(options.monitor, "assembly", 1);
        CondensedStatics::GroupInputs groupInputs;
        groupInputs.constraintSet = static_cast<std::int32_t>(c);
        groupInputs.spc = &spcSets[c];
        groupInputs.selection =
            ConstraintSelection{subcase.label, subcase.spc, subcase.mpc, std::nullopt, subcase.contact};
        groupInputs.singularity = assembly::SingularityOptions::fromParams(context.model.params(), subcase.label, sink);
        linalg::LanczosOptions lanczos;
        if (item.method.blockSize > 0)
            lanczos.blockSize = item.method.blockSize;
        lanczos.control.cancellation = options.cancellation;
        // The frequency scale of the assembly for the cut-off: its lowest vibration roots, statically condensed.
        linalg::EigenRequest reference;
        reference.modeCount = CondensedStatics::BUCKLING_REFERENCE_MODE_COUNT;
        CraigBamptonRecord record;
        linalg::LanczosOptions referenceLanczos;
        referenceLanczos.control.cancellation = options.cancellation;
        const reduction::ModeRequest modeRequest = condensed.chooseModeRequest(
            groupInputs, reference, referenceLanczos, record, budget, sink, options.cutoffFactor);
        condensed.reduceModes(modeRequest, record, true, budget, sink);
        // The preload's initial strains (its temperatures and element deformations), subtracted from the
        // element forces K_d is made of.
        const std::unique_ptr<assembly::TemperatureField> temperatures =
            initialStrainsOf(context.snapshot, context.model.index().points(), preloadSubcase, budget, sink);
        if (progress != nullptr)
            progress->advance();
        results.timings.assembly += context.watch.lap();

        CondensedStatics::EigenGroupInputs eigen;
        eigen.group = groupInputs;
        eigen.request = eigenRequestOf(item.method, AnalysisKind::Buckling);
        eigen.lanczos = lanczos;
        eigen.lanczos.control.progress = beginStage(options.monitor, "eigenvalues", eigen.request.modeCount);
        eigen.isBuckling = true;
        eigen.preload = kept->values.span();
        eigen.temperatures = temperatures ? &temperatures->temperatures() : nullptr;
        const CondensedStatics::EigenGroup solvedGroup = condensed.solveEigenGroup(eigen, budget, sink);
        if (!solvedGroup.singularityTable.empty())
            results.singularityTables.push_back(solvedGroup.singularityTable);
        results.timings.eigenvalues += context.watch.lap();

        const linalg::EigenSolution& solution = solvedGroup.solution;
        BucklingSolve solve;
        solve.subcase = subcase.label;
        solve.preloadSubcase = preloadSubcase.label;
        solve.method = item.method;
        solve.eigenvalues.subcase = subcase.label;
        solve.eigenvalues.analysis = "buckling";
        solve.analysisSetSize = solvedGroup.equationCount;
        solve.modalCoordinateCount = solvedGroup.modalCoordinateCount;
        results.largestAnalysisSetSize = std::max(results.largestAnalysisSetSize, solve.analysisSetSize);
        if (!solution.isComplete)
            sink.report(incompleteProblem(subcase.label, item.method, solution));
        CondensedStatics::ModeVectors vectors = condensed.recoverModes(solvedGroup, budget);
        solve.scales = scaleEigenvectorsOnSet(vectors.vectors, vectors.isAnalysisSet.span(), item.method, dofMap, sink);
        solve.shifts = solution.shifts;
        solve.factorisationCount = solution.factorisationCount;
        solve.verifiedCount = solution.verifiedCount;
        solve.isComplete = solution.isComplete;
        fillEigenvalueRows(solve, solution.eigenvalues);
        const std::size_t modes = solution.eigenvalues.size();
        output::OutputVolumeEstimate estimate(context.model.elementSet(), dofMap);
        for (std::size_t mode = 0; mode < modes; ++mode)
            estimate.addCase(condensedPlans[c], false, [](const std::string&) { return true; });
        estimate.requireWithin(options.outputLimitBytes.value_or(
            static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available()))));
        progress = beginStage(options.monitor, "recovery", modes);
        const ModeRecovery recovery(context.model, "buckling", options.recoveryRangeElements,
                                    options.isRecoveredCaseByCase);
        // The strain energies of every shape at once (ModeRecovery::strainEnergyTables), then each shape's tables.
        std::vector<std::vector<output::ElementResultTable>> energies = recovery.strainEnergyTables(
            subcase.label, condensedPlans[c], vectors.vectors.view(), budget, options.pool, options.cancellation);
        std::vector<std::vector<output::ElementResultTable>> elementTables = recovery.elementTables(
            subcase.label, condensedPlans[c], vectors.vectors.view(), budget, options.pool, options.cancellation);
        for (std::size_t mode = 0; mode < modes; ++mode)
        {
            if (options.cancellation != nullptr)
                options.cancellation->throwIfCancelled();
            recovery.recover(
                subcase.label, condensedPlans[c], std::span<const double>(&vectors.vectors(0, mode), gSize),
                static_cast<std::int32_t>(mode + 1), solution.eigenvalues[mode], nullptr, solution.eigenvalues[mode],
                budget, options.pool, options.cancellation, results.preload.arrays,
                energies.empty() ? nullptr : &energies[mode], elementTables.empty() ? nullptr : &elementTables[mode]);
            if (progress != nullptr)
                progress->advance();
        }
        solve.eigenvectors = std::move(vectors.vectors);
        results.timings.recovery += context.watch.lap();
        results.solves.push_back(std::move(solve));
    }
    results.condensed = condensed.record();
}

} // namespace

BucklingResults BucklingSolution::run(base::MemoryBudget& budget) const
{
    const input::ModelSnapshot& snapshot = *m_snapshot;
    const base::CancellationToken* cancellation = m_options.cancellation;
    base::ThreadPool* pool = m_options.pool;
    BucklingResults results;
    Stopwatch watch;
    base::DiagnosticSink sink;

    // The checks, the model, the differential stiffness of every element, the subcases.
    const AnalysisModel model(snapshot, AnalysisKind::Buckling, budget, pool, cancellation, sink);
    const assembly::DofMap& dofMap = model.dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const assembly::DifferentialStiffnessAssembler differential(model.elementSet(), dofMap);
    differential.check(sink);
    std::vector<DeckSubcase> deck; // bounded by the subcases of the deck
    for (const StaticSubcase& subcase : staticSubcasesOf(snapshot, {}, sink))
    {
        DeckSubcase entry{subcase, std::nullopt, staticSubcaseOf(snapshot, subcase.tableId)};
        if (snapshot.hasTable(input::TableRole::Subcases) && !subcase.isCombination)
            entry.method = snapshot.subcases().selectedSet(subcase.tableId, "method");
        deck.push_back(std::move(entry));
    }
    const bool hasMethods =
        std::any_of(deck.begin(), deck.end(), [](const DeckSubcase& entry) { return entry.method.has_value(); });
    // The buckling subcases: those with METHOD; with a method given by the caller and no METHOD in the deck, every
    // SUBCASE with a preload of its own.
    std::vector<std::size_t> buckling; // indices into deck
    for (std::size_t d = 0; d < deck.size(); ++d)
    {
        const DeckSubcase& entry = deck[d];
        const bool isBuckling = entry.method.has_value() || (!hasMethods && m_options.method &&
                                                             !entry.subcase.isCombination && hasPreload(entry.subcase));
        if (!isBuckling)
            continue;
        if (!m_options.subcases.empty() && std::find(m_options.subcases.begin(), m_options.subcases.end(),
                                                     entry.subcase.label) == m_options.subcases.end())
            continue;
        buckling.push_back(d);
    }
    for (const base::EntityId asked : m_options.subcases)
        if (std::none_of(buckling.begin(), buckling.end(),
                         [&](std::size_t d) { return deck[d].subcase.label == asked; }))
            throw std::invalid_argument(base::concatenate(
                {"subcases: expected buckling subcases of the Case Control (a subcase with "
                 "EigenMethodSelection), found ",
                 std::to_string(asked),
                 "; ask for one of the subcases that select an LanczosEigenMethod or BucklingEigenMethod"}));
    if (buckling.empty())
        sink.report(subcaseProblem(deck.empty() ? 1 : deck.front().subcase.label, base::Severity::Error,
                                   "missing-method",
                                   "expected a buckling subcase (EigenMethodSelection = the set ID of an "
                                   "LanczosEigenMethod or BucklingEigenMethod card), found none; add "
                                   "EigenMethodSelection to the Case Control"));

    // The methods and the preload of each buckling subcase.
    std::vector<BucklingCase> cases; // bounded by the subcases
    for (const std::size_t d : buckling)
    {
        const DeckSubcase& entry = deck[d];
        BucklingCase item;
        item.deckIndex = d;
        if (m_options.method)
            item.method = *m_options.method;
        else if (const std::optional<EigenMethod> method =
                     readEigenMethod(snapshot, *entry.method, AnalysisKind::Buckling, entry.subcase.label, sink))
            item.method = *method;
        else
            continue;
        std::optional<std::size_t> preload;
        if (entry.staticSubcase)
        {
            for (std::size_t p = 0; p < deck.size(); ++p)
                if (deck[p].subcase.label == *entry.staticSubcase && !deck[p].subcase.isCombination)
                    preload = p;
            if (!preload)
                sink.report(
                    subcaseProblem(entry.subcase.label, base::Severity::Error, "unknown-static-subcase",
                                   base::concatenate({"StaticSubcaseSelection = ", std::to_string(*entry.staticSubcase),
                                                      " names no static subcase of the deck; correct it"})));
        }
        else if (hasPreload(entry.subcase))
            preload = d;
        else
            for (std::size_t p = d; p-- > 0;)
                if (!deck[p].method && !deck[p].subcase.isCombination)
                {
                    preload = p;
                    break;
                }
        if (!preload)
        {
            if (!entry.staticSubcase)
                sink.report(subcaseProblem(entry.subcase.label, base::Severity::Error, "missing-preload",
                                           "a buckling subcase needs a static preload: a StaticLoadSelection "
                                           "(or TemperatureSelection(LOAD)) in the subcase, or a static "
                                           "subcase before it (StaticSubcaseSelection); add one"));
            continue;
        }
        item.preloadIndex = *preload;
        item.plan = output::OutputPlan::forSubcase(snapshot, entry.subcase.tableId, sink);
        if (item.plan.of(output::OutputKind::AppliedLoads).isRequested ||
            item.plan.of(output::OutputKind::GridPointForces).isRequested)
            sink.report(subcaseProblem(entry.subcase.label, base::Severity::Information, "output-not-produced",
                                       "AppliedLoadOutput and NodeForceOutput give their tables "
                                       "for the static preload only (a buckling "
                                       "shape has no applied load; the grid point force balance of the shapes is not "
                                       "produced by this version)"));
        cases.push_back(std::move(item));
    }
    stopOnErrors(sink, "the buckling solution");

    // The static subcases: every subcase without METHOD, and the buckling subcases that are their own preload. The
    // design layer gives the options of its responses (BucklingOptions::preloadOptions).
    StaticOptions staticOptions = m_options.preloadOptions.value_or(StaticOptions{});
    staticOptions.subcases.clear();
    for (std::size_t d = 0; d < deck.size(); ++d)
    {
        const bool isOwnPreload = std::any_of(cases.begin(), cases.end(), [d](const BucklingCase& item)
                                              { return item.preloadIndex == d && item.deckIndex == d; });
        if (!deck[d].method || isOwnPreload)
            staticOptions.subcases.push_back(deck[d].subcase.label);
    }
    for (const BucklingCase& item : cases)
        if (std::find(staticOptions.keptDisplacements.begin(), staticOptions.keptDisplacements.end(),
                      deck[item.preloadIndex].subcase.label) == staticOptions.keptDisplacements.end())
            staticOptions.keptDisplacements.push_back(deck[item.preloadIndex].subcase.label);
    staticOptions.pool = pool;
    staticOptions.kernel = m_options.kernel;
    staticOptions.cancellation = cancellation;
    staticOptions.monitor = m_options.monitor;
    staticOptions.analysis = AnalysisKind::Buckling;
    staticOptions.recoveryRangeElements = m_options.recoveryRangeElements;
    staticOptions.structures = m_options.structures;
    // The condensed path: the preload on it too, every unit recovered (each unit's differential stiffness
    // is computed from its own displacements), the units statically condensed (the static keys of the cache).
    const bool isCondensedPath = m_options.condensation && !m_options.condensation->isFullPath();
    staticOptions.condensation = nullptr; // the buckling solution's own path decides the preload's
    if (isCondensedPath)
    {
        staticOptions.condensation = m_options.condensation;
        std::vector<std::int64_t> everyUnit; // bounded by the units
        for (const reduction::UnitPlan& unit : m_options.condensation->units())
            if (unit.decision == reduction::UnitDecision::Condensed)
            {
                everyUnit.push_back(unit.unitId);
                results.uncheckedUnits.push_back(unit.unitId);
            }
        staticOptions.recoveredUnits = everyUnit;
        staticOptions.cache = m_options.cache;
        staticOptions.cacheModelTag = m_options.cacheModelTag;
        staticOptions.designStamps = m_options.designStamps;
    }
    results.timings.assembly += watch.lap();
    results.preload = StaticSolution(snapshot, staticOptions).run(budget);
    results.timings.preload += watch.lap();

    const linalg::DenseKernels kernels(m_options.kernel, linalg::DenseKernelOptions{pool});
    if (isCondensedPath)
    {
        solveCondensed(
            CondensedBucklingContext{snapshot, model, deck, cases, kernels, m_options, budget, sink, results, watch});
        results.diagnostics = sink.diagnostics();
        return results;
    }
    for (const BucklingCase& item : cases)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const StaticSubcase& subcase = deck[item.deckIndex].subcase;
        const StaticSubcase& preloadSubcase = deck[item.preloadIndex].subcase;
        const auto kept =
            std::find_if(results.preload.keptDisplacements.begin(), results.preload.keptDisplacements.end(),
                         [&](const KeptDisplacements& entry) { return entry.subcase == preloadSubcase.label; });
        if (kept == results.preload.keptDisplacements.end())
            throw std::logic_error("BucklingSolution: the static solution did not keep the preload displacements");

        // K of the buckling subcase's constraints, K_d of the preload, both on the pattern of K, reduced to the n set.
        base::ProgressCounter* progress = beginStage(m_options.monitor, "assembly", 1);
        const ConstraintSelection selection{subcase.label, subcase.spc, subcase.mpc, std::nullopt, subcase.contact};
        // The preload's initial strains (its temperatures and element deformations), subtracted from the
        // element forces K_d is made of.
        const std::unique_ptr<assembly::TemperatureField> temperatures =
            initialStrainsOf(snapshot, model.index().points(), preloadSubcase, budget, sink);
        const bool isSubstructuring = m_options.eigenSolver.solver == EigenSolver::Substructuring;
        // multilevel substructuring and the factor out of core take their pencil from the elements: no
        // matrix of the analysis set assembled, K_d made element by element from the preload's displacements; the SPC
        // and MPC forces (Kgg·φ) are not produced there.
        const bool isFromElements = (isSubstructuring || m_options.eigenSolver.solver == EigenSolver::OutOfCore) &&
                                    m_options.eigenSolver.pencil == EigenPencil::Elements;
        output::OutputPlan plan = item.plan;
        std::unique_ptr<ElementEigenSystem> elementSystem;
        std::unique_ptr<ConstrainedSystem> assembledSystem;
        if (isFromElements)
        {
            ElementEigenSystem::Inputs inputs;
            inputs.model = &model;
            inputs.selection = selection;
            inputs.thresholds = orderingThresholdsOf(m_options.eigenSolver);
            inputs.transform = linalg::SpectralTransform::Buckling;
            inputs.preloadDisplacements = kept->values.span();
            inputs.preloadTemperatures = temperatures ? &temperatures->temperatures() : nullptr;
            inputs.pool = pool;
            inputs.cancellation = cancellation;
            elementSystem = std::make_unique<ElementEigenSystem>(inputs, budget, sink);
            for (const output::OutputKind kind : {output::OutputKind::SpcForces, output::OutputKind::MpcForces})
                if (plan.of(kind).isRequested)
                {
                    sink.report(subcaseProblem(
                        subcase.label, base::Severity::Information, "output-not-produced",
                        base::concatenate({output::requestNameOf(kind),
                                           " of the buckling shapes is not produced on the pencil from the elements "
                                           "(no Kgg is assembled); choose the assembled pencil (the assembled "
                                           "matrices) for them"})));
                    plan.setRequest(kind, output::OutputRequestPlan{});
                }
        }
        else
        {
            ConstrainedSystemOptions systemOptions;
            systemOptions.structures = m_options.structures;
            assembledSystem =
                std::make_unique<ConstrainedSystem>(model, selection, budget, pool, cancellation, sink, systemOptions);
        }
        const ConstrainedSystem& system = isFromElements ? elementSystem->system() : *assembledSystem;
        if (!system.singularityTable().empty())
            results.singularityTables.push_back(system.singularityTable());
        std::optional<linalg::BlockSparseMatrix> kdnn;
        if (!isFromElements)
        {
            linalg::BlockSparseMatrix kdgg(system.pattern(), budget, "differential stiffness Kdgg");
            differential.assemble(kept->values.span(), temperatures ? &temperatures->temperatures() : nullptr, kdgg,
                                  pool, cancellation);
            kdnn.emplace(system.reduce(kdgg, budget, "differential stiffness Kdnn"));
        }
        // multilevel substructuring orders with supernodes of at most 256 columns (its separators' dense pencils): an
        // ordering of its own.
        const std::shared_ptr<const FsetOrdering> sharedOrdering =
            isFromElements ? std::shared_ptr<const FsetOrdering>(&elementSystem->ordering(), [](const FsetOrdering*) {})
            : isSubstructuring
                ? std::make_shared<const FsetOrdering>(system.graph(), dofMap, system.solveSets(), assembly::DofSet::F,
                                                       linalg::OrderingMethod::Automatic, budget, pool, cancellation,
                                                       std::span<const std::int32_t>(),
                                                       orderingThresholdsOf(m_options.eigenSolver))
                : orderingOf(m_options.structures, system.graph(), dofMap, system.solveSets(), assembly::DofSet::F,
                             linalg::OrderingMethod::Automatic, budget, pool, cancellation);
        const FsetOrdering& ordering = *sharedOrdering;
        if (progress != nullptr)
            progress->advance();
        results.timings.assembly += watch.lap();
        if (isSubstructuring)
            results.path = AnalysisPath::Substructuring;
        else if (m_options.eigenSolver.solver == EigenSolver::OutOfCore)
            results.path = AnalysisPath::OutOfCore;

        BucklingSolve solve;
        solve.subcase = subcase.label;
        solve.preloadSubcase = preloadSubcase.label;
        solve.method = item.method;
        solve.eigenvalues.subcase = subcase.label;
        solve.eigenvalues.analysis = "buckling";
        solve.analysisSetSize = ordering.equationCount();
        results.largestAnalysisSetSize = std::max(results.largestAnalysisSetSize, solve.analysisSetSize);
        if (ordering.equationCount() == 0)
        {
            sink.report(subcaseProblem(subcase.label, base::Severity::Warning, "no-free-dof",
                                       "every DOF is constrained: the analysis set is empty and there is no "
                                       "buckling mode"));
            results.solves.push_back(std::move(solve));
            continue;
        }
        const linalg::DofSelection& dofs = ordering.selection();
        std::optional<linalg::SymmetricCsc> stiffness;
        std::optional<linalg::SymmetricCsc> differentialStiffness;
        if (!isFromElements)
        {
            stiffness.emplace(linalg::SymmetricCsc::fromBlocks(system.knn(), dofs, budget, pool));
            differentialStiffness.emplace(linalg::SymmetricCsc::fromBlocks(*kdnn, dofs, budget, pool));
        }
        // multilevel substructuring takes its basis from the vibration modes of (K, M): the mass of the analysis set; a
        // model without mass keeps the nodes' own buckling modes, said in an information diagnostic.
        std::optional<linalg::SymmetricCsc> mass;
        const bool hasNoMass = isSubstructuring && isFromElements && !elementSystem->pencil()->hasMassSource();
        if (isSubstructuring && !isFromElements)
        {
            linalg::BlockSparseMatrix mgg(system.pattern(), budget, "mass Mgg");
            {
                const assembly::NonStructuralMassShares shares(model.elementSet());
                assembly::MassAssembler(model.elementSet(), dofMap, shares,
                                        assembly::MassAssemblyOptions::fromParams(model.params(), subcase.label, sink))
                    .assemble(mgg, pool, cancellation);
            }
            const linalg::BlockSparseMatrix mnn = system.reduce(mgg, budget, "mass Mnn");
            mass.emplace(linalg::SymmetricCsc::fromBlocks(mnn, dofs, budget, pool));
            const std::span<const double> values = mass->values();
            if (std::all_of(values.begin(), values.end(), [](double value) { return value == 0.0; }))
                mass.reset();
        }
        if (isSubstructuring && (hasNoMass || (!isFromElements && !mass)))
            sink.report(
                subcaseProblem(subcase.label, base::Severity::Information, "substructuring-buckling-basis",
                               "Substructuring: the analysis set has no mass, so its basis is the units' own buckling "
                               "modes (local, of "
                               "percent-level accuracy) instead of their vibration modes; give the model its mass "
                               "(IsoMat IsoMat.density, PointMassInertia) for the vibration basis, or use the factor "
                               "out of core for the exact "
                               "load factors"));
        linalg::FactorOptions factorOptions;
        factorOptions.equationNamer = [&dofMap, &dofs](std::int32_t equation)
        { return equationName(dofMap, static_cast<base::DofIndex>(dofs.dofOfEquation(equation))); };
        factorOptions.cancellation = cancellation;
        const linalg::EigenRequest request = eigenRequestOf(item.method, AnalysisKind::Buckling);
        linalg::LanczosOptions lanczos;
        if (item.method.blockSize > 0)
            lanczos.blockSize = item.method.blockSize;
        lanczos.control.cancellation = cancellation;
        lanczos.control.progress = beginStage(m_options.monitor, "eigenvalues", request.modeCount);
        FullPathEigen eigen =
            isFromElements ? solveFullPathEigen(m_options.eigenSolver, ordering.symbolic(), *elementSystem->pencil(),
                                                elementSystem->productBytes(lanczos.blockSize),
                                                linalg::SpectralTransform::Buckling, request, lanczos, factorOptions,
                                                kernels, budget)
                           : solveFullPathEigen(m_options.eigenSolver, ordering.symbolic(), *stiffness,
                                                *differentialStiffness, linalg::SpectralTransform::Buckling, request,
                                                lanczos, factorOptions, kernels, budget, mass ? &*mass : nullptr);
        if (isSubstructuring || m_options.eigenSolver.solver == EigenSolver::OutOfCore)
            results.eigenPencil = isFromElements ? "elements" : "assembled";
        linalg::EigenSolution solution = std::move(eigen.solution);
        if (eigen.substructuring)
            results.substructuring = std::move(eigen.substructuring);
        if (eigen.outOfCore)
            results.outOfCore = std::move(eigen.outOfCore);
        results.timings.eigenvalues += watch.lap();

        // The normalisation, the eigenvalue table, the output volume, the tables of the shapes.
        solve.scales = scaleEigenvectors(solution.eigenvectors, item.method, dofMap, dofs, sink);
        solve.shifts = solution.shifts;
        solve.factorisationCount = solution.factorisationCount;
        solve.verifiedCount = solution.verifiedCount;
        solve.isComplete = solution.isComplete;
        if (!solution.isComplete)
            sink.report(incompleteProblem(subcase.label, item.method, solution));
        const std::size_t modes = solution.eigenvalues.size();
        fillEigenvalueRows(solve, solution.eigenvalues);
        output::OutputVolumeEstimate estimate(model.elementSet(), dofMap);
        for (std::size_t mode = 0; mode < modes; ++mode)
            estimate.addCase(plan, system.gm().rowCount() > 0, [](const std::string&) { return true; });
        estimate.requireWithin(m_options.outputLimitBytes.value_or(
            static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available()))));

        progress = beginStage(m_options.monitor, "recovery", modes);
        const ModeRecovery recovery(model, system, "buckling", m_options.recoveryRangeElements,
                                    m_options.isRecoveredCaseByCase);
        solve.eigenvectors = linalg::DenseMatrix(budget, gSize, modes, "buckling shapes");
        for (std::size_t mode = 0; mode < modes; ++mode)
        {
            if (cancellation != nullptr)
                cancellation->throwIfCancelled();
            const base::BudgetedBuffer<double> modeG = recovery.expand(solution.eigenvectors, mode, dofs, budget);
            std::copy(modeG.data(), modeG.data() + gSize, &solve.eigenvectors(0, mode));
        }
        // The strain energies of every shape at once (ModeRecovery::strainEnergyTables), then each shape's tables.
        std::vector<std::vector<output::ElementResultTable>> energies =
            recovery.strainEnergyTables(subcase.label, plan, solve.eigenvectors.view(), budget, pool, cancellation);
        std::vector<std::vector<output::ElementResultTable>> elementTables =
            recovery.elementTables(subcase.label, plan, solve.eigenvectors.view(), budget, pool, cancellation);
        for (std::size_t mode = 0; mode < modes; ++mode)
        {
            if (cancellation != nullptr)
                cancellation->throwIfCancelled();
            const std::span<const double> modeG(&solve.eigenvectors(0, mode), gSize);
            // The SPC and MPC forces of a buckling shape are those of Kgg·φ alone (ModeRecovery class comment).
            recovery.recover(subcase.label, plan, modeG, static_cast<std::int32_t>(mode + 1),
                             solution.eigenvalues[mode], nullptr, 0.0, budget, pool, cancellation,
                             results.preload.arrays, energies.empty() ? nullptr : &energies[mode],
                             elementTables.empty() ? nullptr : &elementTables[mode]);
            if (progress != nullptr)
                progress->advance();
        }
        results.timings.recovery += watch.lap();
        results.solves.push_back(std::move(solve));
    }
    results.diagnostics = sink.diagnostics();
    return results;
}

} // namespace oos::solutions
