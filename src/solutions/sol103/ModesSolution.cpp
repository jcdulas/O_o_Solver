#include "oos/solutions/sol103/ModesSolution.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/diagnostics/SubcaseChecks.hpp"
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
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

namespace oos::solutions
{

namespace
{

// 2π: cycles per unit time from radians per unit time.
constexpr double TWO_PI = 6.283185307179586;
// The share of the memory budget left when the recovery starts that the result tables may take when the caller gives
// no limit (as the static solution).
constexpr double OUTPUT_BUDGET_FRACTION = 0.5;
// PARAM GRDPNT: −1 (the default) no grid point weight generator; 0 about the origin; a grid point about it.
constexpr std::int32_t NO_WEIGHT_REFERENCE_NODE = -1;
constexpr std::size_t RIGID_MODES = 6;

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

// One subcase of the analysis with its method and outputs.
struct ModalSubcase
{
    StaticSubcase subcase;
    EigenMethod method;
    output::OutputPlan plan;
};

bool isSameSolve(const ModalSubcase& left, const ModalSubcase& right)
{
    return left.subcase.spc == right.subcase.spc && left.subcase.mpc == right.subcase.mpc &&
           left.subcase.contact == right.subcase.contact && left.method.card == right.method.card &&
           left.method.setId == right.method.setId;
}

// The modal effective mass of the modes: with D_a the six
// rigid-body motions of the analysis-set DOFs about the reference point and M_aa the analysis-set mass,
//     p_ij = φ_iᵀ·M_aa·D_j,   Γ_ij = p_ij / m_i,   M_eff,ij = p_ij² / m_i,   m_i = φ_iᵀ·M_aa·φ_i,
// and the fractions M_eff,ij / (Dᵀ·M_aa·D)_jj. With every mode of the analysis set the effective masses sum to the
// rigid-body mass (Σ_i φ_i·φ_iᵀ·M / m_i is the identity on the range of M).
output::ModalEffectiveMassTable modalEffectiveMassOf(
    base::EntityId subcase, std::int32_t referencePoint, base::Vec3 referencePosition, const AnalysisModel& model,
    const linalg::DofSelection& selection, const linalg::SymmetricCsc& mass, const linalg::DenseMatrix& vectors,
    const std::vector<double>& eigenvalues, const std::vector<double>& scales, base::MemoryBudget& budget,
    base::ThreadPool* pool)
{
    const std::size_t equations = vectors.rows();
    const assembly::RigidBodyModes rigid(model.dofMap(), referencePosition, budget);
    linalg::DenseMatrix motions(budget, equations, RIGID_MODES, "MEFFMASS rigid motions");
    for (std::size_t e = 0; e < equations; ++e)
    {
        const auto dof = static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(e)));
        for (std::size_t j = 0; j < RIGID_MODES; ++j)
            motions(e, j) = rigid.modes()(dof, j);
    }
    linalg::DenseMatrix massMotions(budget, equations, RIGID_MODES, "MEFFMASS M D");
    mass.multiply(motions.view(), massMotions.view(), pool);

    output::ModalEffectiveMassTable table;
    table.subcase = subcase;
    table.referencePoint = referencePoint;
    for (std::size_t j = 0; j < RIGID_MODES; ++j)
        for (std::size_t k = 0; k < RIGID_MODES; ++k)
        {
            double sum = 0.0;
            for (std::size_t e = 0; e < equations; ++e)
                sum += motions(e, j) * massMotions(e, k);
            table.rigidBodyMass[j * RIGID_MODES + k] = sum;
        }
    std::array<double, RIGID_MODES> running{};
    for (std::size_t i = 0; i < vectors.columns(); ++i)
    {
        output::ModalEffectiveMassRow row;
        row.mode = static_cast<std::int32_t>(i + 1);
        row.cycles = std::sqrt(std::abs(eigenvalues[i])) / TWO_PI;
        const double generalizedMass = scales[i] * scales[i];
        for (std::size_t j = 0; j < RIGID_MODES; ++j)
        {
            double product = 0.0;
            for (std::size_t e = 0; e < equations; ++e)
                product += vectors(e, i) * massMotions(e, j);
            row.participation[j] = product / generalizedMass;
            row.effectiveMass[j] = product * product / generalizedMass;
            const double total = table.rigidBodyMass[j * RIGID_MODES + j];
            row.fraction[j] = total > 0.0 ? row.effectiveMass[j] / total : 0.0;
            running[j] += row.fraction[j];
            row.fractionSum[j] = running[j];
            table.totalEffectiveMass[j] += row.effectiveMass[j];
        }
        table.rows.push_back(row);
    }
    return table;
}

// The eigenvalue table of a subcase: λ, radians, cycles, generalised mass s² and stiffness s²·λ of each mode.
output::EigenvalueTable eigenvalueTableOf(base::EntityId subcase, const std::vector<double>& eigenvalues,
                                          const std::vector<double>& scales)
{
    output::EigenvalueTable table;
    table.subcase = subcase;
    table.analysis = "modes";
    for (std::size_t mode = 0; mode < eigenvalues.size(); ++mode)
    {
        output::EigenvalueRow row;
        row.mode = static_cast<std::int32_t>(mode + 1);
        row.extractionOrder = row.mode;
        row.eigenvalue = eigenvalues[mode];
        row.radians = std::sqrt(std::abs(row.eigenvalue));
        row.cycles = row.radians / TWO_PI;
        row.generalizedMass = scales[mode] * scales[mode];
        row.generalizedStiffness = row.generalizedMass * row.eigenvalue;
        table.rows.push_back(row);
    }
    return table;
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

// What the condensed solve of one group of subcases reads and writes.
struct CondensedGroupContext
{
    const AnalysisModel& model;
    CondensedStatics& condensed;
    const std::vector<output::OutputPlan>& plans; // the condensed output plans, per subcase
    const std::vector<assembly::SpcSet>& spcSets; // per subcase
    const std::vector<ModalSubcase>& subcases;
    const std::vector<std::size_t>& group;
    std::size_t groupIndex;
    const linalg::DenseKernels& kernels;
    const ModesOptions& options;
    base::MemoryBudget& budget;
    base::DiagnosticSink& sink;
    ModesResults& results;
    Stopwatch& watch;
};

// One group on the condensed path (Craig-Bampton for SOL 103): the units' fixed-interface modes (the
// plan's count, or the cut-off CB_CUTOFF_FACTOR times the highest frequency requested), the reduced eigenproblem on the
// interface (K* and K_qq, the Craig-Bampton masses, the interface elements), the modes recovered on the interface and
// in the recovered units, normalised on the known components of the full path's analysis set, and their tables.
void solveCondensedGroup(const CondensedGroupContext& context)
{
    const ModalSubcase& first = context.subcases[context.group.front()];
    const EigenMethod& method = first.method;
    const assembly::DofMap& dofMap = context.model.dofMap();
    const ModesOptions& options = context.options;
    base::MemoryBudget& budget = context.budget;
    base::DiagnosticSink& sink = context.sink;
    ModesResults& results = context.results;
    CondensedStatics& condensed = context.condensed;

    base::ProgressCounter* progress = beginStage(options.monitor, "assembly", 1);
    CondensedStatics::GroupInputs groupInputs;
    groupInputs.constraintSet = static_cast<std::int32_t>(context.groupIndex);
    groupInputs.spc = &context.spcSets[context.group.front()];
    groupInputs.selection = ConstraintSelection{first.subcase.label, first.subcase.spc, first.subcase.mpc, std::nullopt,
                                                first.subcase.contact};
    groupInputs.singularity =
        assembly::SingularityOptions::fromParams(context.model.params(), first.subcase.label, sink);
    const linalg::EigenRequest request = eigenRequestOf(method, AnalysisKind::Modes);
    linalg::LanczosOptions lanczos;
    if (method.blockSize > 0)
        lanczos.blockSize = method.blockSize;
    lanczos.control.cancellation = options.cancellation;
    CraigBamptonRecord record;
    const reduction::ModeRequest modeRequest =
        condensed.chooseModeRequest(groupInputs, request, lanczos, record, budget, sink, options.cutoffFactor);
    condensed.reduceModes(modeRequest, record, false, budget, sink);
    if (progress != nullptr)
        progress->advance();
    results.timings.assembly += context.watch.lap();

    CondensedStatics::EigenGroupInputs eigen;
    eigen.group = groupInputs;
    eigen.request = request;
    eigen.lanczos = lanczos;
    eigen.lanczos.control.progress = beginStage(options.monitor, "eigenvalues", request.modeCount);
    const CondensedStatics::EigenGroup solved = condensed.solveEigenGroup(eigen, budget, sink);
    if (!solved.singularityTable.empty())
        results.singularityTables.push_back(solved.singularityTable);
    results.timings.eigenvalues += context.watch.lap();

    const linalg::EigenSolution& solution = solved.solution;
    ModesSolve solve;
    for (const std::size_t s : context.group)
        solve.subcases.push_back(context.subcases[s].subcase.label);
    solve.method = method;
    solve.analysisSetSize = solved.equationCount;
    solve.modalCoordinateCount = solved.modalCoordinateCount;
    results.largestAnalysisSetSize = std::max(results.largestAnalysisSetSize, solve.analysisSetSize);
    if (!solution.isComplete)
        sink.report(incompleteProblem(first.subcase.label, method, solution));
    CondensedStatics::ModeVectors vectors = condensed.recoverModes(solved, budget);
    solve.scales = scaleEigenvectorsOnSet(vectors.vectors, vectors.isAnalysisSet.span(), method, dofMap, sink);
    solve.eigenvalues = solution.eigenvalues;
    solve.shifts = solution.shifts;
    solve.factorisationCount = solution.factorisationCount;
    solve.verifiedCount = solution.verifiedCount;
    solve.isComplete = solution.isComplete;

    // The output volume, then the tables of each mode for each subcase (their plans without the units not
    // recovered).
    const std::size_t modes = solution.eigenvalues.size();
    output::OutputVolumeEstimate estimate(context.model.elementSet(), dofMap);
    for (const std::size_t s : context.group)
        for (std::size_t mode = 0; mode < modes; ++mode)
            estimate.addCase(context.plans[s], false, [](const std::string&) { return true; });
    estimate.requireWithin(options.outputLimitBytes.value_or(
        static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available()))));
    progress = beginStage(options.monitor, "recovery", modes);
    const ModeRecovery recovery(context.model, "modes", options.recoveryRangeElements, options.isRecoveredCaseByCase);
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    // The strain energies of every mode at once (ModeRecovery::strainEnergyTables), then each mode's tables.
    std::vector<std::vector<std::vector<output::ElementResultTable>>> energies; // per subcase of the group, per mode
    std::vector<std::vector<std::vector<output::ElementResultTable>>> elementTables; // the same, by blocks of modes
    for (const std::size_t s : context.group)
    {
        energies.push_back(recovery.strainEnergyTables(context.subcases[s].subcase.label, context.plans[s],
                                                       vectors.vectors.view(), budget, options.pool,
                                                       options.cancellation));
        elementTables.push_back(recovery.elementTables(context.subcases[s].subcase.label, context.plans[s],
                                                       vectors.vectors.view(), budget, options.pool,
                                                       options.cancellation));
    }
    for (std::size_t mode = 0; mode < modes; ++mode)
    {
        if (options.cancellation != nullptr)
            options.cancellation->throwIfCancelled();
        const std::span<const double> modeG(&vectors.vectors(0, mode), gSize);
        for (std::size_t k = 0; k < context.group.size(); ++k)
        {
            const std::size_t s = context.group[k];
            recovery.recover(context.subcases[s].subcase.label, context.plans[s], modeG,
                             static_cast<std::int32_t>(mode + 1), solution.eigenvalues[mode], nullptr, 0.0, budget,
                             options.pool, options.cancellation, results.arrays,
                             energies[k].empty() ? nullptr : &energies[k][mode],
                             elementTables[k].empty() ? nullptr : &elementTables[k][mode]);
        }
        if (progress != nullptr)
            progress->advance();
    }
    solve.eigenvectors = std::move(vectors.vectors);
    for (const std::size_t s : context.group)
    {
        ModesSubcaseResult result;
        result.subcase = context.subcases[s].subcase.label;
        result.solve = results.solves.size();
        result.eigenvalues = eigenvalueTableOf(result.subcase, solve.eigenvalues, solve.scales);
        results.subcases.push_back(std::move(result));
    }
    results.timings.recovery += context.watch.lap();
    results.solves.push_back(std::move(solve));
    results.condensed = condensed.record();
}

// The pencil from the elements assembles neither Kgg nor Mgg: the SPC and MPC forces of
// the modes (Kgg·φ − λ·Mgg·φ) are not asked of the recovery, MEFFMASS (Mgg on the analysis set) and PARAM GRDPNT (Mgg)
// are not produced; one information diagnostic each, naming the assembled pencil that gives them.
void reportElementPencilOutputs(std::vector<ModalSubcase>& subcases, std::span<const std::size_t> group,
                                bool hasGridPointWeight, base::DiagnosticSink& sink)
{
    const char* remedy = "; choose the assembled pencil (the assembled matrices) for them";
    for (const std::size_t s : group)
    {
        ModalSubcase& modal = subcases[s];
        for (const output::OutputKind kind : {output::OutputKind::SpcForces, output::OutputKind::MpcForces})
            if (modal.plan.of(kind).isRequested)
            {
                sink.report(subcaseProblem(modal.subcase.label, base::Severity::Information, "output-not-produced",
                                           base::concatenate({output::requestNameOf(kind),
                                                              " of the modes is not produced on the pencil from the "
                                                              "elements (no Kgg is assembled)",
                                                              remedy})));
                modal.plan.setRequest(kind, output::OutputRequestPlan{});
            }
        if (modal.plan.modalEffectiveMass().isRequested)
            sink.report(subcaseProblem(
                modal.subcase.label, base::Severity::Information, "output-not-produced",
                base::concatenate({"ModalEffectiveMassOutput is not produced on the pencil from the elements "
                                   "(no mass matrix is assembled)",
                                   remedy})));
    }
    if (hasGridPointWeight && !group.empty())
        sink.report(subcaseProblem(
            subcases[group.front()].subcase.label, base::Severity::Information, "output-not-produced",
            base::concatenate({"PARAM WeightReferenceNode: the grid point weight generator is not produced "
                               "on the pencil from the elements (no Mgg is assembled)",
                               remedy})));
}

} // namespace

ModesResults ModesSolution::run(base::MemoryBudget& budget) const
{
    const input::ModelSnapshot& snapshot = *m_snapshot;
    const base::CancellationToken* cancellation = m_options.cancellation;
    base::ThreadPool* pool = m_options.pool;
    ModesResults results;
    Stopwatch watch;
    base::DiagnosticSink sink;

    // 1. The checks, the model, the subcases with their methods and outputs.
    base::ProgressCounter* progress = beginStage(m_options.monitor, "connectivity", 1);
    const AnalysisModel model(snapshot, AnalysisKind::Modes, budget, pool, cancellation, sink);
    if (progress != nullptr)
        progress->advance();
    const assembly::DofMap& dofMap = model.dofMap();
    results.gSetSize = dofMap.gSetSize();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    std::vector<ModalSubcase> subcases; // bounded by the subcases of the deck
    for (const StaticSubcase& subcase : staticSubcasesOf(snapshot, m_options.subcases, sink))
    {
        if (subcase.isCombination)
        {
            sink.report(
                subcaseProblem(subcase.label, base::Severity::Error, "unsupported-subcase",
                               "a SubcaseCombination combines static subcases: it has no meaning in a real eigenvalue "
                               "analysis (SOL 103); remove it"));
            continue;
        }
        // Member by member: GCC 12 warns falsely (-Wmaybe-uninitialized) on the aggregate with an empty method.
        ModalSubcase modal;
        modal.subcase = subcase;
        modal.plan = output::OutputPlan::forSubcase(snapshot, subcase.tableId, sink);
        modal.plan.addRequired(m_options.requiredOutputs);
        if (m_options.method)
            modal.method = *m_options.method;
        else
        {
            std::optional<std::int32_t> methodId;
            if (snapshot.hasTable(input::TableRole::Subcases))
                methodId = snapshot.subcases().selectedSet(subcase.tableId, "method");
            if (!methodId)
            {
                sink.report(subcaseProblem(
                    subcase.label, base::Severity::Error, "missing-method",
                    "expected EigenMethodSelection = the set ID of an LanczosEigenMethod (or "
                    "RealEigenMethod) "
                    "card in the subcase or the global section of a real eigenvalue analysis, found none; add "
                    "EigenMethodSelection to the Case Control"));
                continue;
            }
            const std::optional<EigenMethod> method =
                readEigenMethod(snapshot, *methodId, AnalysisKind::Modes, subcase.label, sink);
            if (!method)
                continue;
            modal.method = *method;
        }
        if (modal.plan.of(output::OutputKind::AppliedLoads).isRequested ||
            modal.plan.of(output::OutputKind::GridPointForces).isRequested)
            sink.report(subcaseProblem(subcase.label, base::Severity::Information, "output-not-produced",
                                       "AppliedLoadOutput and NodeForceOutput give no table in a "
                                       "real eigenvalue analysis (a mode has "
                                       "no applied load; the grid point force balance of the modes is not produced "
                                       "by this version)"));
        if (!assembly::supportDofsOf(snapshot, dofMap, subcase.tableId, sink).empty())
            sink.report(subcaseProblem(subcase.label, base::Severity::Warning, "free-body-support-in-modes",
                                       "FreeBodySupport in a real eigenvalue analysis: the r set is not separated; the "
                                       "rigid-body modes are found by the Lanczos solver from the free stiffness "
                                       "(eigenvalues at round-off instead of exact zeros)"));
        subcases.push_back(std::move(modal));
    }
    stopOnErrors(sink, "the eigenvalue solution");

    // 2. The solves: one per constraint set and method.
    std::vector<std::vector<std::size_t>> groups; // bounded by the subcases
    for (std::size_t s = 0; s < subcases.size(); ++s)
    {
        bool isPlaced = false;
        for (std::vector<std::size_t>& group : groups)
            if (isSameSolve(subcases[group.front()], subcases[s]))
            {
                group.push_back(s);
                isPlaced = true;
                break;
            }
        if (!isPlaced)
            groups.push_back({s});
    }
    const linalg::DenseKernels kernels(m_options.kernel, linalg::DenseKernelOptions{pool});
    results.kernel = std::string(linalg::kernelName(m_options.kernel));
    const input::ParamTable* params = model.params();
    const std::int32_t weightReferenceNode =
        assembly::findIntegerParam(params, "WeightReferenceNode", std::nullopt, sink)
            .value_or(NO_WEIGHT_REFERENCE_NODE);

    // The condensed path (Craig-Bampton for SOL 103): the units of the plan condensed once
    // for every subcase with their mass (CondensedStatics with isCraigBampton), the outputs of the units not recovered
    // left out of the plans (and listed).
    const bool isCondensedPath = m_options.condensation && !m_options.condensation->isFullPath();
    std::vector<StaticSubcase> condensedSubcases; // bounded by the subcases
    std::vector<assembly::SpcSet> spcSets;
    std::vector<std::size_t> solvedIndices;
    std::vector<output::OutputPlan> modalPlans;
    std::unique_ptr<assembly::LoadVectorBuilder> loadBuilder;
    std::unique_ptr<CondensedStatics> condensed;
    std::vector<output::OutputPlan> condensedPlans;
    if (isCondensedPath)
    {
        base::ProgressCounter* stage = beginStage(m_options.monitor, "condensation", 1);
        const input::ConstraintTable* constraints =
            snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
        spcSets.reserve(subcases.size());
        for (std::size_t s = 0; s < subcases.size(); ++s)
        {
            // A mode has no load: the units are condensed without load patterns.
            StaticSubcase subcase = subcases[s].subcase;
            subcase.load.reset();
            subcase.dropInitialStrains();
            condensedSubcases.push_back(subcase);
            spcSets.emplace_back(constraints, snapshot.findLoads(), dofMap, subcase.spc, std::nullopt, budget, sink);
            solvedIndices.push_back(s);
            modalPlans.push_back(subcases[s].plan);
        }
        stopOnErrors(sink, "the constraints");
        loadBuilder = std::make_unique<assembly::LoadVectorBuilder>(model.elementSet(), dofMap, budget);
        CondensedStatics::Inputs inputs;
        inputs.snapshot = &snapshot;
        inputs.model = &model;
        inputs.plan = m_options.condensation.get();
        inputs.loads = loadBuilder.get();
        inputs.subcases = condensedSubcases;
        inputs.spcSets = spcSets;
        inputs.solved = solvedIndices;
        inputs.plans = modalPlans;
        inputs.recoveredUnits = m_options.recoveredUnits;
        inputs.recoversDrivenUnits = m_options.recoversDrivenUnits;
        inputs.alsoRecoveredUnits = m_options.alsoRecoveredUnits;
        inputs.cache = m_options.cache;
        inputs.cacheModelTag = m_options.cacheModelTag;
        inputs.designStamps = m_options.designStamps;
        inputs.kernels = &kernels;
        inputs.pool = pool;
        inputs.cancellation = cancellation;
        inputs.stage = CondensedStage::Condensation;
        inputs.isCraigBampton = true;
        condensed = std::make_unique<CondensedStatics>(inputs, budget, sink);
        for (const output::OutputPlan& plan : modalPlans)
            condensedPlans.push_back(condensed->outputPlanOf(plan));
        if (stage != nullptr)
            stage->advance();
        results.path = AnalysisPath::Condensed;
        for (const ModalSubcase& modal : subcases)
        {
            if (modal.plan.of(output::OutputKind::SpcForces).isRequested ||
                modal.plan.of(output::OutputKind::MpcForces).isRequested)
                sink.report(subcaseProblem(modal.subcase.label, base::Severity::Information, "output-not-produced",
                                           "SinglePointConstrForceOutput and MultiPointConstrForceOutput "
                                           "of the modes are not produced on the condensed "
                                           "path of this version (no Kgg is assembled there); run the full path "
                                           "(no condensation) for them"));
            if (modal.plan.modalEffectiveMass().isRequested)
                sink.report(subcaseProblem(
                    modal.subcase.label, base::Severity::Information, "output-not-produced",
                    "ModalEffectiveMassOutput is not produced on the condensed path of this version; run the "
                    "full path (no condensation) for it"));
        }
        if (weightReferenceNode != NO_WEIGHT_REFERENCE_NODE)
            sink.report(subcaseProblem(
                subcases.empty() ? 1 : subcases.front().subcase.label, base::Severity::Information,
                "output-not-produced",
                "PARAM WeightReferenceNode: the grid point weight generator is not produced on the condensed "
                "path of this version; run the full path (no condensation) for it"));
    }
    results.timings.setup += watch.lap();

    for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
    {
        const std::vector<std::size_t>& group = groups[groupIndex];
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const ModalSubcase& first = subcases[group.front()];
        const EigenMethod& method = first.method;
        if (condensed)
        {
            solveCondensedGroup(CondensedGroupContext{model, *condensed, condensedPlans, spcSets, subcases, group,
                                                      groupIndex, kernels, m_options, budget, sink, results, watch});
            continue;
        }

        // 3. The constrained stiffness, the mass, the analysis-set ordering, the pencil.
        progress = beginStage(m_options.monitor, "assembly", 1);
        const ConstraintSelection selection{first.subcase.label, first.subcase.spc, first.subcase.mpc, std::nullopt,
                                            first.subcase.contact};
        const bool isSubstructuring = m_options.eigenSolver.solver == EigenSolver::Substructuring;
        // multilevel substructuring and the factor out of core take their pencil from the elements: no
        // matrix of the analysis set assembled; the outputs that need Kgg or Mgg are not produced there.
        const bool isFromElements = (isSubstructuring || m_options.eigenSolver.solver == EigenSolver::OutOfCore) &&
                                    m_options.eigenSolver.pencil == EigenPencil::Elements;
        std::unique_ptr<ElementEigenSystem> elementSystem;
        std::unique_ptr<ConstrainedSystem> assembledSystem;
        if (isFromElements)
        {
            ElementEigenSystem::Inputs inputs;
            inputs.model = &model;
            inputs.selection = selection;
            inputs.thresholds = orderingThresholdsOf(m_options.eigenSolver);
            inputs.transform = linalg::SpectralTransform::Vibration;
            inputs.pool = pool;
            inputs.cancellation = cancellation;
            elementSystem = std::make_unique<ElementEigenSystem>(inputs, budget, sink);
            reportElementPencilOutputs(subcases, group, weightReferenceNode != NO_WEIGHT_REFERENCE_NODE, sink);
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
        std::optional<linalg::BlockSparseMatrix> mgg;
        std::optional<linalg::BlockSparseMatrix> mnn;
        if (!isFromElements)
        {
            mgg.emplace(system.pattern(), budget, "mass Mgg");
            const assembly::NonStructuralMassShares shares(model.elementSet());
            assembly::MassAssembler(model.elementSet(), dofMap, shares,
                                    assembly::MassAssemblyOptions::fromParams(params, first.subcase.label, sink))
                .assemble(*mgg, pool, cancellation);
            mnn.emplace(system.reduce(*mgg, budget, "mass Mnn"));
        }
        if (weightReferenceNode != NO_WEIGHT_REFERENCE_NODE && !results.gridPointWeight && mgg)
        {
            // PARAM GRDPNT: the rigid-body mass properties, in the units of the input masses (WTMASS removed).
            base::Vec3 referencePoint{};
            if (weightReferenceNode > 0)
            {
                if (const std::optional<std::int32_t> row = model.index().points().findGrid(weightReferenceNode))
                    referencePoint = dofMap.positionInBasic(*row);
                else
                    sink.report(
                        assembly::badParamDiagnostic("WeightReferenceNode", "-1, 0 or a grid point of the model"));
            }
            const double weightToMass =
                assembly::findRealParam(params, "WeightToMass", first.subcase.label, sink).value_or(1.0);
            const assembly::RigidBodyModes rigid(dofMap, referencePoint, budget);
            base::MatN<6> rigidMass = rigid.rigidBodyMass(*mgg, budget);
            if (weightToMass != 0.0)
                rigidMass *= 1.0 / weightToMass;
            results.gridPointWeight = assembly::GridPointWeight::fromRigidBodyMass(
                rigidMass, std::max(weightReferenceNode, 0), referencePoint);
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

        ModesSolve solve;
        for (const std::size_t s : group)
            solve.subcases.push_back(subcases[s].subcase.label);
        solve.method = method;
        solve.analysisSetSize = ordering.equationCount();
        results.largestAnalysisSetSize = std::max(results.largestAnalysisSetSize, solve.analysisSetSize);
        linalg::EigenSolution solution;
        if (ordering.equationCount() == 0)
            sink.report(subcaseProblem(first.subcase.label, base::Severity::Warning, "no-free-dof",
                                       "every DOF is constrained: the analysis set is empty and there is no mode"));
        else
        {
            const linalg::DofSelection& dofs = ordering.selection();
            std::optional<linalg::SymmetricCsc> stiffness;
            std::optional<linalg::SymmetricCsc> mass;
            if (!isFromElements)
            {
                stiffness.emplace(linalg::SymmetricCsc::fromBlocks(system.knn(), dofs, budget, pool));
                mass.emplace(linalg::SymmetricCsc::fromBlocks(*mnn, dofs, budget, pool));
            }
            linalg::FactorOptions factorOptions;
            factorOptions.equationNamer = [&dofMap, &dofs](std::int32_t equation)
            { return equationName(dofMap, static_cast<base::DofIndex>(dofs.dofOfEquation(equation))); };
            factorOptions.cancellation = cancellation;
            const linalg::EigenRequest request = eigenRequestOf(method, AnalysisKind::Modes);
            linalg::LanczosOptions lanczos;
            if (method.blockSize > 0)
                lanczos.blockSize = method.blockSize;
            lanczos.control.cancellation = cancellation;
            lanczos.control.progress = beginStage(m_options.monitor, "eigenvalues", request.modeCount);
            FullPathEigen eigen =
                isFromElements
                    ? solveFullPathEigen(m_options.eigenSolver, ordering.symbolic(), *elementSystem->pencil(),
                                         elementSystem->productBytes(lanczos.blockSize),
                                         linalg::SpectralTransform::Vibration, request, lanczos, factorOptions, kernels,
                                         budget)
                    : solveFullPathEigen(m_options.eigenSolver, ordering.symbolic(), *stiffness, *mass,
                                         linalg::SpectralTransform::Vibration, request, lanczos, factorOptions, kernels,
                                         budget);
            results.eigenPencil = isFromElements ? "elements" : "assembled";
            solution = std::move(eigen.solution);
            if (eigen.substructuring)
                results.substructuring = std::move(eigen.substructuring);
            if (eigen.outOfCore)
                results.outOfCore = std::move(eigen.outOfCore);
            results.timings.eigenvalues += watch.lap();

            // 4. The normalisation, the eigenvalue table, the output volume, the tables of the modes.
            solve.scales = scaleEigenvectors(solution.eigenvectors, method, dofMap, dofs, sink);
            solve.eigenvalues = solution.eigenvalues;
            solve.shifts = solution.shifts;
            solve.factorisationCount = solution.factorisationCount;
            solve.verifiedCount = solution.verifiedCount;
            solve.isComplete = solution.isComplete;
            if (!solution.isComplete)
                sink.report(incompleteProblem(first.subcase.label, method, solution));
            const std::size_t modes = solution.eigenvalues.size();
            output::OutputVolumeEstimate estimate(model.elementSet(), dofMap);
            for (const std::size_t s : group)
                for (std::size_t mode = 0; mode < modes; ++mode)
                    estimate.addCase(subcases[s].plan, system.gm().rowCount() > 0,
                                     [](const std::string&) { return true; });
            estimate.requireWithin(m_options.outputLimitBytes.value_or(
                static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available()))));

            progress = beginStage(m_options.monitor, "recovery", modes);
            const ModeRecovery recovery(model, system, "modes", m_options.recoveryRangeElements,
                                        m_options.isRecoveredCaseByCase);
            solve.eigenvectors = linalg::DenseMatrix(budget, gSize, modes, "eigenvectors");
            for (std::size_t mode = 0; mode < modes; ++mode)
            {
                if (cancellation != nullptr)
                    cancellation->throwIfCancelled();
                const base::BudgetedBuffer<double> modeG = recovery.expand(solution.eigenvectors, mode, dofs, budget);
                std::copy(modeG.data(), modeG.data() + gSize, &solve.eigenvectors(0, mode));
            }
            // The strain energies of every mode at once (ModeRecovery::strainEnergyTables), then each mode's tables.
            std::vector<std::vector<std::vector<output::ElementResultTable>>> energies;      // per subcase, per mode
            std::vector<std::vector<std::vector<output::ElementResultTable>>> elementTables; // the same
            for (const std::size_t s : group)
            {
                energies.push_back(recovery.strainEnergyTables(subcases[s].subcase.label, subcases[s].plan,
                                                               solve.eigenvectors.view(), budget, pool, cancellation));
                elementTables.push_back(recovery.elementTables(subcases[s].subcase.label, subcases[s].plan,
                                                               solve.eigenvectors.view(), budget, pool, cancellation));
            }
            for (std::size_t mode = 0; mode < modes; ++mode)
            {
                if (cancellation != nullptr)
                    cancellation->throwIfCancelled();
                const std::span<const double> modeG(&solve.eigenvectors(0, mode), gSize);
                for (std::size_t k = 0; k < group.size(); ++k)
                {
                    const std::size_t s = group[k];
                    recovery.recover(subcases[s].subcase.label, subcases[s].plan, modeG,
                                     static_cast<std::int32_t>(mode + 1), solution.eigenvalues[mode],
                                     mgg ? &*mgg : nullptr, -solution.eigenvalues[mode], budget, pool, cancellation,
                                     results.arrays, energies[k].empty() ? nullptr : &energies[k][mode],
                                     elementTables[k].empty() ? nullptr : &elementTables[k][mode]);
                }
                if (progress != nullptr)
                    progress->advance();
            }

            for (const std::size_t s : group)
            {
                const ModalSubcase& modal = subcases[s];
                ModesSubcaseResult result;
                result.subcase = modal.subcase.label;
                result.solve = results.solves.size();
                result.eigenvalues = eigenvalueTableOf(modal.subcase.label, solution.eigenvalues, solve.scales);
                const output::ModalEffectiveMassRequest& meff = modal.plan.modalEffectiveMass();
                if (meff.isRequested && mass)
                {
                    base::Vec3 referencePosition{};
                    std::int32_t referencePoint = 0;
                    if (meff.referencePoint > 0)
                    {
                        if (const std::optional<std::int32_t> row =
                                model.index().points().findGrid(meff.referencePoint))
                        {
                            referencePosition = dofMap.positionInBasic(*row);
                            referencePoint = meff.referencePoint;
                        }
                        else
                            sink.report(subcaseProblem(
                                modal.subcase.label, base::Severity::Warning, "modal-effective-mass-unknown-grid",
                                base::concatenate(
                                    {"ModalEffectiveMassOutput(GRID = ", std::to_string(meff.referencePoint),
                                     "): no such grid point; the reference point is the origin of the "
                                     "basic system"})));
                    }
                    result.modalEffectiveMass =
                        modalEffectiveMassOf(modal.subcase.label, referencePoint, referencePosition, model, dofs, *mass,
                                             solution.eigenvectors, solution.eigenvalues, solve.scales, budget, pool);
                }
                results.subcases.push_back(std::move(result));
            }
            results.timings.recovery += watch.lap();
        }
        if (ordering.equationCount() == 0)
            for (const std::size_t s : group)
            {
                ModesSubcaseResult result;
                result.subcase = subcases[s].subcase.label;
                result.solve = results.solves.size();
                result.eigenvalues.subcase = result.subcase;
                results.subcases.push_back(std::move(result));
            }
        results.solves.push_back(std::move(solve));
    }
    results.diagnostics = sink.diagnostics();
    return results;
}

} // namespace oos::solutions
