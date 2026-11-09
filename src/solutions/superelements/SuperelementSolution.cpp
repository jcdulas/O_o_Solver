#include "oos/solutions/superelements/SuperelementSolution.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/output/volume/OutputVolumeEstimate.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/EigenvectorScaling.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/common/ModeRecovery.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"
#include "oos/solutions/superelements/PartReduction.hpp"
#include "oos/solutions/superelements/PlacedSuperelement.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace oos::solutions
{

namespace
{

// 2π: cycles per unit time from radians per unit time.
constexpr double TWO_PI = 6.283185307179586;
// The share of the memory budget left when the recovery starts that the result tables may take (as the static and
// eigenvalue solutions).
constexpr double OUTPUT_BUDGET_FRACTION = 0.5;

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

base::Diagnostic runProblem(std::string code, std::string card, std::int64_t id, std::string message,
                            base::Severity severity = base::Severity::Error)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = std::move(card);
    diagnostic.id = id;
    diagnostic.message = std::move(message);
    return diagnostic;
}

base::ProgressCounter* beginStage(ProgressMonitor* monitor, const char* stage, std::uint64_t total)
{
    return monitor == nullptr ? nullptr : &monitor->beginStage(stage, total);
}

void checkCancelled(const base::CancellationToken* cancellation)
{
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

// One subcase of the residual structure (or of a part) with its outputs and, in a modal analysis, its method.
struct SubcaseOfModel
{
    StaticSubcase subcase;
    output::OutputPlan plan;
    std::optional<EigenMethod> method;
};

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

linalg::PivotPolicy pivotPolicyOf(const input::ParamTable* params, base::DiagnosticSink& sink)
{
    linalg::PivotPolicy policy;
    if (const std::optional<double> ratio = assembly::findRealParam(params, "MaxDiagonalRatio", std::nullopt, sink);
        ratio && *ratio > 0.0)
        policy.maxRatio = *ratio;
    if (const std::optional<std::int32_t> bailout =
            assembly::findIntegerParam(params, "SingularityBailOut", std::nullopt, sink))
        policy.bailout = *bailout;
    return policy;
}

// The static result tables of one case of a model (the residual structure or a part): DISPLACEMENT, OLOAD, SPCFORCES,
// MPCFORCES as the plan asks, the element tables (STRESS, STRAIN, FORCE) and ESE, as the static solution makes them.
struct StaticCase
{
    const StaticSubcase* subcase = nullptr;
    const output::OutputPlan* plan = nullptr;
    std::span<const double> displacements;
    std::span<const double> applied;
    std::span<const double> spcForces;
    std::span<const double> mpcForces;
    std::span<const std::vector<elements::ElementLineLoad>> lineLoads;
};

void appendStaticTables(const AnalysisModel& model, const StaticCase& state, base::MemoryBudget& budget,
                        base::ThreadPool* pool, const base::CancellationToken* cancellation,
                        output::ResultArrays& arrays)
{
    const StaticSubcase& subcase = *state.subcase;
    const output::OutputPlan& plan = *state.plan;
    const output::NodalResultBuilder nodalBuilder(model.dofMap());
    const auto addNodal = [&](output::OutputKind kind, std::span<const double> values)
    {
        const output::OutputRequestPlan& request = plan.of(kind);
        if (!request.isRequested || values.empty())
            return;
        arrays.nodalTables.push_back(nodalBuilder.build(std::string(output::tableKindOf(kind)), subcase.label, values,
                                                        budget, request.isAll, request.members));
    };
    addNodal(output::OutputKind::Displacement, state.displacements);
    addNodal(output::OutputKind::AppliedLoads, state.applied);
    addNodal(output::OutputKind::SpcForces, state.spcForces);
    addNodal(output::OutputKind::MpcForces, state.mpcForces);
    const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
    std::vector<output::ElementResultTable> elementTables; // bounded by the element types × kinds
    const output::ElementResultBuilder elementBuilder(model.elementSet(), model.dofMap());
    elementBuilder.build(subcase.label, state.displacements, state.lineLoads, budget, elementTables, pool, cancellation,
                         nullptr, &selection);
    if (plan.of(output::OutputKind::StrainEnergy).isRequested)
        output::StiffnessRecovery(model.elementSet(), model.dofMap())
            .strainEnergy(subcase.label, state.displacements, plan.of(output::OutputKind::StrainEnergy), budget,
                          elementTables, cancellation);
    for (output::ElementResultTable& table : elementTables)
        arrays.elementTables.push_back(std::move(table));
}

// The memory the result tables may take (the static and eigenvalue solutions' rule), checked before the recovery.
void requireOutputsFit(const AnalysisModel& model, const std::vector<const output::OutputPlan*>& plans,
                       std::size_t casesPerPlan, base::MemoryBudget& budget)
{
    output::OutputVolumeEstimate estimate(model.elementSet(), model.dofMap());
    for (const output::OutputPlan* plan : plans)
        for (std::size_t c = 0; c < casesPerPlan; ++c)
            estimate.addCase(*plan, true, [](const std::string&) { return true; });
    estimate.requireWithin(
        static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available())));
}

// What the residual structure is made of: its model, its subcases, the superelements placed on it and their parts.
struct Residual
{
    std::unique_ptr<AnalysisModel> model;
    std::vector<SubcaseOfModel> subcases;                    // those that apply to it
    std::vector<std::unique_ptr<PartReduction>> parts;       // per superelement of the options (null: external)
    std::vector<std::unique_ptr<PlacedSuperelement>> placed; // per superelement of the options
    std::vector<base::DofIndex> heldDofs;                    // sorted: held by a superelement at its boundary
    std::vector<base::DofIndex> modalDofs;                   // sorted: the generalised coordinates of the parts
    std::vector<std::int64_t> groupOffsets{0};
    std::vector<std::int32_t> groupNodes;
};

// The subcases of a model (its own selections of the shared Case Control) that `subcase` labels name.
std::vector<SubcaseOfModel> subcasesOf(const input::ModelSnapshot& snapshot, const std::vector<base::EntityId>& labels,
                                       AnalysisKind analysis, base::DiagnosticSink& sink)
{
    std::vector<SubcaseOfModel> result; // bounded by the subcases of the deck
    for (const StaticSubcase& subcase : staticSubcasesOf(snapshot, {}, sink))
    {
        if (std::find(labels.begin(), labels.end(), subcase.label) == labels.end())
            continue;
        SubcaseOfModel entry;
        entry.subcase = subcase;
        entry.plan = output::OutputPlan::forSubcase(snapshot, subcase.tableId, sink);
        if (analysis == AnalysisKind::Modes)
        {
            const std::optional<std::int32_t> methodId =
                snapshot.hasTable(input::TableRole::Subcases)
                    ? snapshot.subcases().selectedSet(subcase.tableId, "method")
                    : std::nullopt;
            if (methodId)
                entry.method = readEigenMethod(snapshot, *methodId, AnalysisKind::Modes, subcase.label, sink);
        }
        result.push_back(std::move(entry));
    }
    return result;
}

// The labels of the subcases that apply to a superelement.
std::vector<base::EntityId> labelsSelecting(const std::vector<SuperelementSubcase>& subcases,
                                            std::int32_t superelementId)
{
    std::vector<base::EntityId> labels; // bounded by the subcases
    for (const SuperelementSubcase& subcase : subcases)
        if (subcase.selects(superelementId))
            labels.push_back(subcase.subcase);
    return labels;
}

// The refusals of what a deck with superelements does not run in this version (the class comment).
void checkResidualSubcases(const Residual& residual, const SuperelementOptions& options, base::DiagnosticSink& sink)
{
    const input::ModelSnapshot& snapshot = residual.model->snapshot();
    for (const SubcaseOfModel& entry : residual.subcases)
    {
        const StaticSubcase& subcase = entry.subcase;
        if (subcase.isCombination)
            sink.report(runProblem("superelement-subcase-combination", "SubcaseStart", subcase.label,
                                   "SubcaseCombination in a deck with superelements is not supported by this "
                                   "version; remove it, "
                                   "or combine the results of its subcases afterwards"));
        if (subcase.temperatureLoad)
            sink.report(
                runProblem("superelement-thermal", "SubcaseStart", subcase.label,
                           "thermal loads (TemperatureSelection(LOAD)) in a deck with superelements are not supported "
                           "by this version; remove the thermal load"));
        if (subcase.deformation)
            sink.report(runProblem("superelement-thermal", "SubcaseStart", subcase.label,
                                   "element deformations (DeformationLoadSelection) in a deck with "
                                   "superelements are not supported by "
                                   "this version; remove the DeformationLoadSelection selection"));
        if (!assembly::supportDofsOf(snapshot, residual.model->dofMap(), subcase.tableId, sink).empty())
            sink.report(
                runProblem("superelement-free-body-support", "SubcaseStart", subcase.label,
                           "FreeBodySupport (inertia relief, the r set) in a deck with superelements is not supported "
                           "by this version; remove it"));
        if (options.analysis == AnalysisKind::Modes && !entry.method)
            sink.report(runProblem("missing-method", "SubcaseStart", subcase.label,
                                   "expected EigenMethodSelection = the set ID of an LanczosEigenMethod (or "
                                   "RealEigenMethod) card of "
                                   "the residual structure "
                                   "in the subcase or the global section of the real eigenvalue analysis, found none; "
                                   "add EigenMethodSelection to the Case Control"));
    }
    if (const std::optional<std::int32_t> inertiaRelief =
            assembly::findIntegerParam(residual.model->params(), "InertiaRelief", std::nullopt, sink);
        inertiaRelief && *inertiaRelief != 0)
        sink.report(
            runProblem("superelement-inertia-relief", "PARAM InertiaRelief", 0,
                       "inertia relief (PARAM InertiaRelief) in a deck with superelements is not supported by this "
                       "version; remove the parameter"));
    if (options.analysis == AnalysisKind::Statics)
        for (const SuperelementSubcase& subcase : options.subcases)
            if (!subcase.superelements.empty())
                sink.report(runProblem("superelement-super-static", "SubcaseStart", subcase.subcase,
                                       "SuperelementSelection selects some superelements in a static analysis: their "
                                       "loads would combine by load "
                                       "sequence, which this version does not support; use SuperelementSelection = "
                                       "ALL (or no SuperelementSelection) so that every "
                                       "subcase applies to every superelement"));
}

// Reduces every superelement and places it on the residual's DOFs.
void reduceSuperelements(const SuperelementOptions& options, const linalg::DenseKernels& kernels, Residual& residual,
                         SuperelementResults& results, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const assembly::DofMap& residualDofs = residual.model->dofMap();
    std::vector<base::EntityId> residualLabels; // the residual's subcases, the columns of the placed loads
    for (const SubcaseOfModel& entry : residual.subcases)
        residualLabels.push_back(entry.subcase.label);
    base::ProgressCounter* progress = beginStage(options.monitor, "superelements", options.parts.size());
    for (const SuperelementPart& part : options.parts)
    {
        checkCancelled(options.cancellation);
        SuperelementRecord record;
        record.superelementId = part.superelementId;
        record.subcases = labelsSelecting(options.subcases, part.superelementId);
        std::unique_ptr<PartReduction> reduction;
        std::unique_ptr<PlacedSuperelement> placed;
        if (part.external)
        {
            const ExternalSuperelementMatrices& matrices = *part.external;
            record.isExternal = true;
            record.boundaryDofCount = static_cast<std::int64_t>(matrices.labels.size());
            placed = std::make_unique<PlacedSuperelement>(part.superelementId, matrices.labels, part.boundary,
                                                          residualDofs, nullptr, sink);
            linalg::DenseMatrix stiffness(budget, matrices.labels.size(), matrices.labels.size(),
                                          "external superelement stiffness");
            for (std::size_t j = 0; j < matrices.labels.size(); ++j)
                for (std::size_t i = 0; i < matrices.labels.size(); ++i)
                    stiffness(i, j) = matrices.stiffness(i, j);
            placed->setStiffness(std::move(stiffness));
            if (options.analysis == AnalysisKind::Modes)
            {
                if (!matrices.mass)
                    sink.report(
                        runProblem("superelement-no-mass", "SuperelementBulkOptions", part.superelementId,
                                   base::concatenate({"external superelement ", std::to_string(part.superelementId),
                                                      " gives no mass matrix (MAA) for a modal analysis; "
                                                      "write it into its external matrix file"})));
                else
                {
                    linalg::DenseMatrix mass(budget, matrices.labels.size(), matrices.labels.size(),
                                             "external superelement mass");
                    for (std::size_t j = 0; j < matrices.labels.size(); ++j)
                        for (std::size_t i = 0; i < matrices.labels.size(); ++i)
                            mass(i, j) = (*matrices.mass)(i, j);
                    placed->setMass(std::move(mass));
                }
            }
            // No load: the loads of an external superelement are those of its creation (its PA), which an assembly
            // run applies only through a load sequence this version does not read; the residual's load sets do not
            // apply to it.
            placed->setLoads(linalg::DenseMatrix(budget, matrices.labels.size(), residualLabels.size(),
                                                 "external superelement loads"));
        }
        else
        {
            PartReduction::Inputs inputs;
            inputs.part = &part;
            inputs.analysis = options.analysis;
            inputs.subcases = record.subcases;
            inputs.kernels = &kernels;
            inputs.pool = options.pool;
            inputs.cancellation = options.cancellation;
            reduction = std::make_unique<PartReduction>(inputs, budget, sink);
            const reduction::ModalReduction* modal = reduction->modal();
            const std::size_t boundary = reduction->boundaryLabels().size();
            const std::size_t modes = modal != nullptr ? modal->modeCount() : 0;
            std::vector<reduction::UnitDofLabel> labels(reduction->boundaryLabels().begin(),
                                                        reduction->boundaryLabels().end());
            for (std::size_t q = 0; q < modes; ++q)
                labels.push_back(reduction::UnitDofLabel{reduction::UnitPointKind::Scalar, reduction->modalPoints()[q],
                                                         reduction::FastenerSide::None, 0});
            placed = std::make_unique<PlacedSuperelement>(part.superelementId, labels, part.boundary, residualDofs,
                                                          &reduction->model().dofMap(), sink);
            const std::size_t rows = boundary + modes;
            // K_CB = [K* 0; 0 K_qq] (K_bq = 0: Ψ̃ᵀ·K·Φ̃ = 0, reduction::ModalReduction).
            linalg::DenseMatrix stiffness(budget, rows, rows, "superelement stiffness");
            const linalg::DenseMatrix& reduced = reduction->stiffness();
            for (std::size_t j = 0; j < boundary; ++j)
                for (std::size_t i = 0; i < boundary; ++i)
                    stiffness(i, j) = reduced(i, j);
            for (std::size_t j = 0; j < modes; ++j)
                for (std::size_t i = 0; i < modes; ++i)
                    stiffness(boundary + i, boundary + j) = modal->modalStiffness()(i, j);
            placed->setStiffness(std::move(stiffness));
            if (options.analysis == AnalysisKind::Modes)
            {
                // M_CB = [M_bb M_bq; M_qb M_qq].
                linalg::DenseMatrix mass(budget, rows, rows, "superelement mass");
                const linalg::DenseMatrix& boundaryMass = reduction->boundaryMass();
                for (std::size_t j = 0; j < boundary; ++j)
                    for (std::size_t i = 0; i < boundary; ++i)
                        mass(i, j) = boundaryMass(i, j);
                for (std::size_t q = 0; q < modes; ++q)
                {
                    for (std::size_t b = 0; b < boundary; ++b)
                    {
                        mass(b, boundary + q) = modal->couplingMass()(b, q);
                        mass(boundary + q, b) = modal->couplingMass()(b, q);
                    }
                    for (std::size_t r = 0; r < modes; ++r)
                        mass(boundary + r, boundary + q) = modal->modalMass()(r, q);
                }
                placed->setMass(std::move(mass));
                for (std::size_t q = 0; q < modes; ++q)
                    record.fixedInterfaceEigenvalues.push_back(modal->eigenvalues()(q, 0));
            }
            // f* of each residual subcase that applies to the part (the same label), zero for the others.
            linalg::DenseMatrix loads(budget, rows, residualLabels.size(), "superelement loads");
            const std::span<const StaticSubcase> own = reduction->subcases();
            const linalg::DenseMatrix& condensedLoads = reduction->loads();
            for (std::size_t c = 0; c < residualLabels.size(); ++c)
                for (std::size_t s = 0; s < own.size() && s < condensedLoads.columns(); ++s)
                    if (own[s].label == residualLabels[c])
                        for (std::size_t b = 0; b < boundary; ++b)
                            loads(b, c) = condensedLoads(b, s);
            placed->setLoads(std::move(loads));
            // The generalised coordinates are never singular (the q set: no AUTOSPC on them).
            for (std::size_t q = 0; q < modes; ++q)
                if (placed->dofs()[boundary + q] >= 0)
                    residual.modalDofs.push_back(placed->dofs()[boundary + q]);
            record.boundaryDofCount = static_cast<std::int64_t>(boundary);
            record.interiorDofCount = reduction->interiorDofCount();
            record.modalCoordinateCount = static_cast<std::int64_t>(modes);
            // The boundary DOFs the part holds: held in the residual structure too (the permanent constraints of the
            // part and of the residual structure are united).
            for (const reduction::UnitDofLabel& label : reduction->heldBoundaryLabels())
            {
                const auto found = std::lower_bound(part.boundary.begin(), part.boundary.end(), label.id,
                                                    [](const SuperelementBoundaryPoint& point, base::EntityId id)
                                                    { return point.nodeId < id; });
                if (found == part.boundary.end() || found->nodeId != label.id)
                    continue;
                const std::optional<base::DofIndex> dof = residualDofs.findDof(
                    found->residualNodeId, label.kind == reduction::UnitPointKind::Scalar ? 0 : label.component);
                if (dof)
                    residual.heldDofs.push_back(*dof);
            }
        }
        for (const std::int32_t node : placed->nodes())
            residual.groupNodes.push_back(node);
        residual.groupOffsets.push_back(static_cast<std::int64_t>(residual.groupNodes.size()));
        residual.parts.push_back(std::move(reduction));
        residual.placed.push_back(std::move(placed));
        results.superelements.push_back(std::move(record));
        if (progress != nullptr)
            progress->advance();
    }
    std::sort(residual.modalDofs.begin(), residual.modalDofs.end());
    std::sort(residual.heldDofs.begin(), residual.heldDofs.end());
    residual.heldDofs.erase(std::unique(residual.heldDofs.begin(), residual.heldDofs.end()), residual.heldDofs.end());
}

// The options of the residual's constrained system: its superelements' cliques, K added to Kgg, their held DOFs.
ConstrainedSystemOptions residualSystemOptions(const Residual& residual)
{
    ConstrainedSystemOptions options;
    options.heldDofs = residual.heldDofs;
    options.uncheckedDofs = residual.modalDofs;
    options.extraCouplings = linalg::NodeGroupList{residual.groupOffsets, residual.groupNodes};
    const Residual* owner = &residual;
    options.addStiffness = [owner](linalg::BlockSparseMatrix& kgg)
    {
        for (const std::unique_ptr<PlacedSuperelement>& placed : owner->placed)
            placed->addStiffnessTo(kgg, owner->model->dofMap());
    };
    return options;
}

// The part subcase `label`'s plan, read in the part's snapshot (a SET there names the part's own points).
struct PartCase
{
    std::size_t subcaseIndex = 0; // into PartReduction::subcases()
    output::OutputPlan plan;
};

std::vector<PartCase> partCasesOf(const PartReduction& part, base::DiagnosticSink& sink)
{
    std::vector<PartCase> cases; // bounded by the part's subcases
    const std::span<const StaticSubcase> subcases = part.subcases();
    for (std::size_t s = 0; s < subcases.size(); ++s)
        cases.push_back({s, output::OutputPlan::forSubcase(part.model().snapshot(), subcases[s].tableId, sink)});
    return cases;
}

// The constrained system of a part for its SPC and MPC forces (its Kgg, Gm and held DOFs), when a plan asks for them.
std::unique_ptr<ConstrainedSystem> partSystemFor(const PartReduction& part, const std::vector<PartCase>& cases,
                                                 const SuperelementOptions& options, base::MemoryBudget& budget,
                                                 base::DiagnosticSink& sink)
{
    bool needed = false;
    for (const PartCase& entry : cases)
        needed = needed || entry.plan.of(output::OutputKind::SpcForces).isRequested ||
                 entry.plan.of(output::OutputKind::MpcForces).isRequested;
    if (!needed || part.subcases().empty())
        return nullptr;
    const StaticSubcase& first = part.subcases().front();
    // The part's own singularity table and warnings are those of its reduction (its interior): this system only gives
    // the forces, and its sink keeps nothing but the errors.
    base::DiagnosticSink local;
    auto system = std::make_unique<ConstrainedSystem>(
        part.model(), ConstraintSelection{first.label, first.spc, first.mpc, first.load, first.contact}, budget,
        options.pool, options.cancellation, local);
    for (const base::Diagnostic& diagnostic : local.diagnostics())
        if (diagnostic.severity == base::Severity::Error)
            sink.report(diagnostic);
    return system;
}

// --- statics -----------------------------------------------------------------------------------------------------

void solveStatics(const SuperelementOptions& options, const linalg::DenseKernels& kernels, Residual& residual,
                  SuperelementResults& results, base::MemoryBudget& budget, base::DiagnosticSink& sink,
                  Stopwatch& watch)
{
    const AnalysisModel& model = *residual.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const input::ModelSnapshot& snapshot = model.snapshot();
    const assembly::LoadVectorBuilder loadBuilder(model.elementSet(), dofMap, budget);
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const std::size_t subcaseCount = residual.subcases.size();
    const std::vector<std::vector<elements::ElementLineLoad>> noLineLoads;
    std::vector<assembly::SpcSet> spcSets; // bounded by the subcases
    spcSets.reserve(subcaseCount);
    for (const SubcaseOfModel& entry : residual.subcases)
        spcSets.emplace_back(constraints, snapshot.findLoads(), dofMap, entry.subcase.spc, entry.subcase.load, budget,
                             sink);
    stopOnErrors(sink, "the constraints of the residual structure");

    // The displacements of every subcase on the residual's g set, then each superelement recovered from them.
    linalg::DenseMatrix displacements(budget, gSize, subcaseCount, "residual displacements");
    std::vector<std::vector<std::size_t>> groups; // the subcases of one constraint set (bounded by the subcases)
    for (std::size_t s = 0; s < subcaseCount; ++s)
    {
        bool isPlaced = false;
        for (std::vector<std::size_t>& group : groups)
        {
            const StaticSubcase& first = residual.subcases[group.front()].subcase;
            if (first.spc == residual.subcases[s].subcase.spc && first.mpc == residual.subcases[s].subcase.mpc)
            {
                group.push_back(s);
                isPlaced = true;
                break;
            }
        }
        if (!isPlaced)
            groups.push_back({s});
    }
    SuperelementOutput residualOutput;
    residualOutput.superelementId = RESIDUAL_STRUCTURE;
    const ConstrainedSystemOptions systemOptions = residualSystemOptions(residual);
    for (const std::vector<std::size_t>& group : groups)
    {
        checkCancelled(options.cancellation);
        const StaticSubcase& first = residual.subcases[group.front()].subcase;
        ConstrainedSystemOptions ownOptions = systemOptions;
        ownOptions.spc = &spcSets[group.front()];
        const ConstrainedSystem system(
            model, ConstraintSelection{first.label, first.spc, first.mpc, first.load, first.contact}, budget,
            options.pool, options.cancellation, sink, ownOptions);
        if (!system.singularityTable().empty())
            results.singularityTables.push_back(system.singularityTable());
        FsetSolverOptions solverOptions;
        solverOptions.pivotPolicy = pivotPolicyOf(model.params(), sink);
        solverOptions.cancellation = options.cancellation;
        const FsetSolver solver(system.knn(), system.graph(), dofMap, system.solveSets(), kernels, solverOptions,
                                budget, options.pool);
        results.largestAnalysisSetSize = std::max<std::int64_t>(results.largestAnalysisSetSize, solver.equationCount());
        const auto equations = static_cast<std::size_t>(solver.equationCount());

        // The loads of the residual's own Bulk Data (with Mgg for inertial loads: GravityLoad acts on the residual's
        // elements; a part's own GravityLoad is its own load) and the superelements' f*.
        bool needsMass = false;
        for (const std::size_t s : group)
            needsMass = needsMass || (residual.subcases[s].subcase.load &&
                                      loadBuilder.needsMass(*residual.subcases[s].subcase.load));
        std::unique_ptr<linalg::BlockSparseMatrix> mass;
        if (needsMass)
        {
            mass = std::make_unique<linalg::BlockSparseMatrix>(system.pattern(), budget, "residual mass Mgg");
            const assembly::NonStructuralMassShares shares(model.elementSet());
            assembly::MassAssembler(model.elementSet(), dofMap, shares,
                                    assembly::MassAssemblyOptions::fromParams(model.params(), first.label, sink))
                .assemble(*mass, options.pool, options.cancellation);
        }
        StaticLoadCases::Inputs caseInputs;
        caseInputs.snapshot = &snapshot;
        caseInputs.points = &model.index().points();
        caseInputs.dofMap = &dofMap;
        caseInputs.loads = &loadBuilder;
        for (const std::size_t s : group)
        {
            caseInputs.cases.push_back(&residual.subcases[s].subcase);
            caseInputs.spcs.push_back(&spcSets[s]);
        }
        caseInputs.mass = mass.get();
        caseInputs.pool = options.pool;
        const StaticLoadCases cases(std::move(caseInputs), budget, sink);
        linalg::DenseMatrix rightHandSides(budget, equations, group.size(), "residual right-hand sides");
        linalg::DenseMatrix applied(budget, gSize, group.size(), "residual applied loads");
        linalg::DenseMatrix enforcedOf(budget, gSize, group.size(), "residual enforced displacements");
        {
            base::BudgetedBuffer<double> loads(budget, gSize, "residual loads Pg");
            base::BudgetedBuffer<double> enforced(budget, gSize, "residual enforced displacements");
            base::BudgetedBuffer<double> reduced(budget, gSize, "residual loads Pn");
            base::BudgetedBuffer<double> kus(budget, gSize, "residual Knn us");
            for (std::size_t c = 0; c < group.size(); ++c)
            {
                cases.exactCase(c, loads.span(), enforced.span(), budget, sink);
                // OLOAD of the residual: its own loads (each superelement prints its own).
                for (std::size_t dof = 0; dof < gSize; ++dof)
                {
                    applied(dof, c) = loads[dof];
                    enforcedOf(dof, c) = enforced[dof];
                }
                for (const std::unique_ptr<PlacedSuperelement>& placed : residual.placed)
                    placed->addLoadsTo(group[c], loads.span());
                std::copy(loads.data(), loads.data() + gSize, reduced.data());
                system.gm().reduceToN(reduced.span());
                // Pf = P̄n,f + (Gmᵀ·Pm)_f − K_fs·u_s (the static solution's reduction of the loads).
                std::fill(kus.data(), kus.data() + gSize, 0.0);
                system.knn().multiply(enforced.span(), kus.span());
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    reduced[dof] -= kus[dof];
                for (std::size_t e = 0; e < equations; ++e)
                    rightHandSides(e, c) =
                        reduced[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))];
            }
        }
        stopOnErrors(sink, "the loads of the residual structure");
        solver.solve(rightHandSides.view(), options.cancellation);

        // ug, then the SPC and MPC forces: Kgg·u − P on the held DOFs (Kgg with the superelements' K), the MPC forces
        // from Gm (the static solution's rules), and the residual's tables.
        const std::span<const std::uint8_t> isConstrained = system.isConstrained();
        base::BudgetedBuffer<double> ug(budget, gSize, "residual displacements ug");
        base::BudgetedBuffer<double> loadsWithParts(budget, gSize, "residual loads with the superelements");
        base::BudgetedBuffer<double> kug(budget, gSize, "residual Kgg ug");
        base::BudgetedBuffer<double> spcForces(budget, gSize, "residual SPC forces");
        base::BudgetedBuffer<double> mpcForces(budget, gSize, "residual MPC forces");
        for (std::size_t c = 0; c < group.size(); ++c)
        {
            checkCancelled(options.cancellation);
            const std::size_t s = group[c];
            std::fill(ug.data(), ug.data() + gSize, 0.0);
            for (std::size_t e = 0; e < equations; ++e)
                ug[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))] = rightHandSides(e, c);
            for (std::size_t dof = 0; dof < gSize; ++dof)
                if (enforcedOf(dof, c) != 0.0)
                    ug[dof] = enforcedOf(dof, c);
            system.gm().expandToG(ug.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
            {
                displacements(dof, s) = ug[dof];
                loadsWithParts[dof] = applied(dof, c);
            }
            for (const std::unique_ptr<PlacedSuperelement>& placed : residual.placed)
                placed->addLoadsTo(s, loadsWithParts.span());
            std::fill(kug.data(), kug.data() + gSize, 0.0);
            system.kgg().multiply(ug.span(), kug.span());
            std::fill(mpcForces.data(), mpcForces.data() + gSize, 0.0);
            system.gm().computeMpcForces(kug.span(), loadsWithParts.span(), mpcForces.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
                spcForces[dof] = isConstrained[dof] != 0 ? kug[dof] - loadsWithParts[dof] - mpcForces[dof] : 0.0;
            const SubcaseOfModel& entry = residual.subcases[s];
            const std::vector<std::vector<elements::ElementLineLoad>> lineLoads =
                entry.subcase.load ? loadBuilder.lineLoadsOf(*entry.subcase.load) : noLineLoads;
            StaticCase state;
            state.subcase = &entry.subcase;
            state.plan = &entry.plan;
            state.displacements = ug.span();
            state.applied = std::span<const double>(&applied(0, c), gSize);
            state.spcForces = spcForces.span();
            state.mpcForces =
                system.gm().rowCount() > 0 ? std::span<const double>(mpcForces.span()) : std::span<const double>();
            state.lineLoads = lineLoads;
            appendStaticTables(model, state, budget, options.pool, options.cancellation, residualOutput.arrays);
        }
    }
    results.residualSeconds += watch.lap();
    results.outputs.push_back(std::move(residualOutput));

    // The superelements: each part recovered from the residual's displacements in its subcases (the same labels).
    for (std::size_t p = 0; p < residual.parts.size(); ++p)
    {
        SuperelementOutput output;
        output.superelementId = options.parts[p].superelementId;
        const PartReduction* part = residual.parts[p].get();
        if (part == nullptr)
        {
            results.outputs.push_back(std::move(output));
            continue; // an external superelement has no table
        }
        const std::vector<PartCase> cases = partCasesOf(*part, sink);
        const std::unique_ptr<ConstrainedSystem> partSystem = partSystemFor(*part, cases, options, budget, sink);
        const assembly::DofMap& partDofs = part->model().dofMap();
        const auto partSize = static_cast<std::size_t>(partDofs.gSetSize());
        const assembly::LoadVectorBuilder partLoads(part->model().elementSet(), partDofs, budget);
        const PlacedSuperelement& placed = *residual.placed[p];
        const std::size_t boundary = part->boundaryLabels().size();
        for (const PartCase& entry : cases)
        {
            checkCancelled(options.cancellation);
            const StaticSubcase& subcase = part->subcases()[entry.subcaseIndex];
            std::size_t residualColumn = subcaseCount;
            for (std::size_t s = 0; s < subcaseCount; ++s)
                if (residual.subcases[s].subcase.label == subcase.label)
                    residualColumn = s;
            if (residualColumn == subcaseCount)
                continue; // a subcase of the part only: no residual solve gives its boundary (refused for statics)
            const linalg::DenseMatrix values =
                placed.gather(displacements.view().block(0, residualColumn, gSize, 1), budget);
            const std::size_t subcaseIndex = entry.subcaseIndex;
            const linalg::DenseMatrix ug = part->recoverStatic(values.view().block(0, 0, boundary, 1),
                                                               std::span<const std::size_t>(&subcaseIndex, 1), budget);
            const std::span<const double> partU(ug.data(), partSize);
            const std::span<const double> partApplied = part->appliedLoads(entry.subcaseIndex);
            base::BudgetedBuffer<double> partSpc(budget, partSize, "superelement SPC forces");
            base::BudgetedBuffer<double> partMpc(budget, partSize, "superelement MPC forces");
            bool hasMset = false;
            if (partSystem)
            {
                base::BudgetedBuffer<double> partKu(budget, partSize, "superelement Kgg ug");
                partSystem->kgg().multiply(partU, partKu.span());
                partSystem->gm().computeMpcForces(partKu.span(), partApplied, partMpc.span());
                const std::span<const std::uint8_t> held = partSystem->isConstrained();
                for (std::size_t dof = 0; dof < partSize; ++dof)
                    partSpc[dof] = held[dof] != 0 ? partKu[dof] - partApplied[dof] - partMpc[dof] : 0.0;
                hasMset = partSystem->gm().rowCount() > 0;
            }
            const std::vector<std::vector<elements::ElementLineLoad>> lineLoads =
                subcase.load ? partLoads.lineLoadsOf(*subcase.load) : noLineLoads;
            StaticCase state;
            state.subcase = &subcase;
            state.plan = &entry.plan;
            state.displacements = partU;
            state.applied = partApplied;
            state.spcForces = partSystem ? std::span<const double>(partSpc.span()) : std::span<const double>();
            state.mpcForces = hasMset ? std::span<const double>(partMpc.span()) : std::span<const double>();
            state.lineLoads = lineLoads;
            appendStaticTables(part->model(), state, budget, options.pool, options.cancellation, output.arrays);
        }
        results.outputs.push_back(std::move(output));
    }
    results.recoverySeconds += watch.lap();
}

// --- modes -------------------------------------------------------------------------------------------------------

// One eigenvalue solve of the residual structure: its subcases, its roots, its normalised vectors on the residual's g
// set (g × modes), their scales.
struct ResidualModes
{
    std::vector<std::size_t> subcases; // indices into Residual::subcases
    std::vector<double> eigenvalues;
    std::vector<double> scales;
    linalg::DenseMatrix vectors;
};

bool isSameSolve(const SubcaseOfModel& left, const SubcaseOfModel& right)
{
    return left.subcase.spc == right.subcase.spc && left.subcase.mpc == right.subcase.mpc && left.method &&
           right.method && left.method->card == right.method->card && left.method->setId == right.method->setId;
}

void solveModes(const SuperelementOptions& options, const linalg::DenseKernels& kernels, Residual& residual,
                SuperelementResults& results, base::MemoryBudget& budget, base::DiagnosticSink& sink, Stopwatch& watch)
{
    const AnalysisModel& model = *residual.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    std::vector<std::vector<std::size_t>> groups; // the subcases of one solve (bounded by the subcases)
    for (std::size_t s = 0; s < residual.subcases.size(); ++s)
    {
        bool isPlaced = false;
        for (std::vector<std::size_t>& group : groups)
            if (isSameSolve(residual.subcases[group.front()], residual.subcases[s]))
            {
                group.push_back(s);
                isPlaced = true;
                break;
            }
        if (!isPlaced)
            groups.push_back({s});
    }
    SuperelementOutput residualOutput;
    residualOutput.superelementId = RESIDUAL_STRUCTURE;
    std::vector<ResidualModes> solved; // bounded by the groups
    const ConstrainedSystemOptions systemOptions = residualSystemOptions(residual);
    for (const std::vector<std::size_t>& group : groups)
    {
        checkCancelled(options.cancellation);
        const SubcaseOfModel& first = residual.subcases[group.front()];
        const EigenMethod& method = *first.method;
        const ConstrainedSystem system(model,
                                       ConstraintSelection{first.subcase.label, first.subcase.spc, first.subcase.mpc,
                                                           std::nullopt, first.subcase.contact},
                                       budget, options.pool, options.cancellation, sink, systemOptions);
        if (!system.singularityTable().empty())
            results.singularityTables.push_back(system.singularityTable());
        // Mgg: the residual's elements and the superelements' reduced masses M_CB.
        linalg::BlockSparseMatrix mgg(system.pattern(), budget, "residual mass Mgg");
        {
            const assembly::NonStructuralMassShares shares(model.elementSet());
            assembly::MassAssembler(
                model.elementSet(), dofMap, shares,
                assembly::MassAssemblyOptions::fromParams(model.params(), first.subcase.label, sink))
                .assemble(mgg, options.pool, options.cancellation);
            for (const std::unique_ptr<PlacedSuperelement>& placed : residual.placed)
                placed->addMassTo(mgg, dofMap);
        }
        const linalg::BlockSparseMatrix mnn = system.reduce(mgg, budget, "residual mass Mnn");
        const FsetOrdering ordering(system.graph(), dofMap, system.solveSets(), assembly::DofSet::F,
                                    linalg::OrderingMethod::Automatic, budget, options.pool, options.cancellation);
        results.largestAnalysisSetSize =
            std::max<std::int64_t>(results.largestAnalysisSetSize, ordering.equationCount());
        ResidualModes modes;
        modes.subcases = group;
        SuperelementSolve solve;
        solve.method = method;
        solve.analysisSetSize = ordering.equationCount();
        for (const std::size_t s : group)
            solve.subcases.push_back(residual.subcases[s].subcase.label);
        if (ordering.equationCount() == 0)
        {
            sink.report(runProblem("no-free-dof", "SubcaseStart", first.subcase.label,
                                   "every DOF of the residual structure is constrained: the analysis set is empty and "
                                   "there is no mode",
                                   base::Severity::Warning));
            modes.vectors = linalg::DenseMatrix(budget, gSize, 0, "residual eigenvectors");
        }
        else
        {
            const linalg::DofSelection& dofs = ordering.selection();
            const linalg::SymmetricCsc stiffness =
                linalg::SymmetricCsc::fromBlocks(system.knn(), dofs, budget, options.pool);
            const linalg::SymmetricCsc mass = linalg::SymmetricCsc::fromBlocks(mnn, dofs, budget, options.pool);
            linalg::FactorOptions factorOptions;
            factorOptions.equationNamer = [&dofMap, &dofs](std::int32_t equation)
            { return equationName(dofMap, static_cast<base::DofIndex>(dofs.dofOfEquation(equation))); };
            factorOptions.cancellation = options.cancellation;
            linalg::SparsePencil pencil(ordering.symbolic(), stiffness, mass, kernels, factorOptions, budget);
            linalg::EigenRequest request = eigenRequestOf(method, AnalysisKind::Modes);
            // A request of more roots than the analysis set has (an LanczosEigenMethod ND above the size of the reduced
            // residual): every root there is, without an error.
            const auto size = static_cast<std::size_t>(ordering.equationCount());
            if (request.modeCount > size)
                request.modeCount = size;
            linalg::LanczosOptions lanczos;
            if (method.blockSize > 0)
                lanczos.blockSize = method.blockSize;
            lanczos.control.cancellation = options.cancellation;
            lanczos.control.progress = beginStage(options.monitor, "eigenvalues", request.modeCount);
            linalg::EigenSolution solution = linalg::BlockLanczos(lanczos).solve(pencil, request, kernels, budget);
            modes.scales = scaleEigenvectors(solution.eigenvectors, method, dofMap, dofs, sink);
            modes.eigenvalues = solution.eigenvalues;
            solve.eigenvalues = solution.eigenvalues;
            solve.factorisationCount = solution.factorisationCount;
            solve.isComplete = solution.isComplete;
            if (!solution.isComplete)
                sink.report(runProblem(
                    "eigen-incomplete", "SubcaseStart", first.subcase.label,
                    base::concatenate({"the Lanczos solver reached its limit of shifts before the "
                                       "request of ",
                                       method.card, " ", std::to_string(method.setId),
                                       " was met; narrow the range or lower ", method.card, ".desiredModeCount"}),
                    base::Severity::Warning));
            const std::size_t count = solution.eigenvalues.size();
            std::vector<const output::OutputPlan*> plans; // bounded by the subcases
            for (const std::size_t s : group)
                plans.push_back(&residual.subcases[s].plan);
            requireOutputsFit(model, plans, count, budget);
            const ModeRecovery recovery(model, system, "modes");
            modes.vectors = linalg::DenseMatrix(budget, gSize, count, "residual eigenvectors");
            for (std::size_t mode = 0; mode < count; ++mode)
            {
                checkCancelled(options.cancellation);
                const base::BudgetedBuffer<double> modeG = recovery.expand(solution.eigenvectors, mode, dofs, budget);
                std::copy(modeG.data(), modeG.data() + gSize, &modes.vectors(0, mode));
            }
            // The strain energies of every mode at once (ModeRecovery::strainEnergyTables), then each mode's tables.
            std::vector<std::vector<std::vector<output::ElementResultTable>>> energies;      // per subcase, per mode
            std::vector<std::vector<std::vector<output::ElementResultTable>>> elementTables; // the same
            for (const std::size_t s : group)
            {
                energies.push_back(recovery.strainEnergyTables(residual.subcases[s].subcase.label,
                                                               residual.subcases[s].plan, modes.vectors.view(), budget,
                                                               options.pool, options.cancellation));
                elementTables.push_back(recovery.elementTables(residual.subcases[s].subcase.label,
                                                               residual.subcases[s].plan, modes.vectors.view(), budget,
                                                               options.pool, options.cancellation));
            }
            for (std::size_t mode = 0; mode < count; ++mode)
            {
                checkCancelled(options.cancellation);
                const std::span<const double> modeG(&modes.vectors(0, mode), gSize);
                for (std::size_t k = 0; k < group.size(); ++k)
                {
                    const std::size_t s = group[k];
                    recovery.recover(residual.subcases[s].subcase.label, residual.subcases[s].plan, modeG,
                                     static_cast<std::int32_t>(mode + 1), solution.eigenvalues[mode], &mgg,
                                     -solution.eigenvalues[mode], budget, options.pool, options.cancellation,
                                     residualOutput.arrays, energies[k].empty() ? nullptr : &energies[k][mode],
                                     elementTables[k].empty() ? nullptr : &elementTables[k][mode]);
                }
            }
        }
        for (const std::size_t s : group)
            residualOutput.eigenvalues.push_back(
                eigenvalueTableOf(residual.subcases[s].subcase.label, modes.eigenvalues, modes.scales));
        results.solves.push_back(std::move(solve));
        solved.push_back(std::move(modes));
    }
    results.residualSeconds += watch.lap();
    results.outputs.push_back(std::move(residualOutput));

    // The superelements: each part's mode shapes from the residual's (its boundary values and generalised coordinates),
    // in the subcases that apply to it: a subcase of the residual takes its own solve, a subcase of the part only the
    // residual's one solve.
    base::ProgressCounter* progress = beginStage(options.monitor, "recovery", residual.parts.size());
    for (std::size_t p = 0; p < residual.parts.size(); ++p)
    {
        SuperelementOutput output;
        output.superelementId = options.parts[p].superelementId;
        const PartReduction* part = residual.parts[p].get();
        if (part == nullptr)
        {
            results.outputs.push_back(std::move(output));
            continue; // an external superelement has no table
        }
        const std::vector<PartCase> cases = partCasesOf(*part, sink);
        const std::unique_ptr<ConstrainedSystem> partSystem = partSystemFor(*part, cases, options, budget, sink);
        std::unique_ptr<linalg::BlockSparseMatrix> partMass;
        if (partSystem)
        {
            partMass =
                std::make_unique<linalg::BlockSparseMatrix>(partSystem->pattern(), budget, "superelement mass Mgg");
            const assembly::NonStructuralMassShares shares(part->model().elementSet());
            assembly::MassAssembler(
                part->model().elementSet(), part->model().dofMap(), shares,
                assembly::MassAssemblyOptions::fromParams(part->model().params(), std::nullopt, sink))
                .assemble(*partMass, options.pool, options.cancellation);
        }
        const std::unique_ptr<ModeRecovery> recovery =
            partSystem ? std::make_unique<ModeRecovery>(part->model(), *partSystem, "modes")
                       : std::make_unique<ModeRecovery>(part->model(), "modes");
        const PlacedSuperelement& placed = *residual.placed[p];
        const std::size_t boundary = part->boundaryLabels().size();
        const std::size_t modalCount = part->modal() != nullptr ? part->modal()->modeCount() : 0;
        const auto partSize = static_cast<std::size_t>(part->model().dofMap().gSetSize());
        for (const PartCase& entry : cases)
        {
            checkCancelled(options.cancellation);
            const StaticSubcase& subcase = part->subcases()[entry.subcaseIndex];
            const ResidualModes* source = nullptr;
            for (const ResidualModes& modes : solved)
                for (const std::size_t s : modes.subcases)
                    if (residual.subcases[s].subcase.label == subcase.label)
                        source = &modes;
            if (source == nullptr)
            {
                if (solved.size() != 1)
                {
                    sink.report(runProblem(
                        "superelement-solve", "SubcaseStart", subcase.label,
                        base::concatenate({"subcase ", std::to_string(subcase.label), " applies to superelement ",
                                           std::to_string(part->superelementId()),
                                           " but not to the residual structure, which is solved ",
                                           std::to_string(solved.size()),
                                           " times: which modes the part's subcase takes is not defined; let the "
                                           "subcase apply to the residual structure too "
                                           "(SuperelementSelection), or solve the "
                                           "residual structure once"})));
                    continue;
                }
                source = &solved.front();
            }
            const std::size_t count = source->eigenvalues.size();
            if (count == 0)
                continue;
            requireOutputsFit(part->model(), {&entry.plan}, count, budget);
            const linalg::DenseMatrix values = placed.gather(source->vectors.view(), budget);
            const linalg::DenseMatrix shapes =
                part->recoverModes(values.view().block(0, 0, boundary, count),
                                   values.view().block(boundary, 0, modalCount, count), budget);
            for (std::size_t mode = 0; mode < count; ++mode)
                recovery->recover(subcase.label, entry.plan,
                                  std::span<const double>(shapes.data() + mode * shapes.leadingDimension(), partSize),
                                  static_cast<std::int32_t>(mode + 1), source->eigenvalues[mode], partMass.get(),
                                  -source->eigenvalues[mode], budget, options.pool, options.cancellation,
                                  output.arrays);
            output.eigenvalues.push_back(eigenvalueTableOf(subcase.label, source->eigenvalues, source->scales));
        }
        results.outputs.push_back(std::move(output));
        if (progress != nullptr)
            progress->advance();
    }
    results.recoverySeconds += watch.lap();
}

} // namespace

SuperelementResults SuperelementSolution::run(base::MemoryBudget& budget) const
{
    if (m_options.analysis != AnalysisKind::Statics && m_options.analysis != AnalysisKind::Modes)
        throw std::invalid_argument(
            "SuperelementSolution: a deck with superelements runs SOL 101 (SESTATIC) or SOL 103 (SEMODES) in this "
            "version; run it as one of them");
    SuperelementResults results;
    base::DiagnosticSink sink;
    Stopwatch watch;
    const linalg::DenseKernels kernels(m_options.kernel, linalg::DenseKernelOptions{m_options.pool});
    results.kernel = std::string(linalg::kernelName(m_options.kernel));

    Residual residual;
    residual.model = std::make_unique<AnalysisModel>(*m_residual, m_options.analysis, budget, m_options.pool,
                                                     m_options.cancellation, sink);
    results.residualGSetSize = residual.model->dofMap().gSetSize();
    residual.subcases =
        subcasesOf(*m_residual, labelsSelecting(m_options.subcases, RESIDUAL_STRUCTURE), m_options.analysis, sink);
    checkResidualSubcases(residual, m_options, sink);
    for (const SuperelementPart& part : m_options.parts)
        if (!part.snapshot && !part.external)
            throw std::invalid_argument(
                base::concatenate({"SuperelementSolution: superelement ", std::to_string(part.superelementId),
                                   " has neither a snapshot (a part) nor matrices (an external superelement); give "
                                   "one of them"}));
    stopOnErrors(sink, "the analysis of the superelements");

    reduceSuperelements(m_options, kernels, residual, results, budget, sink);
    stopOnErrors(sink, "the analysis of the residual structure");
    results.reductionSeconds = watch.lap();
    if (m_options.analysis == AnalysisKind::Statics)
        solveStatics(m_options, kernels, residual, results, budget, sink, watch);
    else
        solveModes(m_options, kernels, residual, results, budget, sink, watch);
    results.diagnostics = sink.diagnostics();
    return results;
}

} // namespace oos::solutions
