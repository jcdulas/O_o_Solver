#include "oos/design/problem/DesignProblem.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/model/DesignStamps.hpp"
#include "oos/design/problem/ShapeDistortion.hpp"
#include "oos/design/responses/ModeTracking.hpp"
#include "oos/design/sensitivity/StaticSensitivity.hpp"
#include "oos/solutions/common/RunChecks.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>

namespace oos::design
{

namespace
{

const input::ModelSnapshot& requireSnapshot(const std::shared_ptr<const input::ModelSnapshot>& snapshot)
{
    if (!snapshot)
        throw std::invalid_argument(
            "DesignProblem: expected a model snapshot, found none; take one with "
            "O_o_Solver.take_snapshot");
    return *snapshot;
}

// `outputs` with the requests of `extra` added (the union of their members).
void mergeOutputs(std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT>& outputs,
                  const std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT>& extra)
{
    for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
    {
        if (!extra[k].isRequested)
            continue;
        output::OutputRequestPlan& target = outputs[k];
        if (!target.isRequested)
        {
            target = extra[k];
            continue;
        }
        if (target.isAll || extra[k].isAll)
        {
            target.isAll = true;
            target.members.clear();
            continue;
        }
        std::vector<std::int32_t> members;
        std::set_union(target.members.begin(), target.members.end(), extra[k].members.begin(), extra[k].members.end(),
                       std::back_inserter(members));
        target.members = std::move(members);
    }
}

} // namespace

DesignProblem::DesignProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, DesignModelParts parts)
    : DesignProblem(std::move(snapshot), std::move(parts), {}, {})
{
}

DesignProblem::DesignProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, DesignModelParts parts,
                             ResponseParts responses, ConstraintParts constraints)
    : DesignProblem(std::move(snapshot), std::move(parts), std::move(responses), std::move(constraints), nullptr)
{
}

DesignProblem::DesignProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, DesignModelParts parts,
                             ResponseParts responses, ConstraintParts constraints,
                             std::shared_ptr<const SectionLibrary> sections)
    : m_snapshot(std::move(snapshot)),
      m_model(std::move(parts)),
      m_targets(m_model, requireSnapshot(m_snapshot), std::move(sections)),
      m_responses(std::move(responses), m_model, *m_snapshot),
      m_constraints(std::move(constraints), m_responses)
{
    if (m_snapshot->hasTable(input::TableRole::Subcases))
    {
        const auto& subcases = m_snapshot->subcases();
        for (std::int64_t row = 0; row < subcases.rowCount(); ++row)
            if (subcases.selectedSetAtRow(row, "nonStructMass"))
                throw base::OptimisationError(
                    "selected non-structural mass sets are supported in static runs, but their design derivatives "
                    "are not implemented; optimization with NonStructMassSelection is refused");
    }
}

std::vector<std::int64_t> DesignProblem::drivenUnits() const
{
    std::vector<std::int64_t> units; // bounded by the snapshot's units
    const std::vector<std::int64_t>& unitIds = m_targets.unitIds();
    for (const FieldTarget& target : m_targets.targets())
        for (const std::size_t u : target.units)
            units.push_back(unitIds[u]);
    for (const ShapeTarget& shape : m_targets.shapeTargets())
        for (const std::size_t u : shape.units)
            units.push_back(unitIds[u]);
    std::sort(units.begin(), units.end());
    units.erase(std::unique(units.begin(), units.end()), units.end());
    return units;
}

std::shared_ptr<const input::ModelSnapshot> DesignProblem::overlaySnapshot(const DesignPoint& point) const
{
    return std::make_shared<const input::ModelSnapshot>(point.applyTo(*m_snapshot, m_snapshot));
}

std::vector<base::Diagnostic> DesignProblem::distortionsOf(const input::ModelSnapshot& overlaid,
                                                           base::MemoryBudget& budget) const
{
    if (m_model.shapes().isEmpty())
        return {};
    std::call_once(m_baseProblems->once,
                   [&] { m_baseProblems->problems = ShapeDistortion::elementProblems(*m_snapshot, budget); });
    return ShapeDistortion::distortions(m_baseProblems->problems, ShapeDistortion::elementProblems(overlaid, budget));
}

std::vector<base::Diagnostic> DesignProblem::distortions(const DesignPoint& point, base::MemoryBudget& budget) const
{
    if (m_model.shapes().isEmpty())
        return {};
    return distortionsOf(*overlaySnapshot(point), budget);
}

DesignProblem::PreparedPoint DesignProblem::preparePoint(std::span<const double> x,
                                                         std::span<const SuppliedField> supplied,
                                                         base::MemoryBudget& budget) const
{
    DesignPoint point = designPoint(x, supplied);
    std::shared_ptr<const input::ModelSnapshot> overlaid = overlaySnapshot(point);
    // A design point that moves grid points: the elements it distorts (ShapeDistortion), before the analysis.
    std::vector<base::Diagnostic> distorted = distortionsOf(*overlaid, budget);
    {
        base::DiagnosticSink errors;
        for (const base::Diagnostic& distortion : distorted)
            if (distortion.severity == base::Severity::Error)
                errors.report(distortion);
        if (errors.hasErrors())
            solutions::stopOnErrors(errors, "the analysis of the design point");
    }
    return {std::move(point), std::move(overlaid), std::move(distorted)};
}

DesignProblem::PreparedStatics DesignProblem::prepareStatics(const DesignPoint& point,
                                                             const input::ModelSnapshot& overlaid,
                                                             const solutions::StaticOptions& options,
                                                             bool needsDisplacements) const
{
    solutions::StaticOptions withResponses = options;
    withResponses.usesCaseControlOutputs = withResponses.usesCaseControlOutputs && m_deckOutputs;
    mergeOutputs(withResponses.requiredOutputs, m_responses.requiredOutputs());
    // The section and cut loads of the SECTLOAD and CUTLOAD responses, computed by the analysis on its load basis.
    output::SectionLoadRequest sections = m_responses.sectionLoadRequest();
    if (!sections.isEmpty())
    {
        output::SectionLoadRequest& request = withResponses.sectionLoads;
        request.sections.insert(request.sections.end(), sections.sections.begin(), sections.sections.end());
        request.cuts.insert(request.cuts.end(), sections.cuts.begin(), sections.cuts.end());
        request.keepsValues = true;
    }
    // The zones of the contour and external responses, computed by the analysis on its load basis.
    output::ZoneRequest zones = m_responses.zoneRequest();
    if (!zones.isEmpty())
    {
        output::ZoneRequest& request = withResponses.zones;
        request.zones.insert(request.zones.end(), zones.zones.begin(), zones.zones.end());
        if (request.contourSubcases.empty())
            request.contourSubcases = zones.contourSubcases;
        request.keepsValues = request.keepsValues || zones.keepsValues;
        request.keepsBasis = request.keepsBasis || zones.keepsBasis;
    }
    // The condensed path (the designed units kept full by default, the others condensed once): the design stamp of
    // every unit the point drives (part of the cache key: a design point is an overlay, which O_o_Reader's stamps do
    // not see), and the driven units recovered (their pseudo-loads read the displacements of their elements).
    const bool isCondensed = options.condensation && !options.condensation->isFullPath();
    if (isCondensed)
    {
        const std::vector<std::uint64_t>& stamps = point.designStamps();
        const std::vector<std::int64_t>& unitIds = m_targets.unitIds();
        for (std::size_t k = 0; k < stamps.size() && k < unitIds.size(); ++k)
            if (stamps[k] != EMPTY_DESIGN_STAMP)
                withResponses.designStamps.push_back({unitIds[k], stamps[k]});
        withResponses.recoversDrivenUnits = true;
    }
    const solutions::StaticSolution probe(overlaid, withResponses);
    std::vector<std::int32_t> subcases = probe.subcasesToSolve();
    if (needsDisplacements)
    {
        // The gradients solve on the factors of the analysis (StaticSensitivity), kept with the evaluation.
        withResponses.keepsSystems = true;
        if (isCondensed)
        {
            // The g-set displacements of every subcase, kept as vectors: an output of every point would recover every
            // unit; the units a sensitivity reads are recovered (the driven ones, those the responses read).
            withResponses.keptDisplacements.insert(withResponses.keptDisplacements.end(), subcases.begin(),
                                                   subcases.end());
        }
        else
        {
            std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> displacements{};
            output::OutputRequestPlan& request =
                displacements[static_cast<std::size_t>(output::OutputKind::Displacement)];
            request.isRequested = true;
            request.isAll = true;
            mergeOutputs(withResponses.requiredOutputs, displacements);
        }
    }
    return {std::move(withResponses), std::move(subcases)};
}

StaticDesignEvaluation DesignProblem::evaluateStatics(std::span<const double> x,
                                                      const solutions::StaticOptions& options,
                                                      base::MemoryBudget& budget,
                                                      std::span<const SuppliedField> supplied,
                                                      bool needsDisplacements) const
{
    PreparedPoint prepared = preparePoint(x, supplied, budget);
    PreparedStatics statics = prepareStatics(prepared.point, *prepared.overlaid, options, needsDisplacements);
    const solutions::StaticSolution solution(*prepared.overlaid, statics.options);
    solutions::StaticResults results = solution.run(budget);
    results.diagnostics.insert(results.diagnostics.end(), prepared.distortions.begin(), prepared.distortions.end());
    return {std::move(prepared.point), std::move(prepared.overlaid), std::move(results), std::move(statics.subcases)};
}

bool DesignProblem::hasBucklingResponses() const
{
    return std::any_of(m_responses.parts().designResp1.begin(), m_responses.parts().designResp1.end(),
                       [](const DesignResp1Definition& definition)
                       { return definition.type == ResponseType::BucklingFactor; });
}

StaticDesignEvaluation DesignProblem::evaluateAnalyses(std::span<const double> x, const DesignAnalyses& analyses,
                                                       const solutions::StaticOptions& options,
                                                       base::MemoryBudget& budget,
                                                       std::span<const SuppliedField> supplied,
                                                       bool needsDisplacements) const
{
    if (!analyses.hasEigenvalues() && !analyses.hasAeroelastic())
    {
        if (!analyses.statics)
            throw std::invalid_argument(
                "DesignProblem::evaluateAnalyses: expected at least one analysis (STATICS, "
                "MODES, BUCK or SAERO), found none; name the subcases of the problem's analyses");
        solutions::StaticOptions staticOptions = options;
        if (!analyses.statics->empty())
            staticOptions.subcases = *analyses.statics;
        return evaluateStatics(x, staticOptions, budget, supplied, needsDisplacements);
    }
    PreparedPoint prepared = preparePoint(x, supplied, budget);
    StaticDesignEvaluation evaluation{std::move(prepared.point), std::move(prepared.overlaid), {}, {}};
    const input::ModelSnapshot& overlaid = *evaluation.snapshot;
    const bool isCondensed = options.condensation && !options.condensation->isFullPath();
    // The eigenvalue subcases are no static subcase: "every subcase" of the statics leaves them out.
    std::vector<std::int32_t> eigenSubcases;
    if (analyses.modes)
        eigenSubcases.insert(eigenSubcases.end(), analyses.modes->begin(), analyses.modes->end());
    if (analyses.buckling)
        eigenSubcases.insert(eigenSubcases.end(), analyses.buckling->begin(), analyses.buckling->end());
    // Nor are the SAERO subcases (trimmed by TrimSolution).
    if (analyses.aeroelastic)
        eigenSubcases.insert(eigenSubcases.end(), analyses.aeroelastic->begin(), analyses.aeroelastic->end());
    solutions::StaticOptions staticOptions = options;
    if (analyses.statics && !analyses.statics->empty())
        staticOptions.subcases = *analyses.statics;
    PreparedStatics statics = prepareStatics(evaluation.point, overlaid, staticOptions, needsDisplacements);
    std::vector<std::int32_t> staticSubcases;
    if (analyses.statics)
        for (const std::int32_t subcase : statics.subcases)
            if (std::find(eigenSubcases.begin(), eigenSubcases.end(), subcase) == eigenSubcases.end())
                staticSubcases.push_back(subcase);
    // The units a point drives (their design stamps) and those recover= names, recovered by the eigenvalue analyses
    // too: the derivatives read the mode shapes of their elements.
    const auto condensedOptions = [&](auto& eigen)
    {
        if (!isCondensed)
            return;
        eigen.condensation = options.condensation;
        eigen.cache = options.cache;
        eigen.cacheModelTag = options.cacheModelTag;
        eigen.designStamps = statics.options.designStamps;
        eigen.recoversDrivenUnits = true;
        eigen.alsoRecoveredUnits = options.alsoRecoveredUnits;
    };
    if (analyses.buckling)
    {
        // The buckling solution solves the static subcases as its preloads, with the options of the responses.
        solutions::BucklingOptions bucklingOptions;
        bucklingOptions.subcases = *analyses.buckling;
        bucklingOptions.pool = options.pool;
        bucklingOptions.kernel = options.kernel;
        bucklingOptions.cancellation = options.cancellation;
        bucklingOptions.monitor = options.monitor;
        bucklingOptions.structures = options.structures;
        condensedOptions(bucklingOptions);
        if (isCondensed && hasBucklingResponses())
        {
            // The derivative of a load factor reads the shape and the preload displacements of every element.
            for (const reduction::UnitPlan& unit : options.condensation->units())
                if (unit.decision == reduction::UnitDecision::Condensed)
                    bucklingOptions.alsoRecoveredUnits.push_back(unit.unitId);
        }
        bucklingOptions.preloadOptions = statics.options;
        solutions::BucklingResults buckling = solutions::BucklingSolution(overlaid, bucklingOptions).run(budget);
        evaluation.results = std::move(buckling.preload);
        buckling.preload = solutions::StaticResults{};
        for (const solutions::BucklingSolve& solve : buckling.solves)
        {
            evaluation.bucklingSubcases.push_back(solve.subcase);
            if (std::find(evaluation.preloadSubcases.begin(), evaluation.preloadSubcases.end(), solve.preloadSubcase) ==
                evaluation.preloadSubcases.end())
                evaluation.preloadSubcases.push_back(solve.preloadSubcase);
        }
        evaluation.buckling.emplace(std::move(buckling));
    }
    else if (!staticSubcases.empty())
    {
        statics.options.subcases = staticSubcases;
        evaluation.results = solutions::StaticSolution(overlaid, statics.options).run(budget);
    }
    if (analyses.modes)
    {
        solutions::ModesOptions modesOptions;
        modesOptions.subcases = *analyses.modes;
        modesOptions.pool = options.pool;
        modesOptions.kernel = options.kernel;
        modesOptions.cancellation = options.cancellation;
        modesOptions.monitor = options.monitor;
        modesOptions.structures = options.structures;
        condensedOptions(modesOptions);
        solutions::ModesResults modes = solutions::ModesSolution(overlaid, modesOptions).run(budget);
        for (const solutions::ModesSubcaseResult& subcase : modes.subcases)
            evaluation.modesSubcases.push_back(subcase.subcase);
        // The diagnostics of the analyses together (the warnings of a point, whatever its analysis).
        evaluation.results.diagnostics.insert(evaluation.results.diagnostics.end(), modes.diagnostics.begin(),
                                              modes.diagnostics.end());
        evaluation.modes.emplace(std::move(modes));
    }
    if (analyses.aeroelastic)
    {
        // The trim of the SAERO subcases, kept as a function of the structure for the gradients; on the
        // condensed path the plan keeps the spline nodes on the units' boundaries (the caller retains them).
        solutions::TrimOptions trimOptions;
        trimOptions.subcases.assign(analyses.aeroelastic->begin(), analyses.aeroelastic->end());
        trimOptions.links = analyses.links;
        trimOptions.pool = options.pool;
        trimOptions.kernel = options.kernel;
        trimOptions.cancellation = options.cancellation;
        trimOptions.monitor = options.monitor;
        trimOptions.requiredOutputs = statics.options.requiredOutputs;
        trimOptions.keepsLinearisation = needsDisplacements;
        if (isCondensed)
        {
            trimOptions.condensation = options.condensation;
            trimOptions.cache = options.cache;
            trimOptions.cacheModelTag = options.cacheModelTag;
        }
        solutions::TrimResults trim = solutions::TrimSolution(overlaid, trimOptions).run(budget);
        for (const solutions::TrimSubcaseResult& subcase : trim.subcases)
            evaluation.aeroelasticSubcases.push_back(subcase.subcase);
        output::ResultArrays& arrays = evaluation.results.arrays;
        for (output::NodalResultTable& table : trim.arrays.nodalTables)
            arrays.nodalTables.push_back(std::move(table));
        for (output::ElementResultTable& table : trim.arrays.elementTables)
            arrays.elementTables.push_back(std::move(table));
        for (output::GridPointForceTable& table : trim.arrays.gridPointForceTables)
            arrays.gridPointForceTables.push_back(std::move(table));
        trim.arrays = output::ResultArrays{};
        evaluation.results.diagnostics.insert(evaluation.results.diagnostics.end(), trim.diagnostics.begin(),
                                              trim.diagnostics.end());
        evaluation.trim.emplace(std::move(trim));
    }
    if (evaluation.buckling)
        evaluation.results.diagnostics.insert(evaluation.results.diagnostics.end(),
                                              evaluation.buckling->diagnostics.begin(),
                                              evaluation.buckling->diagnostics.end());
    evaluation.results.diagnostics.insert(evaluation.results.diagnostics.end(), prepared.distortions.begin(),
                                          prepared.distortions.end());
    evaluation.subcases = std::move(staticSubcases);
    evaluation.staticSubcasesRead = evaluation.subcases;
    for (const std::int32_t subcase : evaluation.preloadSubcases)
        if (std::find(evaluation.staticSubcasesRead.begin(), evaluation.staticSubcasesRead.end(), subcase) ==
            evaluation.staticSubcasesRead.end())
            evaluation.staticSubcasesRead.push_back(subcase);
    return evaluation;
}

std::vector<std::int32_t> DesignProblem::analysedSubcasesOf(const StaticDesignEvaluation& analysis)
{
    std::vector<std::int32_t> subcases = analysis.subcases;
    subcases.insert(subcases.end(), analysis.modesSubcases.begin(), analysis.modesSubcases.end());
    subcases.insert(subcases.end(), analysis.bucklingSubcases.begin(), analysis.bucklingSubcases.end());
    subcases.insert(subcases.end(), analysis.aeroelasticSubcases.begin(), analysis.aeroelasticSubcases.end());
    std::sort(subcases.begin(), subcases.end());
    subcases.erase(std::unique(subcases.begin(), subcases.end()), subcases.end());
    return subcases;
}

ResponseSubcases DesignProblem::subcasesOfEvaluation(const StaticDesignEvaluation& analysis) const
{
    const std::vector<std::int32_t> analysed = analysedSubcasesOf(analysis);
    const auto analysisOf = [&analysis](std::int32_t subcase)
    {
        if (std::find(analysis.modesSubcases.begin(), analysis.modesSubcases.end(), subcase) !=
            analysis.modesSubcases.end())
            return SubcaseAnalysis::Modes;
        if (std::find(analysis.bucklingSubcases.begin(), analysis.bucklingSubcases.end(), subcase) !=
            analysis.bucklingSubcases.end())
            return SubcaseAnalysis::Buckling;
        if (std::find(analysis.aeroelasticSubcases.begin(), analysis.aeroelasticSubcases.end(), subcase) !=
            analysis.aeroelasticSubcases.end())
            return SubcaseAnalysis::Aeroelastic;
        return SubcaseAnalysis::Statics;
    };
    // Each response in the subcases of its analysis: a monitored stress is not evaluated in a MODES subcase.
    const auto keepItsAnalysis = [&](ResponseSubcases& subcases)
    {
        for (auto& [id, list] : subcases)
            list.erase(std::remove_if(list.begin(), list.end(), [&, id = id](std::int32_t subcase)
                                      { return subcase != 0 && !m_responses.isEvaluatedIn(id, analysisOf(subcase)); }),
                       list.end());
    };
    ResponseSubcases subcases = m_constraints.subcasesOfResponses(analysed, m_responses);
    if (analysis.modesSubcases.empty() && analysis.bucklingSubcases.empty() && analysis.aeroelasticSubcases.empty())
        return m_responses.closeSubcases(std::move(subcases));
    keepItsAnalysis(subcases);
    subcases = m_responses.closeSubcases(std::move(subcases));
    keepItsAnalysis(subcases);
    // A constraint or the objective on a response that its analysis leaves in no subcase: refused, naming it.
    const std::optional<std::int32_t> objective = m_constraints.objective();
    const auto isConstrained = [this](std::int32_t id)
    {
        return std::any_of(m_constraints.parts().constraints.begin(), m_constraints.parts().constraints.end(),
                           [id](const ConstraintDefinition& constraint) { return constraint.responseId == id; });
    };
    for (const auto& [id, list] : subcases)
        if (list.empty() && (isConstrained(id) || (objective && *objective == id)))
            throw base::OptimisationError(base::concatenate(
                {"the design response ", std::to_string(id), " (", m_responses.label(id),
                 ") is constrained or is the objective, but no analysed subcase is of its analysis (a static response "
                 "in STATICS and SAERO subcases, EIGN and FREQ in MODES ones, LAMA in BUCK ones, TRIM and STABDER in "
                 "SAERO ones, an equation where every response it reads is); select a subcase of its analysis "
                 "(SubcaseDesignConstrSelection) or change the AnalysisKind"}));
    return subcases;
}

std::vector<std::int32_t> DesignProblem::analysedSubcases(const solutions::StaticOptions& options) const
{
    return solutions::StaticSolution(*m_snapshot, options).subcasesToSolve();
}

namespace
{

// The g-set shapes of the modes of a subcase of an eigenvalue analysis (null when the subcase has none).
const linalg::DenseMatrix* shapesOf(const StaticDesignEvaluation& analysis, ResponseType type, std::int32_t subcase)
{
    if (type == ResponseType::BucklingFactor)
    {
        if (!analysis.buckling)
            return nullptr;
        for (const solutions::BucklingSolve& solve : analysis.buckling->solves)
            if (solve.subcase == subcase)
                return &solve.eigenvectors;
        return nullptr;
    }
    if (!analysis.modes)
        return nullptr;
    for (const solutions::ModesSubcaseResult& result : analysis.modes->subcases)
        if (result.subcase == subcase)
            return &analysis.modes->solves[result.solve].eigenvectors;
    return nullptr;
}

} // namespace

void DesignProblem::trackModes(const StaticDesignEvaluation& analysis, const ResponseSubcases& subcases,
                               const ModeReferences* references, std::vector<TrackedMode>& tracked,
                               std::vector<base::Diagnostic>& warnings) const
{
    for (const DesignResp1Definition& definition : m_responses.parts().designResp1)
    {
        if (!definition.tracksMode || !isEigenType(definition.type))
            continue;
        const auto found = subcases.find(definition.id);
        if (found == subcases.end())
            continue;
        for (const std::int32_t subcase : found->second)
        {
            const linalg::DenseMatrix* shapes = shapesOf(analysis, definition.type, subcase);
            if (shapes == nullptr || shapes->columns() == 0)
                continue;
            TrackedMode mode;
            mode.responseId = definition.id;
            mode.subcase = subcase;
            mode.requestedMode = definition.attributeA.value_or(1);
            mode.mode = mode.requestedMode;
            const auto reference =
                references != nullptr ? references->find({definition.id, subcase}) : ModeReferences::const_iterator{};
            if (references != nullptr && reference != references->end() && reference->second.size() == shapes->rows())
            {
                const ModeTracking::Match match = ModeTracking::match(reference->second, *shapes);
                mode.mode = static_cast<std::int32_t>(match.mode) + 1;
                mode.mac = match.mac;
                mode.secondMode = static_cast<std::int32_t>(match.secondMode) + 1;
                mode.secondMac = match.secondMac;
                mode.hasReference = true;
                mode.isUncertain = match.isUncertain;
                if (match.isUncertain)
                {
                    base::Diagnostic warning;
                    warning.severity = base::Severity::Warning;
                    warning.code = "mode-tracking-uncertain";
                    warning.card = "DesignResp1";
                    warning.id = definition.id;
                    warning.field = "attributeA";
                    warning.message = base::concatenate(
                        {"the mode tracking of response ", std::to_string(definition.id), " (", definition.label,
                         ") in subcase ", std::to_string(subcase), " is uncertain: mode ", std::to_string(mode.mode),
                         " matches the shape of the previous design cycle with a modal assurance criterion of ",
                         std::to_string(mode.mac), ", mode ", std::to_string(mode.secondMode), " with ",
                         std::to_string(mode.secondMac),
                         " (a repeated root or a crossing in progress); check the modes, or take smaller design "
                         "steps"});
                    warnings.push_back(std::move(warning));
                }
            }
            if (mode.mode >= 1 && static_cast<std::size_t>(mode.mode) <= shapes->columns())
            {
                const auto column = static_cast<std::size_t>(mode.mode - 1);
                const double* first = shapes->data() + column * shapes->leadingDimension();
                mode.shape.assign(first, first + shapes->rows());
            }
            tracked.push_back(std::move(mode));
        }
    }
}

DesignEvaluation DesignProblem::evaluateResponses(StaticDesignEvaluation analysis, const CallableRegistry& callables,
                                                  base::MemoryBudget& budget,
                                                  std::span<const SuppliedField> initialSupplied,
                                                  std::size_t blockItems, const ExternalRegistry& externals,
                                                  const ModeReferences* references) const
{
    std::vector<ResponseValues> responses;
    std::vector<ConstraintValue> constraints;
    std::optional<double> objective;
    std::map<std::int32_t, ItemSelection> selections;
    std::vector<TrackedMode> tracked;
    std::vector<base::Diagnostic> trackingWarnings;
    const std::vector<std::int32_t> analysed = analysedSubcasesOf(analysis);
    if (!m_responses.isEmpty())
    {
        ResultLookup lookup(analysis.results.arrays);
        ResponseContext context;
        context.snapshot = analysis.snapshot.get();
        context.snapshotOwner = analysis.snapshot;
        context.results = &lookup;
        context.model = &m_model;
        context.design = &analysis.point.values();
        context.budget = &budget;
        context.checks = analysis.results.checks;
        context.arrays = &analysis.results.arrays;
        context.sectionLoads = analysis.results.sectionLoads ? &*analysis.results.sectionLoads : nullptr;
        context.cutLoads = analysis.results.cutLoads ? &*analysis.results.cutLoads : nullptr;
        context.zones = analysis.results.zones;
        context.analysedSubcases = analysis.subcases;
        base::DiagnosticSink ignored; // a bad COUPMASS is reported by the analysis itself
        const input::ParamTable* params =
            analysis.snapshot->hasTable(input::TableRole::Params) ? &analysis.snapshot->params() : nullptr;
        context.massFormulation = assembly::MassAssemblyOptions::fromParams(params, std::nullopt, ignored).formulation;
        // FRMASS: the snapshot at the initial design (a second overlay; built only when a response needs it).
        const bool needsInitial = std::any_of(
            m_responses.parts().designResp1.begin(), m_responses.parts().designResp1.end(),
            [](const DesignResp1Definition& definition) { return definition.type == ResponseType::FractionalMass; });
        std::shared_ptr<const input::ModelSnapshot> initial;
        if (needsInitial)
        {
            initial = overlaySnapshot(designPoint(initialValues(), initialSupplied));
            context.initialSnapshot = initial.get();
            context.initialSnapshotOwner = initial;
        }
        const ResponseSubcases subcases = subcasesOfEvaluation(analysis);
        // The eigenvalue tables of the MODES and BUCK subcases (EIGN, FREQ, LAMA), and the modes tracked.
        if (analysis.modes)
            for (const solutions::ModesSubcaseResult& subcase : analysis.modes->subcases)
                context.eigenvalueTables.push_back(&subcase.eigenvalues);
        if (analysis.buckling)
            for (const solutions::BucklingSolve& solve : analysis.buckling->solves)
                context.eigenvalueTables.push_back(&solve.eigenvalues);
        // The trim of the SAERO subcases (TRIM, STABDER).
        context.trim = analysis.trim ? &*analysis.trim : nullptr;
        trackModes(analysis, subcases, references, tracked, trackingWarnings);
        for (const TrackedMode& mode : tracked)
            context.trackedModes[{mode.responseId, mode.subcase}] = mode.mode;
        responses = m_responses.evaluate(subcases, context, callables, blockItems, externals, &selections);
        constraints = m_constraints.evaluate(responses, analysed, m_responses);
        objective = m_constraints.objectiveValue(responses);
    }
    DesignEvaluation evaluation{std::move(analysis),    analysed,  std::move(responses),
                                std::move(constraints), objective, std::move(selections)};
    evaluation.trackedModes = std::move(tracked);
    evaluation.trackingWarnings = std::move(trackingWarnings);
    return evaluation;
}

ResponseSubcases DesignProblem::responseSubcases(const DesignEvaluation& evaluation) const
{
    return subcasesOfEvaluation(evaluation.analysis);
}

StaticSensitivityInputs DesignProblem::sensitivityInputs(
    const DesignEvaluation& evaluation, const ResponseSubcases& subcases, const CallableRegistry& callables,
    const solutions::StaticOptions& analysisOptions, std::size_t blockItems, const ExternalRegistry* externals) const
{
    const StaticDesignEvaluation& analysis = evaluation.analysis;
    base::DiagnosticSink ignored; // a bad COUPMASS is reported by the analysis itself
    const input::ParamTable* params =
        analysis.snapshot->hasTable(input::TableRole::Params) ? &analysis.snapshot->params() : nullptr;
    StaticSensitivityInputs inputs;
    inputs.model = &m_model;
    inputs.targets = &m_targets;
    inputs.responses = &m_responses;
    inputs.base = m_snapshot;
    inputs.overlaid = analysis.snapshot;
    inputs.design = &analysis.point.values();
    inputs.results = &analysis.results;
    inputs.subcases = analysis.subcases;
    // The eigenvalue analyses: the modes and buckling shapes the EIGN, FREQ and LAMA items read, and the
    // static subcases of the preloads among the subcases the sensitivities read (the LAMA items read theirs).
    inputs.modes = analysis.modes ? &*analysis.modes : nullptr;
    inputs.buckling = analysis.buckling ? &*analysis.buckling : nullptr;
    // The trim of the SAERO subcases: the TRIM and STABDER items and the static items of those subcases
    // differentiate the trimmed state (TrimItems on its linearisation).
    inputs.trim = analysis.trim ? &*analysis.trim : nullptr;
    if (!analysis.staticSubcasesRead.empty())
        inputs.subcases = analysis.staticSubcasesRead;
    inputs.values = &evaluation.responses;
    inputs.responseSubcases = &subcases;
    inputs.callables = &callables;
    inputs.externals = externals;
    inputs.blockItems = blockItems;
    inputs.massFormulation = assembly::MassAssemblyOptions::fromParams(params, std::nullopt, ignored).formulation;
    inputs.kernel = analysisOptions.kernel;
    inputs.pool = analysisOptions.pool;
    return inputs;
}

LinearityReport DesignProblem::linearityReport(const StaticDesignEvaluation& analysis, base::MemoryBudget& budget,
                                               base::ThreadPool* pool) const
{
    // PERFORMANCE: the shells' corners, normals, areas and neighbours depend on the positions alone, which a design
    // without shape variables leaves as they are: built once per problem, the thicknesses of each point from its
    // volumes on the pool (LinearityIndicator), the same bits. Measured: the 1 M DOF
    // plate's check 1.5 s -> 0.3 s per point, most of it the volumes.
    if (!m_model.shapes().isEmpty())
    {
        LinearityReport report = LinearityIndicator(*analysis.snapshot, analysis.snapshot, budget, pool)
                                     .report(analysis.results.arrays, analysis.subcases);
        report.geometry.bytes = LinearityGeometry::estimateBytes(*analysis.snapshot);
        return report;
    }
    std::shared_ptr<const base::KeptObject<LinearityGeometry>> geometry;
    bool isReused = false;
    {
        const std::lock_guard<std::mutex> lock(m_keptLinearity->mutex);
        isReused = m_keptLinearity->geometry != nullptr;
        if (!isReused)
            m_keptLinearity->geometry = base::KeptObject<LinearityGeometry>::build(
                [&](base::MemoryBudget& own) { return LinearityGeometry(*m_snapshot, own); });
        geometry = m_keptLinearity->geometry;
    }
    const LinearityIndicator indicator(geometry->lend(budget, "linearity indicator geometry (kept)"),
                                       *analysis.snapshot, analysis.snapshot, budget, pool);
    LinearityReport report = indicator.report(analysis.results.arrays, analysis.subcases);
    report.geometry = {isReused, geometry->bytes()};
    return report;
}

GradientSize DesignProblem::gradientSize(const DesignEvaluation& evaluation) const
{
    std::size_t rows = 0;
    for (const ResponseValues& values : evaluation.responses)
        rows += values.itemCount();
    return GradientStream::size(rows, m_model.independentCount());
}

GradientStreamReport DesignProblem::streamGradients(const DesignEvaluation& evaluation,
                                                    const CallableRegistry& callables, base::MemoryBudget& budget,
                                                    const GradientStreamOptions& options, GradientSink& sink,
                                                    const solutions::StaticOptions& analysisOptions,
                                                    std::size_t blockItems, const ExternalRegistry* externals) const
{
    const ResponseSubcases subcases = subcasesOfEvaluation(evaluation.analysis);
    const StaticSensitivityInputs inputs =
        sensitivityInputs(evaluation, subcases, callables, analysisOptions, blockItems, externals);
    return StaticSensitivity::stream(inputs, options, sink, budget);
}

DesignGradients DesignProblem::gradients(const DesignEvaluation& evaluation, const CallableRegistry& callables,
                                         base::MemoryBudget& budget, const GradientOptions& options,
                                         const solutions::StaticOptions& analysisOptions, std::size_t blockItems,
                                         const ExternalRegistry* externals) const
{
    const ResponseSubcases subcases = subcasesOfEvaluation(evaluation.analysis);
    const StaticSensitivityInputs inputs =
        sensitivityInputs(evaluation, subcases, callables, analysisOptions, blockItems, externals);
    DesignGradients gradients = StaticSensitivity::compute(inputs, options, budget);

    // The normalised constraints (ConstraintNormalisation: dg/dr = ∓1/GNORM) and the objective, from the gradients of
    // their items; none for an item whose gradient was not asked for (its response, or its row of an item mask).
    std::map<std::int32_t, const ResponseGradient*> byId;
    for (const ResponseGradient& gradient : gradients.responses)
        byId[gradient.id] = &gradient;
    const std::size_t n = gradients.variableCount;
    const ConstraintNormalisation normalisation;
    for (const ConstraintValue& constraint : evaluation.constraints)
    {
        std::vector<double> row;
        if (const auto found = byId.find(constraint.responseId);
            found != byId.end() && isItemDifferentiated(options, constraint.responseId, constraint.item))
        {
            const double scale = normalisation.derivative(constraint.side, constraint.bound);
            const std::vector<double>& values = found->second->values;
            row.assign(values.begin() + static_cast<std::ptrdiff_t>(constraint.item * n),
                       values.begin() + static_cast<std::ptrdiff_t>((constraint.item + 1) * n));
            for (double& value : row)
                value *= scale;
        }
        gradients.constraints.push_back(std::move(row));
    }
    if (const std::optional<std::int32_t> objective = m_constraints.objective())
        if (const auto found = byId.find(*objective); found != byId.end() && found->second->itemCount == 1)
            gradients.objective = found->second->values;
    return gradients;
}

} // namespace oos::design
