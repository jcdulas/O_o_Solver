#include "oos/api/Optimization.hpp"
#include "OptimizationDerivatives.hpp"
#include "OptimizationProducts.hpp"

#include "oos/api/Threading.hpp"
#include "oos/design/aids/BucklingHessian.hpp"
#include "oos/design/aids/ModalHessian.hpp"
#include "oos/design/aids/TrimHessian.hpp"
#include "oos/base/arrow/TablePackage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/platform/MemoryInfo.hpp"
#include "oos/design/constraints/ConstraintTables.hpp"
#include "oos/design/expressions/ProgramTables.hpp"
#include "oos/design/model/DesignTables.hpp"
#include "oos/design/sensitivity/ShapeElements.hpp"
#include "oos/design/responses/ResponseTables.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <set>
#include <string_view>

namespace oos::api
{
namespace
{
using Readers = std::map<std::string, base::ContractReader>;

const base::ContractReader* table(const Readers& tables, const std::string& name)
{
    const auto found = tables.find(name);
    return found == tables.end() ? nullptr : &found->second;
}

template <typename T>
void append(std::vector<T>& target, const std::vector<T>& source)
{
    target.insert(target.end(), source.begin(), source.end());
}

design::DesignProblem makeProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, const OptimizationInput& input,
                                  std::map<std::string, base::ImportedTable> nativeTables)
{
    Readers tables;
    const std::vector<base::TableContract> contracts = {design::designVariablesContract(),
                                                        design::designLinksContract(),
                                                        design::designRelationsContract(),
                                                        design::designConstantsContract(),
                                                        design::designShapesContract(),
                                                        design::designEquationSymbolsContract(),
                                                        design::designEquationInstructionsContract(),
                                                        design::designResponsesContract(),
                                                        design::designResponseArgumentsContract(),
                                                        design::designConstraintsContract(),
                                                        design::designConstraintSetsContract(),
                                                        design::designSubcasesContract(),
                                                        design::designScreeningContract()};
    const auto contractFor = [&](const std::string& name) -> const base::TableContract&
    {
        const auto contract =
            std::find_if(contracts.begin(), contracts.end(), [&](const auto& entry) { return entry.table == name; });
        if (contract == contracts.end())
            throw std::invalid_argument("unknown optimization table: " + name);
        return *contract;
    };
    for (const auto& [name, bytes] : input.tables)
    {
        const auto& contract = contractFor(name);
        tables.emplace(name, base::ContractReader(base::importTablePackage(bytes), contract));
    }
    for (auto& [name, imported] : nativeTables)
    {
        if (tables.contains(name))
            throw std::invalid_argument("duplicate optimization table: " + name);
        tables.emplace(name, base::ContractReader(std::move(imported), contractFor(name)));
    }
    std::vector<design::ExpressionProgram> equations;
    const auto* symbols = table(tables, "designEquationSymbols");
    const auto* instructions = table(tables, "designEquationInstructions");
    if ((symbols == nullptr) != (instructions == nullptr))
        throw std::invalid_argument("both design equation tables are required");
    if (symbols)
        equations = design::readExpressionPrograms(*symbols, *instructions);
    auto model = design::readDesignTables(table(tables, "designVariablesTable"), table(tables, "designLinksTable"),
                                          table(tables, "designRelationsTable"), table(tables, "designConstantsTable"),
                                          std::move(equations));
    if (const auto* shapes = table(tables, "shapeVariablesTable"))
        model.shapes = design::readShapeTable(*shapes);
    append(model.variables, input.model.variables);
    append(model.links, input.model.links);
    append(model.relations, input.model.relations);
    append(model.equations, input.model.equations);
    append(model.constants, input.model.constants);
    append(model.shapes, input.model.shapes);
    design::ResponseParts responses;
    if (const auto* responseTable = table(tables, "designResponsesTable"))
        responses = design::readResponseTables(*responseTable, table(tables, "designResponseArgumentsTable"));
    append(responses.designResp1, input.responses.designResp1);
    append(responses.equations, input.responses.equations);
    append(responses.externals, input.responses.externals);
    append(responses.zones, input.responses.zones);
    append(responses.zoneSubcases, input.responses.zoneSubcases);
    auto constraints = design::readConstraintTables(
        table(tables, "designConstraintsTable"), table(tables, "designConstraintSetsTable"),
        table(tables, "designSubcasesTable"), table(tables, "designScreeningTable"));
    append(constraints.constraints, input.constraints.constraints);
    append(constraints.unions, input.constraints.unions);
    append(constraints.screening, input.constraints.screening);
    if (input.constraints.selection.objective)
    {
        constraints.selection.objective = input.constraints.selection.objective;
        constraints.selection.objectiveSubcase = input.constraints.selection.objectiveSubcase;
        constraints.selection.isMaximised = input.constraints.selection.isMaximised;
    }
    else if (!constraints.selection.objective)
        if (const auto* responseTable = table(tables, "designResponsesTable"))
            if (const auto objective = design::readTableObjective(*responseTable))
            {
                constraints.selection.objective = objective->responseId;
                constraints.selection.isMaximised = objective->isMaximised;
            }
    return {std::move(snapshot), std::move(model), std::move(responses), std::move(constraints)};
}

void requireFinite(std::span<const double> values, const char* name)
{
    if (std::any_of(values.begin(), values.end(), [](double value) { return !std::isfinite(value); }))
        throw std::invalid_argument(std::string(name) + " must contain finite numbers");
}

void requireFiniteWeights(const design::ProductWeights& weights)
{
    for (const auto& [id, values] : weights.weights)
    {
        (void)id;
        requireFinite(values, "weights");
    }
    for (const auto& [id, values] : weights.sparseWeights)
    {
        (void)id;
        requireFinite(values.values, "weights");
    }
}

std::uint64_t budgetBytes(std::uint64_t requested)
{
    const auto available = base::MemoryInfo::query().availableBytes;
    return requested ? requested : (available ? available : base::MemoryBudget::UNLIMITED);
}

bool needsFullShapeColumns(const design::DesignEvaluation& evaluation, const OptimizationProblem& owner,
                           std::optional<std::int32_t> coordinateResponse = {}, const design::DesignProblem* coordinateBasis = nullptr)
{
    if (!coordinateResponse && evaluation.analysis.point.values().shapes.empty())
        return false;
    if (owner.settings().analyses.hasAeroelastic())
    {
        const auto& a = evaluation.analysis;
        if (!a.trim || !a.trim->linearisation || !a.trim->linearisation->supportsTransferRates(false) ||
            owner.settings().analyses.modes || owner.settings().analyses.statics)
            return true;
        if (!a.trim->linearisation->supportsTransferRates(true))
        {
            const auto& basis = coordinateBasis ? *coordinateBasis : owner.problem();
            const design::ShapeElements reached(basis.model(), basis.targets(), a.trim->linearisation->elementSet());
            for (std::size_t b = 0; b < basis.model().shapes().bases().size(); ++b)
                if (!reached.of(b).empty())
                    return true;
        }
        for (const auto& response : evaluation.responses)
        {
            if (coordinateResponse && response.id != *coordinateResponse)
                continue;
            const auto* r = owner.problem().responses().findDesignResp1(response.id);
            if (!r || !(design::isTrimLoadType(r->definition().type) ||
                        r->definition().type == design::ResponseType::TrimVariable ||
                        r->definition().type == design::ResponseType::BucklingFactor))
                return true;
        }
        if (a.buckling && a.buckling->path != solutions::AnalysisPath::Full)
            return true;
        if (a.buckling)
            for (const auto& solve : a.buckling->solves)
                if (std::find(a.aeroelasticSubcases.begin(), a.aeroelasticSubcases.end(), solve.preloadSubcase) ==
                    a.aeroelasticSubcases.end())
                    return true;
        return false;
    }
    // A moving constraint changes the expansion and normalisation of a buckling mode.
    // Re-evaluate geometric columns for retained buckling field responses.
    for (const auto& response : evaluation.responses)
        for (const auto& item : response.items)
            if (item.mode > 0 &&
                std::find(evaluation.analysis.bucklingSubcases.begin(), evaluation.analysis.bucklingSubcases.end(),
                          item.subcase) != evaluation.analysis.bucklingSubcases.end())
                return true;
    const auto& results = evaluation.analysis.results;
    if (!results.reliefs.empty())
        return true;
    for (const auto& system : results.condensedSystems)
        if (system != nullptr && system->relief() != nullptr)
            return true;
    return std::any_of(results.streamed.begin(), results.streamed.end(),
                       [](const auto& set) { return set.isRelieved; });
}

} // namespace

OptimizationProblem::OptimizationProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, OptimizationInput input,
                                         OptimizationSettings settings)
    : OptimizationProblem(std::move(snapshot), std::move(input), std::move(settings), {})
{
}

OptimizationProblem::OptimizationProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, OptimizationInput input,
                                         OptimizationSettings settings, std::map<std::string, base::ImportedTable> tables)
    : m_input(std::move(input)),
      m_settings(std::move(settings)),
      m_pool(acquireThreadPool(m_settings.threads)),
      m_problem(makeProblem(std::move(snapshot), m_input, std::move(tables)))
{
    if (m_settings.blockItems == 0 || m_settings.maximumOutputBytes == 0)
        throw std::invalid_argument("blockItems and maximumOutputBytes must be positive");
    m_settings.analysisOptions.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    m_settings.analysisOptions.pool = m_pool.get();
    const auto links = aero::trimLinks(m_problem.snapshot());
    m_settings.analyses.links.insert(m_settings.analyses.links.begin(), links.begin(), links.end());
    m_problem.setDeckOutputs(m_settings.deckOutputs);
    m_input.tables.clear(); // the checked design parts own their data
}

design::CallableRegistry OptimizationProblem::callables() const
{
    design::CallableRegistry result;
    for (const auto& [name, callable] : m_input.callables)
        result.emplace(name, callable.get());
    return result;
}

std::vector<design::SuppliedField> OptimizationProblem::supplied(std::span<const double> x) const
{
    return m_input.suppliedAt ? m_input.suppliedAt(x) : std::vector<design::SuppliedField>{};
}

std::shared_ptr<OptimizationEvaluation> OptimizationProblem::evaluate(std::vector<double> x,
                                                                      const design::ModeReferences& references)
{
    requireFinite(x, "x");
    return std::make_shared<OptimizationEvaluation>(shared_from_this(), std::move(x), references);
}

OptimizationEvaluation::OptimizationEvaluation(std::shared_ptr<OptimizationProblem> owner, std::vector<double> x,
                                               const design::ModeReferences& references)
    : m_owner(std::move(owner)),
      m_x(std::move(x)),
      m_budget(budgetBytes(m_owner->settings().memoryBytes)),
      m_evaluation(
          [&]
          {
              const auto& settings = m_owner->settings();
              auto analysis = m_owner->problem().evaluateAnalyses(m_x, settings.analyses, settings.analysisOptions,
                                                                  m_budget, m_owner->supplied(m_x), true);
              const auto initial = m_owner->supplied(m_owner->problem().initialValues());
              return m_owner->problem().evaluateResponses(std::move(analysis), m_owner->callables(), m_budget, initial,
                                                          settings.blockItems, {},
                                                          references.empty() ? nullptr : &references);
          }())
{
}

void OptimizationEvaluation::requireOutput(std::size_t rows, std::size_t columns) const
{
    const auto maximum = m_owner->settings().maximumOutputBytes / sizeof(double);
    if (columns != 0 && rows > maximum / columns)
        throw std::invalid_argument("derivative output exceeds maximumOutputBytes; use streamed blocks or products");
}

design::ProductContext OptimizationEvaluation::context(const design::CallableRegistry& callables) const
{
    if (!m_owner->settings().analyses.keepsTrimLinearisation && m_owner->settings().analyses.hasAeroelastic())
        throw std::invalid_argument("native derivatives were disabled; use full-analysis numerical derivatives");
    design::ProductContext result;
    result.callables = &callables;
    result.analysisOptions = m_owner->settings().analysisOptions;
    result.blockItems = m_owner->settings().blockItems;
    const auto owner = m_owner;
    result.suppliedAt = [owner](std::span<const double> x) { return owner->supplied(x); };
    return result;
}

NodalCoordinateGradient OptimizationEvaluation::nodalCoordinateGradient(std::int32_t responseId,
                                                                        std::span<const std::int32_t> nodes,
                                                                        std::size_t item)
{
    const auto& problem = m_owner->problem();
    const auto& snapshot = m_evaluation.analysis.snapshot;
    const auto* primary = problem.responses().findDesignResp1(responseId);
    const auto row = std::find_if(m_evaluation.responses.begin(), m_evaluation.responses.end(),
                                  [&](const auto& r) { return r.id == responseId; });
    if (!primary || row == m_evaluation.responses.end() || item >= row->itemCount() || !row->isOk(item))
        throw std::invalid_argument("nodalCoordinateGradient requires an evaluated scalar item of a primary response");
    if (primary->definition().type == design::ResponseType::FractionalMass)
        throw std::invalid_argument("fractional mass requires the design's original reference mass");
    const auto ids = snapshot->nodes().ids();
    if (nodes.empty())
        nodes = ids;
    requireOutput(nodes.size(), 3);
    if (nodes.size() > static_cast<std::size_t>(INT32_MAX) / 3)
        throw std::invalid_argument("too many coordinate variables");
    OptimizationInput input;
    std::set<std::int32_t> seen;
    for (auto id : nodes)
    {
        if (!std::binary_search(ids.begin(), ids.end(), id) || !seen.insert(id).second)
            throw std::invalid_argument("unknown or duplicate node in nodalCoordinateGradient: " + std::to_string(id));
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto k = static_cast<std::int32_t>(input.model.variables.size() + 1);
            input.model.variables.push_back({k, "coordinate", 0., -1., 1., {}, {}});
            design::ShapeTerm term{k, id, 0, 1., {}};
            term.direction[axis] = 1.;
            input.model.shapes.push_back(term);
        }
    }
    NodalCoordinateGradient result;
    result.nodeIds.assign(nodes.begin(), nodes.end());
    if (nodes.empty())
        return result;
    auto definition = primary->definition();
    const auto label = row->items[item];
    if (label.subcase)
        definition.subcases = {label.subcase};
    const auto& settings = m_owner->settings();
    design::GradientOptions options;
    options.responseIds = {responseId};
    std::size_t selected = 0;
    design::DesignGradients gradient;
    const design::DesignProblem coordinateBasis(snapshot, input.model);
    if (needsFullShapeColumns(m_evaluation, *m_owner, responseId, &coordinateBasis))
    {
        input.responses.designResp1.push_back(std::move(definition));
        auto probeSettings = settings;
        // OptimizationProblem imports the deck links on construction. Preserve the caller's additional links.
        const auto imported = aero::trimLinks(*snapshot).size();
        probeSettings.analyses.links.erase(probeSettings.analyses.links.begin(),
            probeSettings.analyses.links.begin() + static_cast<std::ptrdiff_t>(imported));
        auto coordinates = std::make_shared<OptimizationProblem>(snapshot, std::move(input), std::move(probeSettings));
        auto x = coordinates->problem().initialValues();
        design::ModeReferences refs;
        for (const auto& mode : m_evaluation.trackedModes)
            if (mode.responseId == responseId)
                refs[{responseId, mode.subcase}] = mode.shape;
        auto base = coordinates->evaluate(x, refs);
        const auto& items = base->evaluation().responses.at(0).items;
        const auto found = std::find(items.begin(), items.end(), label);
        if (found == items.end())
            throw std::invalid_argument("the response item changed at the coordinate reference point");
        selected = static_cast<std::size_t>(found - items.begin());
        options.itemMasks.push_back({responseId, std::vector<std::uint8_t>(items.size())});
        options.itemMasks.back().isRetained[selected] = 1;
        gradient = base->gradients(options);
    }
    else
    {
        // Bind the XYZ basis to the current geometry; reuse the factors, states and selected modal roots of this point.
        const auto basis = coordinateBasis.designPoint(coordinateBasis.initialValues());
        const auto callables = m_owner->callables();
        const auto subcases = problem.responseSubcases(m_evaluation);
        auto in =
            problem.sensitivityInputs(m_evaluation, subcases, callables, settings.analysisOptions, settings.blockItems);
        in.base = snapshot;
        in.model = &coordinateBasis.model();
        in.targets = &coordinateBasis.targets();
        in.design = &basis.values();
        // Only one item is differentiated, even for a response that recovers all nodes.
        const std::vector<design::ResponseValues> values{{responseId, row->label, {label}, {row->values[item]}, {}}};
        in.values = &values;
        gradient = design::StaticSensitivity::compute(in, options, m_budget);
    }
    if (gradient.report.repeatedRoots)
        throw std::invalid_argument("a nodal coordinate gradient of an eigenvalue requires an isolated mode");
    const auto& values = gradient.responses.at(0).values;
    const auto first = values.begin() + static_cast<std::ptrdiff_t>(selected * gradient.variableCount);
    result.values.assign(first, first + static_cast<std::ptrdiff_t>(gradient.variableCount));
    result.report = std::move(gradient.report);
    return result;
}

design::DesignGradients OptimizationEvaluation::gradients(const design::GradientOptions& options)
{
    if (!m_owner->settings().analyses.keepsTrimLinearisation && m_owner->settings().analyses.hasAeroelastic())
        throw std::invalid_argument("native derivatives were disabled; use method finiteDifference");
    const auto isRequested = [&](std::int32_t id)
    {
        return options.responseIds.empty() ||
               std::find(options.responseIds.begin(), options.responseIds.end(), id) != options.responseIds.end();
    };
    std::size_t rows = 0;
    for (const auto& response : m_evaluation.responses)
        if (isRequested(response.id))
            rows += response.itemCount();
    for (const auto& constraint : m_evaluation.constraints)
        rows += isRequested(constraint.responseId) ? 1 : 0;
    if (const auto objective = m_owner->problem().constraints().objective(); objective && isRequested(*objective))
        ++rows;
    requireOutput(rows, m_x.size());
    design::GradientOptions nativeOptions = options;
    std::vector<std::size_t> shapeColumns;
    if (!options.isVariableWanted.empty() && options.isVariableWanted.size() != m_x.size())
        throw std::invalid_argument("isVariableWanted must have one entry per independent variable");
    if (needsFullShapeColumns(m_evaluation, *m_owner))
    {
        for (const auto& shape : m_evaluation.analysis.point.values().shapes)
            for (const auto& [column, rate] : shape.gradient)
                if (rate != 0.0 && column < m_x.size() &&
                    (options.isVariableWanted.empty() || options.isVariableWanted[column] != 0))
                    shapeColumns.push_back(column);
        std::sort(shapeColumns.begin(), shapeColumns.end());
        shapeColumns.erase(std::unique(shapeColumns.begin(), shapeColumns.end()), shapeColumns.end());
        if (!shapeColumns.empty())
        {
            if (nativeOptions.isVariableWanted.empty())
                nativeOptions.isVariableWanted.assign(m_x.size(), 1);
            for (const auto column : shapeColumns)
                nativeOptions.isVariableWanted[column] = 0;
        }
    }
    auto result = m_owner->problem().gradients(m_evaluation, m_owner->callables(), m_budget, nativeOptions,
                                                m_owner->settings().analysisOptions, m_owner->settings().blockItems);
    if (!shapeColumns.empty())
        detail::completeShapeGradients(*this, options, result, shapeColumns);
    return result;
}

OptimizerValues OptimizationEvaluation::optimizerValues() const
{
    OptimizerValues result;
    result.objective = m_evaluation.objective;
    result.constraints.reserve(m_evaluation.constraints.size());
    for (const auto& row : m_evaluation.constraints)
        result.constraints.push_back(row.normalised);
    for (const auto& response : m_evaluation.responses)
        if (std::find(response.statuses.begin(), response.statuses.end(), design::ItemStatus::Failed) !=
            response.statuses.end())
        {
            result.partial = true;
            break;
        }
    result.uncertainMode = std::any_of(m_evaluation.trackedModes.begin(), m_evaluation.trackedModes.end(),
                                       [](const auto& mode) { return mode.isUncertain; });
    return result;
}

design::GradientStreamReport OptimizationEvaluation::streamGradients(design::GradientSink& sink, std::size_t block)
{
    if (!m_owner->settings().analyses.keepsTrimLinearisation && m_owner->settings().analyses.hasAeroelastic())
        throw std::invalid_argument("native derivatives were disabled; use full-analysis numerical derivatives");
    if (needsFullShapeColumns(m_evaluation, *m_owner))
        return detail::fullShapeStream(*this, m_budget, sink, block);
    design::GradientStreamOptions options;
    options.block = block;
    return m_owner->problem().streamGradients(m_evaluation, m_owner->callables(), m_budget,
                                              options, sink,
                                              m_owner->settings().analysisOptions, m_owner->settings().blockItems);
}

design::ProductReport OptimizationEvaluation::jvp(std::span<const double> directions, std::size_t count,
                                                  std::size_t block, design::ProductSink& sink,
                                                  const design::ProductSelection& selection)
{
    requireFinite(directions, "directions");
    if (needsFullShapeColumns(m_evaluation, *m_owner))
        return detail::fullShapeJvp(*this, directions, count, block, sink, selection);
    const auto callables = m_owner->callables();
    return design::VectorJacobianProducts::jvp(m_owner->problem(), m_evaluation, directions, count, block, sink,
                                               m_budget, context(callables), &m_linearisation, selection);
}

design::ProductReport OptimizationEvaluation::vjp(const design::ProductWeights& weights, std::size_t block,
                                                  design::ProductSink& sink)
{
    requireFiniteWeights(weights);
    if (needsFullShapeColumns(m_evaluation, *m_owner))
        return detail::fullShapeVjp(*this, weights, block, sink);
    const auto callables = m_owner->callables();
    return design::VectorJacobianProducts::vjp(m_owner->problem(), m_evaluation, weights, block, sink, m_budget,
                                               context(callables), &m_linearisation);
}

design::ProductReport OptimizationEvaluation::hessianVector(const design::ProductWeights& weights,
                                                            std::span<const double> directions, std::size_t count,
                                                            std::vector<double>& values)
{
    requireFiniteWeights(weights);
    requireOutput(m_x.size(), count);
    requireFinite(directions, "directions");
    const auto& analyses = m_owner->settings().analyses;
    if (m_evaluation.analysis.buckling)
        if (auto report = design::bucklingHessianVector(m_owner->problem(), m_evaluation, weights, directions, count,
                                                        values, m_budget, m_owner->settings().analysisOptions.kernel,
                                                        m_owner->settings().analysisOptions.pool))
            return *report;
    if (analyses.hasAeroelastic())
        if (auto report = design::trimHessianVector(m_owner->problem(), m_evaluation, weights, directions, count,
                                                    values, m_budget, m_owner->settings().analysisOptions.pool))
            return *report;
    if (analyses.hasEigenvalues())
        if (auto report = design::modalHessianVector(m_owner->problem(), m_evaluation, weights, directions, count,
                                                     values, m_budget, m_owner->settings().analysisOptions.kernel,
                                                     m_owner->settings().analysisOptions.pool))
            return *report;
    if (analyses.hasEigenvalues() || analyses.hasAeroelastic() ||
        !m_evaluation.analysis.point.values().shapes.empty())
        return detail::nativeGradientHessian(*this, weights, directions, count, values);
    const auto callables = m_owner->callables();
    try
    {
        return design::VectorJacobianProducts::hessianVector(m_owner->problem(), m_evaluation, weights, directions,
                                                             values, m_budget, context(callables), &m_linearisation,
                                                             count);
    }
    catch (const base::OptimisationError& error)
    {
        // The fast path names its unsupported second-order item with this prefix. Other failures remain errors.
        if (!std::string_view(error.what()).starts_with("hessian_vector:"))
            throw;
        return detail::nativeGradientHessian(*this, weights, directions, count, values);
    }
}

std::vector<double> OptimizationEvaluation::hessian(const design::ProductWeights& weights)
{
    const auto n = m_x.size();
    requireOutput(n, n);
    auto reservation = m_budget.reserve(n * n * sizeof(double), "dense reduced Hessian");
    std::vector<double> matrix(n * n), direction(n), product;
    for (std::size_t column = 0; column < n; ++column)
    {
        direction[column] = 1.0;
        (void)hessianVector(weights, direction, 1, product);
        for (std::size_t row = 0; row < n; ++row)
            matrix[row * n + column] = product[row];
        direction[column] = 0.0;
    }
    return matrix;
}

design::DesignBlocks OptimizationEvaluation::blocks(const design::ProductWeights* weights)
{
    const auto callables = m_owner->callables();
    return design::VectorJacobianProducts::blocks(m_owner->problem(), m_evaluation, m_budget, context(callables),
                                                  &m_linearisation, weights);
}
} // namespace oos::api
