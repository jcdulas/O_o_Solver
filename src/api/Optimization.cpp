#include "oos/api/Optimization.hpp"
#include "OptimizationDerivatives.hpp"
#include "OptimizationProducts.hpp"

#include "oos/api/Threading.hpp"
#include "oos/base/arrow/TablePackage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/platform/MemoryInfo.hpp"
#include "oos/design/constraints/ConstraintTables.hpp"
#include "oos/design/expressions/ProgramTables.hpp"
#include "oos/design/model/DesignTables.hpp"
#include "oos/design/responses/ResponseTables.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
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

design::DesignProblem makeProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, const OptimizationInput& input)
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
    for (const auto& [name, bytes] : input.tables)
    {
        const auto contract =
            std::find_if(contracts.begin(), contracts.end(), [&](const auto& entry) { return entry.table == name; });
        if (contract == contracts.end())
            throw std::invalid_argument("unknown optimization table: " + name);
        tables.emplace(name, base::ContractReader(base::importTablePackage(bytes), *contract));
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

bool needsFullShapeColumns(const design::DesignEvaluation& evaluation,
                           const OptimizationSettings& settings)
{
    if (evaluation.analysis.point.values().shapes.empty())
        return false;
    if (settings.analyses.hasAeroelastic())
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
    : m_input(std::move(input)),
      m_settings(std::move(settings)),
      m_pool(acquireThreadPool(m_settings.threads)),
      m_problem(makeProblem(std::move(snapshot), m_input))
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
    if (needsFullShapeColumns(m_evaluation, m_owner->settings()))
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

design::GradientStreamReport OptimizationEvaluation::streamGradients(design::GradientSink& sink, std::size_t block)
{
    if (!m_owner->settings().analyses.keepsTrimLinearisation && m_owner->settings().analyses.hasAeroelastic())
        throw std::invalid_argument("native derivatives were disabled; use full-analysis numerical derivatives");
    if (needsFullShapeColumns(m_evaluation, m_owner->settings()))
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
    if (needsFullShapeColumns(m_evaluation, m_owner->settings()))
        return detail::fullShapeJvp(*this, directions, count, block, sink, selection);
    const auto callables = m_owner->callables();
    return design::VectorJacobianProducts::jvp(m_owner->problem(), m_evaluation, directions, count, block, sink,
                                               m_budget, context(callables), &m_linearisation, selection);
}

design::ProductReport OptimizationEvaluation::vjp(const design::ProductWeights& weights, std::size_t block,
                                                  design::ProductSink& sink)
{
    requireFiniteWeights(weights);
    if (needsFullShapeColumns(m_evaluation, m_owner->settings()))
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
