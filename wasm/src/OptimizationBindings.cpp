// Conversion only: the design and derivative computations live in oos::api and oos::design.
#include "Model.hpp"
#include "oos/api/Optimization.hpp"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace oos::wasm
{
emscripten::val optimizationResults(api::OptimizationEvaluation& point);
}

namespace
{
using emscripten::val;
using namespace oos;

template <typename Work>
auto guarded(Work work) -> decltype(work())
{
    try
    {
        return work();
    }
    catch (const std::exception& error)
    {
        val::global("Error").new_(std::string(error.what())).throw_();
    }
    __builtin_unreachable();
}
bool present(const val& value)
{
    return !value.isUndefined() && !value.isNull();
}
val get(const val& object, const char* key)
{
    return present(object) ? object[key] : val::undefined();
}
template <typename T>
T option(const val& object, const char* key, T fallback)
{
    const auto value = get(object, key);
    return present(value) ? value.as<T>() : fallback;
}
std::size_t countOf(const val& value, std::size_t fallback = 0)
{
    if (!present(value))
        return fallback;
    const auto number = value.as<double>();
    if (!std::isfinite(number) || number < 0 || number > 9007199254740991.0 || std::floor(number) != number)
        throw std::invalid_argument("expected a non-negative safe integer");
    return static_cast<std::size_t>(number);
}
std::int32_t idOf(const val& value)
{
    const double number = value.as<double>();
    if (!std::isfinite(number) || number < std::numeric_limits<std::int32_t>::min() ||
        number > std::numeric_limits<std::int32_t>::max() || std::floor(number) != number)
        throw std::invalid_argument("expected a 32-bit integer ID");
    return static_cast<std::int32_t>(number);
}
template <typename T>
std::vector<T> numbers(const val& value)
{
    if (!present(value))
        return {};
    if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::size_t>)
    {
        std::vector<T> result;
        for (std::size_t i = 0, n = countOf(value["length"]); i < n; ++i)
            if constexpr (std::is_same_v<T, std::int32_t>)
                result.push_back(idOf(value[i]));
            else
                result.push_back(countOf(value[i]));
        return result;
    }
    return emscripten::convertJSArrayToNumberVector<T>(value);
}
template <typename T>
val array(const std::vector<T>& values, const char* kind = "Float64Array")
{
    // Copy: callers may retain this after the evaluation is deleted or its memory grows.
    const auto view = val(emscripten::typed_memory_view(values.size(), values.data()));
    auto result = val::global(kind).new_(static_cast<double>(values.size()));
    result.call<void>("set", view);
    return result;
}
template <typename Work>
void each(const val& values, Work work)
{
    if (!present(values))
        return;
    for (std::size_t i = 0, n = countOf(values["length"]); i < n; ++i)
        work(values[i]);
}
template <typename Work>
void entries(const val& values, Work work)
{
    if (!present(values))
        return;
    each(val::global("Object").call<val>("keys", values),
         [&](const val& key)
         {
             const auto name = key.as<std::string>();
             work(name, values[name]);
         });
}
template <typename T>
std::optional<T> optional(const val& object, const char* key)
{
    const auto value = get(object, key);
    if (!present(value))
        return std::nullopt;
    if constexpr (std::is_same_v<T, std::int32_t>)
        return idOf(value);
    return value.as<T>();
}

val argumentsObject(std::span<const design::BatchArgument> arguments)
{
    auto result = val::array();
    for (const auto& argument : arguments)
    {
        auto object = val::object();
        object.set("name", argument.name);
        object.set("components", double(argument.components));
        object.set("isResponse", argument.isResponse);
        object.set("values", array(argument.values));
        result.call<void>("push", object);
    }
    return result;
}

class JsCallable final : public design::BatchCallable
{
public:
    explicit JsCallable(val callbacks) : m_callbacks(std::move(callbacks)) {}
    design::BatchResult evaluate(std::span<const design::BatchArgument> arguments, std::size_t items) override
    {
        return invoke(arguments, items, false);
    }
    design::BatchResult evaluateWithPartials(std::span<const design::BatchArgument> arguments,
                                             std::size_t items) override
    {
        return invoke(arguments, items, true);
    }
    bool hasGradient() const noexcept override { return present(get(m_callbacks, "evaluateWithPartials")); }
    bool hasHessianVector() const noexcept override { return present(get(m_callbacks, "hessianVector")); }
    std::vector<std::vector<double>> hessianVector(std::span<const design::BatchArgument> arguments, std::size_t items,
                                                   std::size_t outputs, std::span<const double> weights,
                                                   std::span<const design::BatchArgument> directions) override
    {
        auto request = val::object();
        request.set("arguments", argumentsObject(arguments));
        request.set("items", double(items));
        request.set("outputs", double(outputs));
        request.set("weights", array(std::vector<double>(weights.begin(), weights.end())));
        request.set("directions", argumentsObject(directions));
        std::vector<std::vector<double>> result;
        each(m_callbacks.call<val>("hessianVector", request),
             [&](const val& row) { result.push_back(numbers<double>(row)); });
        return result;
    }

private:
    design::BatchResult invoke(std::span<const design::BatchArgument> arguments, std::size_t items, bool partials)
    {
        auto request = val::object();
        request.set("arguments", argumentsObject(arguments));
        request.set("items", double(items));
        const char* method = partials ? "evaluateWithPartials" : "evaluate";
        if (!present(get(m_callbacks, method)))
            throw std::invalid_argument(std::string("missing callback: ") + method);
        const auto answer = m_callbacks.call<val>(method, request);
        design::BatchResult result;
        result.outputs = countOf(get(answer, "outputs"), 1);
        result.values = numbers<double>(get(answer, "values"));
        each(get(answer, "partials"), [&](const val& row) { result.partials.push_back(numbers<double>(row)); });
        return result;
    }
    val m_callbacks;
};

std::shared_ptr<api::OptimizationProblem> createOptimization(const std::shared_ptr<wasm::Model>& model,
                                                             const val& declarations, const val& settings)
{
    return guarded(
        [&]
        {
            api::OptimizationInput input;
            entries(get(declarations, "tables"), [&](const auto& name, const val& bytes)
                    { input.tables.emplace(name, numbers<std::uint8_t>(bytes)); });
            each(get(declarations, "variables"),
                 [&](const val& value)
                 {
                     design::DesignVariable variable;
                     variable.id = idOf(value["id"]);
                     variable.label = option<std::string>(value, "label", "");
                     variable.initialValue = value["initialValue"].as<double>();
                     variable.lower = value["lower"].as<double>();
                     variable.upper = value["upper"].as<double>();
                     variable.moveLimit = optional<double>(value, "moveLimit");
                     input.model.variables.push_back(std::move(variable));
                 });
            each(get(declarations, "links"),
                 [&](const val& value)
                 {
                     design::DesignLink link;
                     link.dependentId = idOf(value["dependentId"]);
                     link.constant = option<double>(value, "constant", 0);
                     link.multiplier = option<double>(value, "multiplier", 1);
                     each(get(value, "terms"), [&](const val& term)
                          { link.terms.push_back({idOf(term["variableId"]), term["coefficient"].as<double>()}); });
                     input.model.links.push_back(std::move(link));
                 });
            each(get(declarations, "relations"),
                 [&](const val& value)
                 {
                     design::DesignRelation relation;
                     relation.id = idOf(value["id"]);
                     const auto kind = design::findTargetKind(value["targetKind"].as<std::string>());
                     if (!kind)
                         throw std::invalid_argument("unknown relation targetKind");
                     relation.targetKind = *kind;
                     relation.targetCard = value["targetCard"].as<std::string>();
                     relation.targetId = idOf(value["targetId"]);
                     relation.field = value["field"].as<std::string>();
                     relation.fieldItem = present(get(value, "fieldItem")) ? idOf(value["fieldItem"]) : -1;
                     relation.card = option<std::string>(value, "card", "DesignRelation");
                     relation.constant = option<double>(value, "constant", 0);
                     relation.lowerBound = optional<double>(value, "lower");
                     relation.upperBound = optional<double>(value, "upper");
                     const auto form = option<std::string>(value, "form", "linear");
                     if (form == "equation")
                     {
                         relation.form = design::RelationForm::Equation;
                         relation.equationId = idOf(value["equationId"]);
                     }
                     else if (form == "supplied")
                         relation.form = design::RelationForm::Supplied;
                     else if (form != "linear")
                         throw std::invalid_argument("unknown relation form");
                     each(get(value, "terms"), [&](const val& term)
                          { relation.terms.push_back({idOf(term["variableId"]), term["coefficient"].as<double>()}); });
                     each(get(value, "arguments"),
                          [&](const val& argument)
                          {
                              relation.arguments.push_back(
                                  {present(get(argument, "variableId")) ? idOf(argument["variableId"]) : 0,
                                   option<std::string>(argument, "constantLabel", "")});
                          });
                     input.model.relations.push_back(std::move(relation));
                 });

            entries(get(declarations, "constants"), [&](const auto& name, const val& value)
                    { input.model.constants.emplace_back(name, value.as<double>()); });
            each(get(declarations, "shapes"),
                 [&](const val& value)
                 {
                     design::ShapeTerm shape;
                     shape.variableId = idOf(value["variableId"]);
                     shape.gridId = idOf(value["gridId"]);
                     shape.coefficient = option<double>(value, "coefficient", 1);
                     const auto direction = numbers<double>(value["direction"]);
                     if (direction.size() != 3)
                         throw std::invalid_argument("a shape direction needs three components");
                     std::copy(direction.begin(), direction.end(), shape.direction.begin());
                     input.model.shapes.push_back(shape);
                 });
            each(get(declarations, "responses"),
                 [&](const val& value)
                 {
                     design::DesignResp1Definition response;
                     response.id = idOf(value["id"]);
                     response.label = option<std::string>(value, "label", "");
                     const auto type = design::findResponseType(value["type"].as<std::string>());
                     if (!type)
                         throw std::invalid_argument("unknown response type");
                     response.type = *type;
                     response.propertyType = option<std::string>(value, "propertyType", "");
                     response.attributeA = optional<std::int32_t>(value, "attributeA");
                     response.attributeB = optional<double>(value, "attributeB");
                     response.componentName = option<std::string>(value, "componentName", "");
                     response.region = optional<std::int32_t>(value, "region");
                     response.targets = numbers<std::int32_t>(get(value, "targets"));
                     response.plies = numbers<std::int32_t>(get(value, "plies"));
                     response.subcases = numbers<std::int32_t>(get(value, "subcases"));
                     response.tracksMode = option<bool>(value, "tracksMode", false);
                     input.responses.designResp1.push_back(std::move(response));
                 });
            each(get(declarations, "equationResponses"),
                 [&](const val& value)
                 {
                     design::EquationResponseDefinition response;
                     response.id = idOf(value["id"]);
                     response.label = option<std::string>(value, "label", "");
                     response.callable = option<std::string>(value, "callable", "");
                     response.form =
                         response.callable.empty() ? design::EquationForm::Expression : design::EquationForm::Callable;
                     response.equationId = present(get(value, "equationId")) ? idOf(value["equationId"]) : 0;
                     response.byLabel = option<bool>(value, "byLabel", false);
                     response.subcases = numbers<std::int32_t>(get(value, "subcases"));
                     each(get(value, "arguments"),
                          [&](const val& argument)
                          {
                              design::ResponseArgument a;
                              const auto kind = design::findArgumentKind(argument["kind"].as<std::string>());
                              if (!kind)
                                  throw std::invalid_argument("unknown equation argument kind");
                              a.kind = *kind;
                              a.id = present(get(argument, "id")) ? idOf(argument["id"]) : 0;
                              a.name = option<std::string>(argument, "name", "");
                              a.label = option<std::string>(argument, "label", "");
                              a.component = present(get(argument, "component")) ? idOf(argument["component"]) : 0;
                              response.arguments.push_back(std::move(a));
                          });
                     input.responses.equations.push_back(std::move(response));
                 });
            each(get(declarations, "constraints"),
                 [&](const val& value)
                 {
                     design::ConstraintDefinition c;
                     c.responseId = idOf(value["responseId"]);
                     c.lower = optional<double>(value, "lower");
                     c.upper = optional<double>(value, "upper");
                     c.subcases = numbers<std::int32_t>(get(value, "subcases"));
                     input.constraints.constraints.push_back(std::move(c));
                 });
            if (const auto objective = get(declarations, "objective"); present(objective))
            {
                input.constraints.selection.objective = idOf(objective["responseId"]);
                const auto sense = option<std::string>(objective, "sense", "min");
                if (sense != "min" && sense != "max")
                    throw std::invalid_argument("objective sense must be min or max");
                input.constraints.selection.isMaximised = sense == "max";
                input.constraints.selection.objectiveSubcase = optional<std::int32_t>(objective, "subcase");
            }
            entries(get(declarations, "callables"), [&](const auto& name, const val& callbacks)
                    { input.callables.emplace(name, std::make_shared<JsCallable>(callbacks)); });
            if (const auto callback = get(declarations, "suppliedAt"); present(callback))
                input.suppliedAt = [callback](std::span<const double> x)
                {
                    std::vector<design::SuppliedField> result;
                    each(callback(array(std::vector<double>(x.begin(), x.end()))),
                         [&](const val& value)
                         {
                             result.push_back({countOf(value["relationIndex"]), value["value"].as<double>(),
                                               numbers<double>(value["gradient"])});
                         });
                    return result;
                };
            api::OptimizationSettings chosen;
            chosen.memoryBytes = countOf(get(settings, "memoryBytes"));
            chosen.maximumOutputBytes = countOf(get(settings, "maximumOutputBytes"), chosen.maximumOutputBytes);
            chosen.blockItems = countOf(get(settings, "blockItems"), chosen.blockItems);
            chosen.deckOutputs = option<bool>(settings, "deckOutputs", false);
            if (const auto analyses = get(settings, "analyses"); present(analyses))
            {
                auto selected = [&](const char* name) -> std::optional<std::vector<std::int32_t>>
                {
                    const auto values = get(analyses, name);
                    return present(values) ? std::optional(numbers<std::int32_t>(values)) : std::nullopt;
                };
                chosen.analyses.statics = selected("statics");
                chosen.analyses.modes = selected("modes");
                chosen.analyses.buckling = selected("buckling");
                chosen.analyses.aeroelastic = selected("aeroelastic");
            }
            return std::make_shared<api::OptimizationProblem>(model->snapshot, std::move(input), std::move(chosen));
        });
}

val metadata(api::OptimizationProblem& owner)
{
    return guarded(
        [&]
        {
            auto result = val::object();
            auto variables = val::array();
            for (const auto& variable : owner.problem().model().variables().all())
            {
                auto v = val::object();
                v.set("id", variable.id);
                v.set("label", variable.label);
                v.set("initialValue", variable.initialValue);
                v.set("lower", variable.lower);
                v.set("upper", variable.upper);
                v.set("moveLimit", variable.moveLimit ? val(*variable.moveLimit) : val::null());
                variables.call<void>("push", v);
            }
            result.set("variables", variables);
            result.set("independentIndices",
                       array(std::vector<double>(owner.problem().model().independentIndices().begin(),
                                                 owner.problem().model().independentIndices().end())));
            result.set("initialValues", array(owner.problem().initialValues()));
            result.set("maximumOutputBytes", double(owner.settings().maximumOutputBytes));
            auto relations = val::array();
            for (const auto& relation : owner.problem().model().relations().all())
            {
                auto r = val::object();
                r.set("id", relation.id);
                r.set("targetCard", relation.targetCard);
                r.set("targetId", relation.targetId);
                r.set("field", relation.field);
                r.set("fieldItem", relation.fieldItem);
                r.set("lower", relation.lowerBound ? val(*relation.lowerBound) : val::null());
                r.set("upper", relation.upperBound ? val(*relation.upperBound) : val::null());
                relations.call<void>("push", r);
            }
            result.set("relations", relations);
            const auto& selection = owner.problem().constraints().parts().selection;
            result.set("objectiveResponseId", selection.objective ? val(*selection.objective) : val::null());
            result.set("objectiveSense", std::string(selection.isMaximised ? "max" : "min"));
            result.set("objectiveSubcase", selection.objectiveSubcase.value_or(0));
            return result;
        });
}

val sparseGradient(const std::vector<std::pair<std::size_t, double>>& gradient)
{
    std::vector<double> columns, values;
    for (const auto& [column, value] : gradient)
    {
        columns.push_back(double(column));
        values.push_back(value);
    }
    auto result = val::object();
    result.set("columns", array(columns));
    result.set("values", array(values));
    return result;
}
val designPoint(api::OptimizationProblem& owner, const val& x)
{
    return guarded(
        [&]
        {
            const auto values = numbers<double>(x);
            const auto point = owner.problem().designPoint(values, owner.supplied(values));
            auto result = val::object();
            result.set("variables", array(point.values().variables.values));
            auto variableGradients = val::array();
            for (const auto& gradient : point.values().variables.derivatives)
                variableGradients.call<void>("push", sparseGradient(gradient));
            result.set("variableGradients", variableGradients);
            auto fields = val::array();
            for (const auto& field : point.values().fields)
            {
                auto f = sparseGradient(field.gradient);
                f.set("value", field.value);
                fields.call<void>("push", f);
            }
            result.set("fields", fields);
            return result;
        });
}

std::shared_ptr<api::OptimizationEvaluation> evaluate(api::OptimizationProblem& problem, const val& x,
                                                      const val& references)
{
    return guarded(
        [&]
        {
            design::ModeReferences selected;
            each(references,
                 [&](const val& reference)
                 {
                     const auto key = std::make_pair(idOf(reference["responseId"]), idOf(reference["subcase"]));
                     if (!selected.emplace(key, numbers<double>(reference["shape"])).second)
                         throw std::invalid_argument("duplicate mode reference");
                 });
            return problem.evaluate(numbers<double>(x), selected);
        });
}

val itemObject(const design::ResponseItem& item)
{
    auto result = val::object();
    result.set("subcase", item.subcase);
    result.set("entity", item.entity);
    result.set("component", item.component);
    result.set("layer", item.layer);
    return result;
}
val evaluationValues(api::OptimizationEvaluation& point)
{
    return guarded(
        [&]
        {
            const auto& evaluation = point.evaluation();
            auto result = val::object();
            result.set("x", array(point.x()));
            result.set("subcases", array(evaluation.subcases, "Int32Array"));
            result.set("objective", evaluation.objective ? val(*evaluation.objective) : val::null());
            auto responses = val::array();
            for (const auto& response : evaluation.responses)
            {
                auto r = val::object();
                r.set("id", response.id);
                r.set("label", response.label);
                r.set("values", array(response.values));
                auto items = val::array();
                auto statuses = val::array();
                for (std::size_t i = 0; i < response.itemCount(); ++i)
                {
                    items.call<void>("push", itemObject(response.items[i]));
                    statuses.call<void>("push", std::string(design::itemStatusName(response.statusOf(i))));
                }
                r.set("items", items);
                r.set("statuses", statuses);
                responses.call<void>("push", r);
            }
            result.set("responses", responses);
            auto trackedModes = val::array();
            for (const auto& mode : evaluation.trackedModes)
            {
                auto item = val::object();
                item.set("responseId", mode.responseId);
                item.set("subcase", mode.subcase);
                item.set("mode", mode.mode);
                item.set("mac", mode.mac);
                item.set("isUncertain", mode.isUncertain);
                item.set("shape", array(mode.shape));
                trackedModes.call<void>("push", item);
            }
            result.set("trackedModes", trackedModes);
            auto constraints = val::array();
            for (const auto& c : evaluation.constraints)
            {
                auto value = val::object();
                value.set("setId", c.setId);
                value.set("responseId", c.responseId);
                value.set("item", double(c.item));
                value.set("label", itemObject(c.label));
                value.set("side", std::string(c.side == design::BoundSide::Lower ? "lower" : "upper"));
                value.set("derivativeScale",
                          (c.side == design::BoundSide::Lower ? -1.0 : 1.0) /
                              std::max(std::abs(c.bound), design::DesignConstraints::DEFAULT_CONSTRAINT_SCALE));
                value.set("bound", c.bound);
                value.set("value", c.value);
                value.set("normalised", c.normalised);
                constraints.call<void>("push", value);
            }
            result.set("constraints", constraints);
            return result;
        });
}

val gradients(api::OptimizationEvaluation& point, const val& settings)
{
    return guarded(
        [&]
        {
            design::GradientOptions options;
            options.responseIds = numbers<std::int32_t>(get(settings, "responseIds"));
            const auto method = option<std::string>(settings, "method", "automatic");
            if (method == "chained")
                options.method = design::SensitivityMethod::Chained;
            else if (method == "direct")
                options.method = design::SensitivityMethod::Direct;
            else if (method == "adjoint")
                options.method = design::SensitivityMethod::Adjoint;
            else if (method != "automatic")
                throw std::invalid_argument("unknown gradient method");
            each(get(settings, "itemMasks"),
                 [&](const val& mask)
                 {
                     options.itemMasks.push_back({idOf(mask["responseId"]), numbers<std::uint8_t>(mask["isRetained"])});
                 });
            const auto output = point.gradients(options);
            auto result = val::object();
            result.set("variableCount", double(output.variableCount));
            auto responses = val::array();
            for (const auto& gradient : output.responses)
            {
                auto r = val::object();
                r.set("id", gradient.id);
                r.set("itemCount", double(gradient.itemCount));
                r.set("values", array(gradient.values));
                responses.call<void>("push", r);
            }
            result.set("responses", responses);
            auto constraints = val::array();
            for (const auto& row : output.constraints)
                constraints.call<void>("push", array(row));
            result.set("constraints", constraints);
            result.set("objective", output.objective ? array(*output.objective) : val::null());
            auto report = val::object();
            report.set("solves", double(output.report.solves));
            report.set("factorisations", output.report.factorisations);
            report.set("analyticDerivatives", double(output.report.analyticDerivatives));
            report.set("fieldPerturbations", double(output.report.fieldPerturbations));
            report.set("shapePerturbations", double(output.report.shapePerturbations));
            result.set("report", report);
            return result;
        });
}
design::ProductWeights weightsOf(const val& input)
{
    design::ProductWeights result;
    result.rows = countOf(get(input, "rows"), 1);
    entries(get(input, "responses"),
            [&](const auto& id, const val& weights)
            {
                const auto key = idOf(val::global("Number")(id));
                if (present(get(weights, "rowStart")))
                    result.sparseWeights.emplace(key, design::SparseWeights{numbers<std::size_t>(weights["rowStart"]),
                                                                            numbers<std::size_t>(weights["items"]),
                                                                            numbers<double>(weights["values"])});
                else
                    result.weights.emplace(key, numbers<double>(weights));
            });
    return result;
}
val reportObject(const design::ProductReport& report)
{
    auto result = val::object();
    result.set("solves", double(report.solves));
    result.set("blocks", double(report.blocks));
    result.set("factorisations", report.factorisations);
    result.set("lineariseSeconds", report.lineariseSeconds);
    result.set("productSeconds", report.productSeconds);
    result.set("isLinearisationReused", report.isLinearisationReused);
    result.set("linearisationBytes", double(report.linearisationBytes));
    result.set("adjointSolves", double(report.adjointSolves));
    result.set("isAdjointReused", report.isAdjointReused);
    result.set("hasHessianBlocks", report.hasHessianBlocks);
    result.set("areHessianBlocksReused", report.areHessianBlocksReused);
    result.set("hessianBlocksReason", report.hessianBlocksReason);
    return result;
}
class JsProductSink final : public design::ProductSink
{
public:
    explicit JsProductSink(val callback) : m_callback(std::move(callback)) {}
    void consume(const design::ProductBlock& block) override
    {
        auto value = val::object();
        value.set("first", double(block.first));
        value.set("count", double(block.count));
        value.set("rows", double(block.rows));
        value.set("columns", double(block.columns));
        value.set("values", array(std::vector<double>(block.values.begin(), block.values.end())));
        m_callback(value);
    }

private:
    val m_callback;
};
val jvp(api::OptimizationEvaluation& point, const val& directions, const val& count, const val& callback,
        const val& settings)
{
    return guarded(
        [&]
        {
            JsProductSink sink(callback);
            design::ProductSelection selection;
            selection.responseIds = numbers<std::int32_t>(get(settings, "responseIds"));
            selection.subcases = numbers<std::int32_t>(get(settings, "subcases"));
            return reportObject(point.jvp(numbers<double>(directions), countOf(count, 1),
                                          countOf(get(settings, "block"), 32), sink, selection));
        });
}
val vjp(api::OptimizationEvaluation& point, const val& weights, const val& callback, const val& settings)
{
    return guarded(
        [&]
        {
            JsProductSink sink(callback);
            return reportObject(point.vjp(weightsOf(weights), countOf(get(settings, "block"), 32), sink));
        });
}
val hessianVector(api::OptimizationEvaluation& point, const val& weights, const val& direction, const val& count)
{
    return guarded(
        [&]
        {
            std::vector<double> values;
            const auto report =
                point.hessianVector(weightsOf(weights), numbers<double>(direction), countOf(count, 1), values);
            auto result = val::object();
            result.set("values", array(values));
            result.set("report", reportObject(report));
            return result;
        });
}
val hessian(api::OptimizationEvaluation& point, const val& weights)
{
    return guarded([&] { return array(point.hessian(weightsOf(weights))); });
}

class JsGradientSink final : public design::GradientSink
{
public:
    explicit JsGradientSink(val callback) : m_callback(std::move(callback)) {}
    void consume(const design::GradientBlock& block) override
    {
        auto value = val::object();
        value.set("rowBegin", double(block.rowBegin));
        value.set("rowCount", double(block.rowCount));
        value.set("columnBegin", double(block.columnBegin));
        value.set("columnCount", double(block.columnCount));
        value.set("values", array(std::vector<double>(block.values.begin(), block.values.end())));
        m_callback(value);
    }

private:
    val m_callback;
};
val streamGradients(api::OptimizationEvaluation& point, const val& callback, const val& block)
{
    return guarded(
        [&]
        {
            JsGradientSink sink(callback);
            const auto report = point.streamGradients(sink, countOf(block, 256));
            auto result = val::object();
            result.set("solves", double(report.solves));
            result.set("blocks", double(report.blocks));
            result.set("rows", double(report.plan.size.rowCount));
            result.set("columns", double(report.plan.size.columnCount));
            result.set("orientation", std::string(design::streamOrientationName(report.plan.orientation)));
            return result;
        });
}
template <typename Sparse>
val supportObject(const Sparse& sparse)
{
    auto result = val::object();
    result.set("rows", double(sparse.rows));
    result.set("columns", double(sparse.columns));
    // Sparse indices are copied to Float64Array: exact for the safe integer sizes admitted by this interface.
    result.set("row", array(std::vector<double>(sparse.row.begin(), sparse.row.end())));
    result.set("column", array(std::vector<double>(sparse.column.begin(), sparse.column.end())));
    return result;
}
val tripletsObject(const design::SparseTriplets& sparse)
{
    auto result = supportObject(sparse);
    result.set("values", array(sparse.value));
    return result;
}
val blocks(api::OptimizationEvaluation& point, const val& weights)
{
    return guarded(
        [&]
        {
            const auto chosen = present(weights) ? std::optional(weightsOf(weights)) : std::nullopt;
            const auto output = point.blocks(chosen ? &*chosen : nullptr);
            auto result = val::object();
            auto sets = val::array();
            for (const auto& set : output.sets)
            {
                auto s = val::object();
                s.set("stiffness", tripletsObject(set.stiffness));
                s.set("stiffnessSupport", supportObject(set.stiffnessSupport));
                s.set("transform", tripletsObject(set.transform));
                s.set("equationDofs", array(std::vector<double>(set.equationDofs.begin(), set.equationDofs.end())));
                sets.call<void>("push", s);
            }
            result.set("sets", sets);
            auto subcases = val::array();
            for (const auto& subcase : output.subcases)
            {
                auto s = val::object();
                s.set("subcase", subcase.subcase);
                s.set("set", double(subcase.set));
                s.set("pseudoLoads", tripletsObject(subcase.pseudoLoads));
                s.set("pseudoLoadSupport", supportObject(subcase.pseudoLoadSupport));
                s.set("hessianDisplacement", array(subcase.hessianDisplacement));
                s.set("hessianMixed", array(subcase.hessianMixed));
                s.set("adjoint", array(subcase.adjoint));
                subcases.call<void>("push", s);
            }
            result.set("subcases", subcases);
            result.set("itemWeights", tripletsObject(output.itemWeights));
            result.set("itemWeightSupport", supportObject(output.itemWeightSupport));
            result.set("explicitParts", tripletsObject(output.explicitParts));
            result.set("explicitPartSupport", supportObject(output.explicitPartSupport));
            result.set("itemSubcases", array(output.itemSubcases, "Int32Array"));
            result.set("hasHessian", output.hasHessian);
            result.set("designSupport", supportObject(output.designSupport));
            result.set("hessianDesign", array(output.hessianDesign));
            result.set("hessianReport", reportObject(output.hessianReport));
            result.set("hessianAsymmetry", output.hessianAsymmetry);
            result.set("dofPoints", array(output.dofPoints, "Int32Array"));
            result.set("dofComponents", array(output.dofComponents, "Int8Array"));
            return result;
        });
}
} // namespace

EMSCRIPTEN_BINDINGS(oos_optimization)
{
    using namespace emscripten;
    function("createOptimization", &createOptimization);
    class_<api::OptimizationProblem>("OptimizationProblem")
        .smart_ptr<std::shared_ptr<api::OptimizationProblem>>("OptimizationProblemPointer")
        .function("metadata", &metadata)
        .function("designPoint", &designPoint)
        .function("evaluate", &evaluate);
    class_<api::OptimizationEvaluation>("OptimizationEvaluation")
        .smart_ptr<std::shared_ptr<api::OptimizationEvaluation>>("OptimizationEvaluationPointer")
        .function("values", &evaluationValues)
        .function("analysisResults", &wasm::optimizationResults)
        .function("gradients", &gradients)
        .function("streamGradients", &streamGradients)
        .function("jvp", &jvp)
        .function("vjp", &vjp)
        .function("hessianVector", &hessianVector)
        .function("hessian", &hessian)
        .function("blocks", &blocks)
        .function("clearLinearisation", &api::OptimizationEvaluation::clearLinearisation);
}
