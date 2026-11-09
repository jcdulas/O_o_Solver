#include "oos/design/responses/equation/EquationResponse.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace oos::design
{

namespace
{

constexpr std::size_t NO_INDEX = std::numeric_limits<std::size_t>::max();
// DesignResp2: the component Cm of a DNODE is 1 to 3 (a coordinate of the grid point).
constexpr std::int32_t LAST_COORDINATE = 3;

bool isResponse(ArgumentKind kind)
{
    return kind == ArgumentKind::DesignResp1 || kind == ArgumentKind::DesignResp2;
}

std::string lowerCase(std::string_view text)
{
    std::string lower(text);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}

// The values of response `id` in `subcase` (subcase 0 when the response is global), in item order.
std::vector<const ResponseItem*> itemsOf(const ResponseValues& values, std::int32_t subcase, std::vector<double>& out)
{
    std::vector<const ResponseItem*> items;
    const bool isGlobal = !values.items.empty() && values.items.front().subcase == 0 &&
                          std::all_of(values.items.begin(), values.items.end(),
                                      [](const ResponseItem& item) { return item.subcase == 0; });
    const std::int32_t wanted = isGlobal ? 0 : subcase;
    for (std::size_t i = 0; i < values.items.size(); ++i)
        if (values.items[i].subcase == wanted)
        {
            items.push_back(&values.items[i]);
            out.push_back(values.values[i]);
        }
    return items;
}

} // namespace

EquationResponse::EquationResponse(EquationResponseDefinition definition, const DesignModel& model,
                                   const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& problems)
    : m_definition(std::move(definition))
{
    const auto report = [&](const char* code, std::string field, std::string message)
    {
        problems.push_back(
            designProblem(code, m_definition.card, m_definition.id, std::move(field), std::move(message)));
    };
    if (m_definition.form == EquationForm::Expression)
    {
        const auto& equations = model.equations();
        const auto program = std::find_if(equations.begin(), equations.end(), [this](const ExpressionProgram& candidate)
                                          { return candidate.equationId() == m_definition.equationId; });
        if (program == equations.end())
            report(design_code::MISSING_REFERENCE, "equationId",
                   base::concatenate({"expected a DesignEquation of ID ", std::to_string(m_definition.equationId),
                                      ", found none in the design model"}));
        else
        {
            try
            {
                m_evaluator = std::make_shared<const ExpressionEvaluator>(*program, m_definition.constants);
            }
            catch (const std::invalid_argument& error)
            {
                report(design_code::INVALID_RESPONSE, "equationId", error.what());
            }
        }
    }
    else if (m_definition.callable.empty())
        report(design_code::INVALID_RESPONSE, "groupName",
               "expected the name of the callable of the response, found none");

    const std::span<const std::int32_t> grids = snapshot.nodes().ids();
    for (std::size_t a = 0; a < m_definition.arguments.size(); ++a)
    {
        const ResponseArgument& argument = m_definition.arguments[a];
        m_variableIndices.push_back(NO_INDEX);
        m_constants.push_back(0.0);
        m_relationIndices.push_back(NO_INDEX);
        m_coordinates.push_back(0.0);
        m_coordinateShapes.emplace_back();
        m_labelVariables.emplace_back();
        const std::string field(argumentKindName(argument.kind));
        switch (argument.kind)
        {
        case ArgumentKind::DesignVariable:
        {
            if (!argument.variableByLabel.empty())
            {
                for (const auto& [entity, variable] : argument.variableByLabel)
                {
                    const std::optional<std::size_t> index = model.variables().findIndex(variable);
                    if (!index)
                        report(design_code::MISSING_REFERENCE, field,
                               base::concatenate({"expected a design variable for label ", std::to_string(entity),
                                                  ", found ", std::to_string(variable), ", which is not defined"}));
                    else
                        m_labelVariables.back().emplace_back(entity, *index);
                }
                break;
            }
            const std::optional<std::size_t> index = model.variables().findIndex(argument.id);
            if (!index)
                report(design_code::MISSING_REFERENCE, field,
                       base::concatenate({"expected a DesignVar of ID ", std::to_string(argument.id), ", found none"}));
            else
                m_variableIndices.back() = *index;
            break;
        }
        case ArgumentKind::TableConstant:
        {
            const auto& constants = model.constants();
            const auto found = std::find_if(constants.begin(), constants.end(),
                                            [&](const auto& constant) { return constant.first == argument.label; });
            if (found == constants.end())
                report(design_code::MISSING_REFERENCE, field,
                       base::concatenate({"expected a DesignConstants label '", argument.label, "', found none"}));
            else
                m_constants.back() = found->second;
            break;
        }
        case ArgumentKind::Node:
        {
            const auto found = std::find(grids.begin(), grids.end(), argument.id);
            if (found == grids.end() || argument.component < 1 || argument.component > LAST_COORDINATE)
            {
                report(design_code::MISSING_RESPONSE_TARGET, field,
                       base::concatenate({"expected a grid point of the model and a coordinate 1 to 3 (DesignResp2 "
                                          "DNODE), found grid point ",
                                          std::to_string(argument.id), " component ",
                                          std::to_string(argument.component)}));
                break;
            }
            const auto row = static_cast<std::int64_t>(found - grids.begin());
            const auto component = static_cast<std::size_t>(argument.component - 1);
            m_coordinates.back() = snapshot.nodes().positionsInBasic()[row][component];
            // The shape variables that move the grid point (DVGRID: g = g0 + Σ s·T(g)).
            const std::vector<ShapeBasis>& bases = model.shapes().bases();
            for (std::size_t b = 0; b < bases.size(); ++b)
                for (const ShapeMove& move : bases[b].moves)
                    if (move.gridId == argument.id && move.move[component] != 0.0)
                        m_coordinateShapes.back().emplace_back(b, move.move[component]);
            break;
        }
        case ArgumentKind::DesignResp1:
        case ArgumentKind::DesignResp2:
            break; // resolved by DesignResponses, which knows every response
        default:
        {
            // A designed field: the relation of that card and ID.
            const auto& relations = model.relations().all();
            const auto found = std::find_if(relations.begin(), relations.end(), [&](const DesignRelation& relation)
                                            { return relation.card == field && relation.id == argument.id; });
            if (found == relations.end())
                report(design_code::MISSING_REFERENCE, field,
                       base::concatenate({"expected a ", field, " of ID ", std::to_string(argument.id),
                                          ", found none in the design model"}));
            else
                m_relationIndices.back() = static_cast<std::size_t>(found - relations.begin());
            break;
        }
        }
    }
    if (m_definition.byLabel &&
        std::none_of(m_definition.arguments.begin(), m_definition.arguments.end(),
                     [](const ResponseArgument& argument) { return isResponse(argument.kind); }))
        report(design_code::INVALID_RESPONSE, "by",
               "expected a response among the arguments of a by-label equation (its labels come from them), found "
               "none");
}

std::vector<std::int32_t> EquationResponse::responseArguments() const
{
    std::vector<std::int32_t> ids;
    for (const ResponseArgument& argument : m_definition.arguments)
        if (isResponse(argument.kind))
            ids.push_back(argument.id);
    return ids;
}

std::string EquationResponse::name() const
{
    return base::concatenate({m_definition.card, " ", std::to_string(m_definition.id), " (", m_definition.label, ")"});
}

ArgumentSource EquationResponse::scalarSource(const ResponseArgument& argument, std::int32_t labelEntity) const
{
    const auto a = static_cast<std::size_t>(&argument - m_definition.arguments.data());
    switch (argument.kind)
    {
    case ArgumentKind::DesignVariable:
        if (!m_labelVariables[a].empty())
        {
            for (const auto& [entity, index] : m_labelVariables[a])
                if (entity == labelEntity)
                    return {ArgumentSource::Kind::Variable, 0, index};
            throw base::OptimisationError(base::concatenate(
                {name(), ": expected a design variable for label ", std::to_string(labelEntity), " of argument '",
                 argument.name, "', found none; declare a variable for each label of the argument"}));
        }
        return {ArgumentSource::Kind::Variable, 0, m_variableIndices[a]};
    case ArgumentKind::TableConstant:
        return {ArgumentSource::Kind::Constant, 0, a};
    case ArgumentKind::Node:
        return {ArgumentSource::Kind::Coordinate, 0, a};
    default:
        return {ArgumentSource::Kind::Field, 0, m_relationIndices[a]};
    }
}

double EquationResponse::scalarValue(const ArgumentSource& source, ResponseContext& context) const
{
    switch (source.kind)
    {
    case ArgumentSource::Kind::Variable:
        return context.design->variables.values[source.index];
    case ArgumentSource::Kind::Field:
        return context.design->fields[source.index].value;
    case ArgumentSource::Kind::Coordinate:
    {
        // The base coordinate moved by the shape values of the design point, the bases in order.
        double coordinate = m_coordinates[source.index];
        for (const auto& [basis, rate] : m_coordinateShapes[source.index])
            coordinate += context.design->shapes[basis].value * rate;
        return coordinate;
    }
    default:
    {
        const ResponseArgument& argument = m_definition.arguments[source.index];
        const auto a = source.index;
        return argument.kind == ArgumentKind::TableConstant ? m_constants[a] : m_coordinates[a];
    }
    }
}

EquationResponse::Gathered EquationResponse::gather(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                                    ResponseContext& context) const
{
    return m_definition.byLabel ? gatherByLabel(known, context) : gatherBySubcase(subcases, known, context);
}

EquationResponse::Gathered EquationResponse::gatherBySubcase(std::span<const std::int32_t> subcases,
                                                             const KnownResponses& known,
                                                             ResponseContext& context) const
{
    Gathered gathered;
    for (const std::int32_t subcase : subcases)
        gathered.labels.emplace_back(subcase, 0);
    const std::size_t items = subcases.size();
    // The arguments, as batch arguments: one per argument (an Expression, a named argument of the caller) or one per
    // kind (a deck DesignResp3), each holding `items` rows.
    const bool byKind = m_definition.form == EquationForm::Callable &&
                        std::any_of(m_definition.arguments.begin(), m_definition.arguments.end(),
                                    [](const ResponseArgument& argument) { return argument.name.empty(); });
    std::vector<std::size_t> slotOf;
    for (const ResponseArgument& argument : m_definition.arguments)
    {
        const std::string slotName = byKind ? lowerCase(argumentKindName(argument.kind)) : argument.name;
        const auto found = std::find_if(gathered.arguments.begin(), gathered.arguments.end(),
                                        [&](const BatchArgument& slot) { return byKind && slot.name == slotName; });
        if (found != gathered.arguments.end())
            slotOf.push_back(static_cast<std::size_t>(found - gathered.arguments.begin()));
        else
        {
            slotOf.push_back(gathered.arguments.size());
            gathered.arguments.push_back({slotName, 0, {}});
        }
    }
    gathered.sources.resize(gathered.arguments.size());
    // Per item, per slot, the values of its arguments in order; the width of a slot is fixed by the first item.
    for (std::size_t i = 0; i < items; ++i)
    {
        std::vector<std::vector<double>> itemRows(gathered.arguments.size());
        std::vector<std::vector<ArgumentSource>> itemSources(gathered.arguments.size());
        for (std::size_t a = 0; a < m_definition.arguments.size(); ++a)
        {
            const ResponseArgument& argument = m_definition.arguments[a];
            std::vector<double>& row = itemRows[slotOf[a]];
            std::vector<ArgumentSource>& sources = itemSources[slotOf[a]];
            if (!isResponse(argument.kind))
            {
                const ArgumentSource source = scalarSource(argument, 0);
                row.push_back(scalarValue(source, context));
                sources.push_back(source);
                continue;
            }
            const ResponseValues& values = *known.at(argument.id);
            for (const ResponseItem* item : itemsOf(values, subcases[i], row))
                sources.push_back({ArgumentSource::Kind::Response, argument.id,
                                   static_cast<std::size_t>(item - values.items.data())});
        }
        for (std::size_t s = 0; s < gathered.arguments.size(); ++s)
        {
            BatchArgument& slot = gathered.arguments[s];
            if (i == 0)
                slot.components = itemRows[s].size();
            if (itemRows[s].size() != slot.components || slot.components == 0)
                throw base::OptimisationError(base::concatenate(
                    {name(), ": expected ", std::to_string(slot.components), " values of argument '", slot.name,
                     "' in every subcase, found ", std::to_string(itemRows[s].size()), " in subcase ",
                     std::to_string(subcases[i]),
                     " (its responses have a different number of items there, or none); give the argument's responses "
                     "the same items in every subcase"}));
            slot.values.insert(slot.values.end(), itemRows[s].begin(), itemRows[s].end());
            gathered.sources[s].insert(gathered.sources[s].end(), itemSources[s].begin(), itemSources[s].end());
        }
    }
    return gathered;
}

EquationResponse::Gathered EquationResponse::gatherByLabel(const KnownResponses& known, ResponseContext& context) const
{
    Gathered gathered;
    // The labels: the (subcase, entity) of the items of the first response argument, in its item order.
    for (const ResponseArgument& argument : m_definition.arguments)
    {
        if (!isResponse(argument.kind))
            continue;
        std::set<std::pair<std::int32_t, std::int32_t>> seen;
        for (const ResponseItem& item : known.at(argument.id)->items)
            if (seen.emplace(item.subcase, item.entity).second)
                gathered.labels.emplace_back(item.subcase, item.entity);
        break;
    }
    const std::vector<std::pair<std::int32_t, std::int32_t>>& labels = gathered.labels;
    const std::size_t items = labels.size();
    for (const ResponseArgument& argument : m_definition.arguments)
    {
        BatchArgument slot{argument.name.empty() ? lowerCase(argumentKindName(argument.kind)) : argument.name, 0, {}};
        std::vector<ArgumentSource> sources;
        if (!isResponse(argument.kind))
        {
            slot.components = 1;
            for (const auto& label : labels)
            {
                const ArgumentSource source = scalarSource(argument, label.second);
                slot.values.push_back(scalarValue(source, context));
                sources.push_back(source);
            }
            gathered.arguments.push_back(std::move(slot));
            gathered.sources.push_back(std::move(sources));
            continue;
        }
        const ResponseValues& values = *known.at(argument.id);
        // The items of each label, in item order (one pass: the items of a label are found by a map).
        std::map<std::pair<std::int32_t, std::int32_t>, std::vector<std::size_t>> byLabel;
        for (std::size_t i = 0; i < values.items.size(); ++i)
            byLabel[{values.items[i].subcase, values.items[i].entity}].push_back(i);
        for (std::size_t l = 0; l < items; ++l)
        {
            const auto found = byLabel.find(labels[l]);
            const std::size_t count = found == byLabel.end() ? 0 : found->second.size();
            if (l == 0)
                slot.components = count;
            if (count == 0 || count != slot.components)
                throw base::OptimisationError(base::concatenate(
                    {name(), ": expected ", std::to_string(slot.components), " items of response ",
                     std::to_string(argument.id), " (argument '", slot.name, "') for label ",
                     std::to_string(labels[l].second), " of subcase ", std::to_string(labels[l].first), ", found ",
                     std::to_string(count), "; give the response arguments of a by-label equation the same labels"}));
            for (const std::size_t index : found->second)
            {
                slot.values.push_back(values.values[index]);
                sources.push_back({ArgumentSource::Kind::Response, argument.id, index});
            }
        }
        gathered.arguments.push_back(std::move(slot));
        gathered.sources.push_back(std::move(sources));
    }
    return gathered;
}

std::string EquationResponse::labelOf(const Gathered& gathered, std::size_t item) const
{
    const auto& [subcase, entity] = gathered.labels[item];
    if (m_definition.byLabel)
        return base::concatenate({"label ", std::to_string(entity), ", subcase ", std::to_string(subcase)});
    return subcase == 0 ? std::string("global") : std::string("subcase ").append(std::to_string(subcase));
}

BatchResult EquationResponse::compute(const Gathered& gathered, const CallableRegistry& callables,
                                      std::size_t blockItems, bool withPartials) const
{
    const std::size_t items = gathered.labels.size();
    const auto label = [&](std::size_t i) { return labelOf(gathered, i); };
    if (m_definition.form == EquationForm::Callable)
    {
        const auto found = callables.find(m_definition.callable);
        if (found == callables.end() || found->second == nullptr)
            throw base::OptimisationError(base::concatenate(
                {name(), ": expected the callable '", m_definition.callable,
                 "' to be registered, found none; register it (register_dresp3 for a deck DesignResp3, function= for a "
                 "caller's equation)"}));
        if (withPartials && !found->second->hasGradient())
            throw base::OptimisationError(base::concatenate(
                {name(), ": its gradients need the partial derivatives of the callable '", m_definition.callable,
                 "' (the callable's partials enter the chain of the sensitivities), found no gradient=; "
                 "give gradient= to register_dresp3 or add_equation_response"}));
        BatchCaller caller(*found->second, name(), label, blockItems);
        return caller.run(gathered.arguments, items, withPartials);
    }
    // An Expression: one point per item, the arguments' values in order (every argument scalar per item).
    std::size_t width = 0;
    for (const BatchArgument& argument : gathered.arguments)
        width += argument.components;
    if (width != m_evaluator->freeArguments().size())
        throw base::OptimisationError(
            base::concatenate({name(), ": expected ", std::to_string(m_evaluator->freeArguments().size()),
                               " argument values for the header of DesignEquation ",
                               std::to_string(m_definition.equationId), ", found ", std::to_string(width),
                               " (a DesignResp1 argument gives one value per item); match the "
                               "arguments to the equation"}));
    std::vector<double> points(items * width);
    for (std::size_t i = 0; i < items; ++i)
    {
        std::size_t column = 0;
        for (const BatchArgument& argument : gathered.arguments)
            for (std::size_t c = 0; c < argument.components; ++c)
                points[i * width + column++] = argument.values[i * argument.components + c];
    }
    EvaluationResult evaluated = m_evaluator->evaluate(points, items, withPartials);
    if (!evaluated.isValid())
        throw base::OptimisationError(base::concatenate(
            {name(), ": the equation refuses its arguments: ", base::formatDiagnostic(evaluated.diagnostics.front()),
             " (item 0 is ", label(0), ")"}));
    BatchResult result;
    result.outputs = 1;
    result.values = std::move(evaluated.values);
    if (withPartials)
    {
        // The derivatives of the points (items × width) split into the slots, (items × 1 × components) each.
        std::size_t first = 0;
        for (const BatchArgument& argument : gathered.arguments)
        {
            std::vector<double> partial(items * argument.components);
            for (std::size_t i = 0; i < items; ++i)
                for (std::size_t c = 0; c < argument.components; ++c)
                    partial[i * argument.components + c] = evaluated.derivatives[i * width + first + c];
            result.partials.push_back(std::move(partial));
            first += argument.components;
        }
    }
    return result;
}

ResponseValues EquationResponse::valuesOf(const Gathered& gathered, const BatchResult& result) const
{
    ResponseValues values;
    values.id = m_definition.id;
    values.label = m_definition.label;
    for (std::size_t i = 0; i < gathered.labels.size(); ++i)
        for (std::size_t m = 0; m < result.outputs; ++m)
        {
            values.items.push_back(
                {gathered.labels[i].first, gathered.labels[i].second, static_cast<std::int32_t>(m), 0});
            values.values.push_back(result.values[i * result.outputs + m]);
        }
    return values;
}

std::vector<ItemStatus> EquationResponse::statusesOf(const Gathered& gathered, const KnownResponses& known)
{
    std::vector<ItemStatus> statuses(gathered.labels.size(), ItemStatus::Ok);
    for (std::size_t s = 0; s < gathered.arguments.size(); ++s)
    {
        const std::size_t components = gathered.arguments[s].components;
        for (std::size_t i = 0; i < statuses.size(); ++i)
            for (std::size_t c = 0; c < components; ++c)
            {
                const ArgumentSource& source = gathered.sources[s][i * components + c];
                if (source.kind != ArgumentSource::Kind::Response)
                    continue;
                if (const auto found = known.find(source.responseId); found != known.end())
                    statuses[i] = combinedStatus(statuses[i], found->second->statusOf(source.index));
            }
    }
    return statuses;
}

EquationResponse::Gathered EquationResponse::subset(const Gathered& gathered, std::span<const std::size_t> items)
{
    Gathered result;
    for (const std::size_t i : items)
        result.labels.push_back(gathered.labels[i]);
    for (std::size_t s = 0; s < gathered.arguments.size(); ++s)
    {
        const BatchArgument& argument = gathered.arguments[s];
        BatchArgument slot{argument.name, argument.components, {}};
        std::vector<ArgumentSource> sources;
        for (const std::size_t i : items)
        {
            const auto first = static_cast<std::ptrdiff_t>(i * argument.components);
            const auto last = static_cast<std::ptrdiff_t>((i + 1) * argument.components);
            slot.values.insert(slot.values.end(), argument.values.begin() + first, argument.values.begin() + last);
            sources.insert(sources.end(), gathered.sources[s].begin() + first, gathered.sources[s].begin() + last);
        }
        result.arguments.push_back(std::move(slot));
        result.sources.push_back(std::move(sources));
    }
    return result;
}

ResponseValues EquationResponse::mergedValues(const Gathered& gathered, std::span<const ItemStatus> statuses,
                                              std::span<const std::size_t> okItems, const BatchResult& result) const
{
    ResponseValues values;
    values.id = m_definition.id;
    values.label = m_definition.label;
    const std::size_t outputs = std::max<std::size_t>(result.outputs, 1);
    std::size_t next = 0;
    for (std::size_t i = 0; i < gathered.labels.size(); ++i)
    {
        const bool isOk = next < okItems.size() && okItems[next] == i;
        for (std::size_t m = 0; m < outputs; ++m)
        {
            values.items.push_back(
                {gathered.labels[i].first, gathered.labels[i].second, static_cast<std::int32_t>(m), 0});
            values.values.push_back(isOk ? result.values[next * outputs + m]
                                         : std::numeric_limits<double>::quiet_NaN());
            values.statuses.push_back(isOk ? ItemStatus::Ok : statuses[i]);
        }
        if (isOk)
            ++next;
    }
    return values;
}

ResponseValues EquationResponse::evaluate(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                          ResponseContext& context, const CallableRegistry& callables,
                                          std::size_t blockItems) const
{
    const Gathered gathered = gather(subcases, known, context);
    const std::vector<ItemStatus> statuses = statusesOf(gathered, known);
    std::vector<std::size_t> okItems;
    for (std::size_t i = 0; i < statuses.size(); ++i)
        if (statuses[i] == ItemStatus::Ok)
            okItems.push_back(i);
    if (okItems.size() == statuses.size())
        return valuesOf(gathered, compute(gathered, callables, blockItems, false));
    const Gathered computed = subset(gathered, okItems);
    const BatchResult result = okItems.empty() ? BatchResult{} : compute(computed, callables, blockItems, false);
    return mergedValues(gathered, statuses, okItems, result);
}

std::vector<std::pair<ArgumentSource, double>> EquationResponse::secondOrderTerms(
    std::span<const std::int32_t> subcases, const KnownResponses& known, ResponseContext& context,
    const CallableRegistry& callables, std::size_t blockItems, std::span<const double> itemWeights,
    const std::function<double(const ArgumentSource&)>& directionOf) const
{
    if (m_definition.form != EquationForm::Callable)
        throw base::OptimisationError(base::concatenate(
            {name(),
             ": its second derivatives (hessian_vector) are those of a caller's equation with "
             "hessian_vector=; an expression's are not formed in this version: declare it with function=, "
             "gradient= and hessian_vector="}));
    const auto found = callables.find(m_definition.callable);
    if (found == callables.end() || found->second == nullptr)
        throw base::OptimisationError(
            base::concatenate({name(), ": expected the callable '", m_definition.callable,
                               "' to be registered, found none; register it (function= of add_equation_response)"}));
    if (!found->second->hasHessianVector())
        throw base::OptimisationError(
            base::concatenate({name(),
                               ": its second derivatives need the callable's hessian_vector=, found none; "
                               "give hessian_vector=(x, loads, y, v_x, v_L) -> (w_x, w_L) to add_equation_response"}));
    const Gathered all = gather(subcases, known, context);
    const std::size_t labels = all.labels.size();
    if (labels == 0)
        return {};
    if (itemWeights.size() % labels != 0)
        throw std::invalid_argument(base::concatenate(
            {"EquationResponse::secondOrderTerms: expected the weights of the ", std::to_string(labels),
             " labels' outputs, found ", std::to_string(itemWeights.size()), " weights"}));
    const std::size_t outputs = itemWeights.size() / labels;
    const std::vector<ItemStatus> statuses = statusesOf(all, known);
    // The labels with a weight on an output (the others are not called).
    std::vector<std::size_t> weighted;
    for (std::size_t i = 0; i < labels; ++i)
        if (statuses[i] == ItemStatus::Ok &&
            std::any_of(itemWeights.begin() + static_cast<std::ptrdiff_t>(i * outputs),
                        itemWeights.begin() + static_cast<std::ptrdiff_t>((i + 1) * outputs),
                        [](double y) { return y != 0.0; }))
            weighted.push_back(i);
    if (weighted.empty())
        return {};
    Gathered gathered = subset(all, weighted);
    std::vector<double> weights(weighted.size() * outputs);
    for (std::size_t k = 0; k < weighted.size(); ++k)
        for (std::size_t m = 0; m < outputs; ++m)
            weights[k * outputs + m] = itemWeights[weighted[k] * outputs + m];
    std::vector<BatchArgument> directions; // bounded by the arguments of the callable
    for (std::size_t s = 0; s < gathered.arguments.size(); ++s)
    {
        BatchArgument& argument = gathered.arguments[s];
        argument.isResponse =
            std::any_of(gathered.sources[s].begin(), gathered.sources[s].end(),
                        [](const ArgumentSource& source) { return source.kind == ArgumentSource::Kind::Response; });
        BatchArgument direction{argument.name, argument.components, {}, argument.isResponse};
        direction.values.reserve(argument.values.size());
        for (const ArgumentSource& source : gathered.sources[s])
            direction.values.push_back(directionOf(source));
        directions.push_back(std::move(direction));
    }
    const auto label = [&](std::size_t i) { return labelOf(gathered, i); };
    BatchCaller caller(*found->second, name(), label, blockItems);
    const std::vector<std::vector<double>> products =
        caller.runHessianVector(gathered.arguments, weighted.size(), outputs, weights, directions);
    std::vector<std::pair<ArgumentSource, double>> terms;
    for (std::size_t i = 0; i < weighted.size(); ++i)
        for (std::size_t s = 0; s < gathered.arguments.size(); ++s)
        {
            const std::size_t components = gathered.arguments[s].components;
            for (std::size_t c = 0; c < components; ++c)
                if (const double w = products[s][i * components + c]; w != 0.0)
                    terms.emplace_back(gathered.sources[s][i * components + c], w);
        }
    return terms;
}

EquationPartials EquationResponse::evaluateWithPartials(std::span<const std::int32_t> subcases,
                                                        const KnownResponses& known, ResponseContext& context,
                                                        const CallableRegistry& callables, std::size_t blockItems) const
{
    const Gathered all = gather(subcases, known, context);
    const std::vector<ItemStatus> statuses = statusesOf(all, known);
    std::vector<std::size_t> okItems;
    for (std::size_t i = 0; i < statuses.size(); ++i)
        if (statuses[i] == ItemStatus::Ok)
            okItems.push_back(i);
    const bool isEveryOk = okItems.size() == statuses.size();
    const Gathered gathered = isEveryOk ? all : subset(all, okItems);
    const BatchResult result = gathered.labels.empty() ? BatchResult{} : compute(gathered, callables, blockItems, true);
    EquationPartials partials;
    partials.values = isEveryOk ? valuesOf(gathered, result) : mergedValues(all, statuses, okItems, result);
    const std::size_t outputs = std::max<std::size_t>(result.outputs, 1);
    partials.items.resize(partials.values.itemCount());
    // The partials of the ok items, at their place among every item (the others have none: not differentiated).
    std::vector<std::size_t> placeOf(gathered.labels.size());
    for (std::size_t i = 0; i < gathered.labels.size(); ++i)
        placeOf[i] = isEveryOk ? i : okItems[i];
    for (std::size_t s = 0; s < gathered.arguments.size(); ++s)
    {
        const std::size_t components = gathered.arguments[s].components;
        for (std::size_t i = 0; i < gathered.labels.size(); ++i)
            for (std::size_t m = 0; m < result.outputs; ++m)
                for (std::size_t c = 0; c < components; ++c)
                {
                    const double weight = result.partials[s][(i * result.outputs + m) * components + c];
                    if (weight != 0.0)
                        partials.items[placeOf[i] * outputs + m].push_back(
                            {gathered.sources[s][i * components + c], weight});
                }
    }
    return partials;
}

} // namespace oos::design
