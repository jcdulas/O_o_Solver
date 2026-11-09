#include "oos/design/responses/ResponseTables.hpp"

#include "../common/TableColumns.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"

#include <algorithm>
#include <charconv>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

namespace oos::design
{

namespace
{

using namespace table_columns;

// The DesignResp1 types of other analyses, refused with the version that brings them (none is planned for the types
// left).
std::string laterStepOf(const std::string& /*rtype*/)
{
    return "no planned version";
}

// ATTA (an int32) or ATTB (a float64) of a row: O_o_Reader gives them as the field as written (utf8, data contract 32:
// an ATTA may be a word for some response types), a table built by a caller as numbers. Nothing for a null or blank
// field; refused (refuseRow) for a text that is not a number.
template <typename T>
std::optional<T> numericAttribute(const base::ContractReader& table, const std::string& name, std::int64_t row)
{
    if (!table.hasColumn(name) || !table.validity(name).isValid(row))
        return std::nullopt;
    if (table.columnType(name) != UTF8)
        return table.values<T>(name, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
    std::string text(table.texts(name, base::NullPolicy::Allow)[row]);
    text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
    if (!text.empty() && text.front() == '+')
        text.erase(0, 1); // std::from_chars takes no leading plus
    if (text.empty())
        return std::nullopt;
    if constexpr (std::is_floating_point_v<T>)
    {
        // In the bulk data format a real may omit the E of its exponent ("1.+6", "2.5-3"); put it back.
        for (std::size_t at = 1; at < text.size(); ++at)
        {
            const char previous = text[at - 1];
            if ((text[at] == '+' || text[at] == '-') && previous != 'E' && previous != 'e' && previous != 'D' &&
                previous != 'd')
            {
                text.insert(at, 1, 'E');
                break;
            }
        }
        std::replace(text.begin(), text.end(), 'D', 'E');
        std::replace(text.begin(), text.end(), 'd', 'E');
    }
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size())
        refuseRow(table, row, "expected a number in " + name + ", found '" + text + "'");
    return value;
}

} // namespace

base::TableContract designResponsesContract()
{
    return {"designResponsesTable",
            {required("id", INT32), text("label"), text("responseType"), optionalText("propertyType"),
             optional("region", INT32), base::ColumnContract{"attributeA", UTF8, true, {INT32}},
             base::ColumnContract{"attributeB", UTF8, true, {FLOAT64}}, optional("targets", INT32_LIST),
             optional("isObjective", BOOLEAN), optionalText("objectiveSense"), optionalText("responseCard"),
             optional("equationId", INT32), optionalText("function"), optionalText("group"), optionalText("userType")}};
}

base::TableContract designResponseArgumentsContract()
{
    return {"designResponseArgumentsTable",
            {required("responseId", INT32), text("kind"), required("position", INT32), optional("id", INT32),
             optional("component", INT32), optional("label", UTF8)}};
}

ResponseParts readResponseTables(const base::ContractReader& responses, const base::ContractReader* arguments)
{
    const std::span<const std::int32_t> ids = responses.values<std::int32_t>("id");
    const TextColumn labels(responses, "label");
    const TextColumn types(responses, "responseType");
    const TextColumn propertyTypes(responses, "propertyType");
    const NullableColumn<std::int32_t> regions(responses, "region");
    const IntListColumn targets(responses, "targets");
    const TextColumn cards(responses, "responseCard");
    const NullableColumn<std::int32_t> equations(responses, "equationId");
    const TextColumn functions(responses, "function");
    const TextColumn groups(responses, "group");
    const TextColumn userTypes(responses, "userType");

    ResponseParts parts;
    std::map<std::int32_t, std::size_t> equationOf;
    for (std::int64_t row = 0; row < responses.rowCount(); ++row)
    {
        const std::int32_t id = ids[static_cast<std::size_t>(row)];
        const std::string type = types[row];
        // The card type (DesignResp1, DesignResp2, DesignResp3); without it, the response type says it.
        std::string card = cards[row];
        if (card.empty())
            card = type == "DesignResp2" || type == "DesignResp3" ? type : std::string("DesignResp1");
        if (card == "DesignResp1")
        {
            const std::optional<ResponseType> found = findResponseType(type);
            if (!found)
                throw base::UnsupportedCardError(
                    base::concatenate({"DesignResp1 ", std::to_string(id), ": DesignResp1.responseKind ", type,
                                       " is not a response type of this version; it comes with ", laterStepOf(type),
                                       "; remove the response or replace it by a supported "
                                       "response type"}));
            if (isExtensionType(*found))
                throw base::UnsupportedCardError(base::concatenate(
                    {"DesignResp1 ", std::to_string(id), ": DesignResp1.responseKind ", type,
                     " is not a response type of the input format; it is a response kind of O_o_Solver's optimisation "
                     "aids: declare it with DesignProblem.add_response(kind=...)"}));
            DesignResp1Definition definition;
            definition.id = id;
            definition.label = labels[row];
            definition.type = *found;
            definition.propertyType = propertyTypes[row];
            definition.region = regions[row];
            definition.attributeA = numericAttribute<std::int32_t>(responses, "attributeA", row);
            definition.attributeB = numericAttribute<double>(responses, "attributeB", row);
            definition.targets = targets[row];
            parts.designResp1.push_back(std::move(definition));
            continue;
        }
        if (card != "DesignResp2" && card != "DesignResp3")
            refuseRow(responses, row,
                      "expected a responseCard DesignResp1, DesignResp2 or DesignResp3, found '" + card + "'");
        EquationResponseDefinition definition;
        definition.id = id;
        definition.label = labels[row];
        definition.card = card;
        definition.region = regions[row];
        if (card == "DesignResp2")
        {
            if (!functions[row].empty())
                throw base::UnsupportedCardError(
                    base::concatenate({"DesignResp2 ", std::to_string(id), ": DesignResp2.function = ", functions[row],
                                       " is not supported; write the function as a DesignEquation and give its "
                                       "DesignResp2.equationId"}));
            if (!equations[row])
                refuseRow(responses, row,
                          "expected the equation_id of DesignResp2 " + std::to_string(id) + ", found null");
            definition.form = EquationForm::Expression;
            definition.equationId = *equations[row];
        }
        else
        {
            definition.form = EquationForm::Callable;
            definition.callable = groups[row] + "/" + userTypes[row];
        }
        equationOf.emplace(id, parts.equations.size());
        parts.equations.push_back(std::move(definition));
    }
    if (arguments == nullptr)
        return parts;

    const std::span<const std::int32_t> owners = arguments->values<std::int32_t>("responseId");
    const TextColumn kinds(*arguments, "kind");
    const std::span<const std::int32_t> positions = arguments->values<std::int32_t>("position");
    const NullableColumn<std::int32_t> argumentIds(*arguments, "id");
    const NullableColumn<std::int32_t> components(*arguments, "component");
    const TextColumn argumentLabels(*arguments, "label");
    std::vector<std::vector<std::pair<std::int32_t, ResponseArgument>>> indexed(parts.equations.size());
    for (std::int64_t row = 0; row < arguments->rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        const std::string kindName = kinds[row];
        // The ATTi of a DesignResp1 are rows of kind "attribute" (O_o_Reader contract 39); the solver reads what a
        // DesignResp1 measures from the targets of designResponsesTable.
        if (kindName == "attribute")
            continue;
        const auto owner = equationOf.find(owners[r]);
        if (owner == equationOf.end())
            refuseRow(*arguments, row,
                      "expected the response_id of a DesignResp2 or DesignResp3 of designResponsesTable, found " +
                          std::to_string(owners[r]));
        if (kindName == "DesignLoadRelation1")
            throw base::UnsupportedCardError(base::concatenate(
                {parts.equations[owner->second].card, " ", std::to_string(owners[r]),
                 ": DesignLoadRelation1 arguments are not supported (DesignLoadRelation1 relates loads, which are not "
                 "designed); remove them from the equation"}));
        const std::optional<ArgumentKind> kind = findArgumentKind(kindName);
        if (!kind)
            refuseRow(*arguments, row, "expected an argument kind DesignVar ... DesignResp2, found '" + kindName + "'");
        ResponseArgument argument;
        argument.kind = *kind;
        argument.id = argumentIds[row].value_or(0);
        argument.component = components[row].value_or(0);
        argument.label = argumentLabels[row];
        if (*kind != ArgumentKind::TableConstant && !argumentIds[row])
            refuseRow(*arguments, row, "expected the id of a " + kindName + " argument, found null");
        if (*kind == ArgumentKind::TableConstant && argument.label.empty())
            refuseRow(*arguments, row, "expected the label of a DesignConstants argument, found null");
        indexed[owner->second].emplace_back(positions[r], std::move(argument));
    }
    for (std::size_t e = 0; e < indexed.size(); ++e)
    {
        auto& list = indexed[e];
        std::stable_sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t k = 0; k < list.size(); ++k)
        {
            if (list[k].first != static_cast<std::int32_t>(k))
                throw base::ContractError(base::concatenate(
                    {"table '", arguments->tableName(), "': expected the positions of the arguments of ",
                     parts.equations[e].card, " ", std::to_string(parts.equations[e].id),
                     " to run from 0 by steps of 1, found ", std::to_string(list[k].first), " at place ",
                     std::to_string(k),
                     " (src/design/README.md); take the snapshot again with a O_o_Reader that follows the contract"}));
            parts.equations[e].arguments.push_back(std::move(list[k].second));
        }
    }
    return parts;
}

std::optional<TableObjective> readTableObjective(const base::ContractReader& responses)
{
    if (!responses.hasColumn("isObjective"))
        return std::nullopt;
    const base::BooleanView flags = responses.booleans("isObjective", base::NullPolicy::Allow);
    const base::ValidityView valid = responses.validity("isObjective");
    const TextColumn senses(responses, "objectiveSense");
    const std::span<const std::int32_t> ids = responses.values<std::int32_t>("id");
    for (std::int64_t row = 0; row < responses.rowCount(); ++row)
        if (valid.isValid(row) && flags[row])
            return TableObjective{ids[static_cast<std::size_t>(row)], senses[row] == "max"};
    return std::nullopt;
}

} // namespace oos::design
