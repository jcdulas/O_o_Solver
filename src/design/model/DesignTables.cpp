#include "oos/design/model/DesignTables.hpp"

#include "../common/TableColumns.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

using base::ContractReader;
using base::NullPolicy;
using table_columns::FLOAT64;
using table_columns::INT32;
using table_columns::NullableColumn;
using table_columns::optional;
using table_columns::required;
using table_columns::text;
using table_columns::TextColumn;
using table_columns::UTF8;

[[noreturn]] void refuse(const ContractReader& table, std::int64_t row, const std::string& what)
{
    table_columns::refuseRow(table, row, what);
}

std::vector<DesignVariable> readVariables(const ContractReader& table)
{
    const std::span<const std::int32_t> ids = table.values<std::int32_t>("id");
    const TextColumn labels(table, "label");
    const std::span<const double> values = table.values<double>("value");
    const std::span<const double> lower = table.values<double>("lower");
    const std::span<const double> upper = table.values<double>("upper");
    const NullableColumn<double> delta(table, "delta");
    const NullableColumn<std::int32_t> ddval(table, "discreteValuesId");
    std::vector<DesignVariable> variables;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        variables.push_back({ids[r], labels[row], values[r], lower[r], upper[r], delta[row], ddval[row]});
    }
    return variables;
}

std::vector<DesignLink> readLinks(const ContractReader& table)
{
    const std::span<const std::int32_t> dependent = table.values<std::int32_t>("dependentVariable");
    const std::span<const double> constant = table.values<double>("constant");
    const std::span<const double> multiplier = table.values<double>("multiplier");
    const std::span<const std::int32_t> independent = table.values<std::int32_t>("independentVariable");
    const std::span<const double> coefficient = table.values<double>("coefficient");
    std::vector<DesignLink> links;
    std::map<std::int32_t, std::size_t> linkOf;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        const auto [found, isNew] = linkOf.emplace(dependent[r], links.size());
        if (isNew)
            links.push_back({0, dependent[r], constant[r], multiplier[r], {}});
        DesignLink& link = links[found->second];
        if (link.constant != constant[r] || link.multiplier != multiplier[r])
            refuse(table, row,
                   base::concatenate({"expected the DesignVarLink.constant0 and "
                                      "DesignVarLink.multiplier of the DesignVarLink of DesignVar ",
                                      std::to_string(dependent[r]), " on each of its rows, found different values"}));
        link.terms.push_back({independent[r], coefficient[r]});
    }
    return links;
}

std::vector<DesignRelation> readRelations(const ContractReader& table)
{
    const TextColumn cards(table, "relationCard");
    const std::span<const std::int32_t> ids = table.values<std::int32_t>("relationId");
    const TextColumn targetTypes(table, "targetType");
    const TextColumn targetCards(table, "targetCard");
    const std::span<const std::int32_t> targetIds = table.values<std::int32_t>("targetId");
    const TextColumn fields(table, "field");
    const NullableColumn<std::int32_t> variables(table, "designVariable");
    const NullableColumn<double> coefficients(table, "coefficient");
    const NullableColumn<double> constants(table, "constant");
    const NullableColumn<std::int32_t> equations(table, "equationId");
    const NullableColumn<std::int32_t> items(table, "fieldItem");
    const NullableColumn<std::int32_t> argumentIndices(table, "argumentIndex");
    const NullableColumn<double> fieldMin(table, "fieldMin");
    const NullableColumn<double> fieldMax(table, "fieldMax");
    // The DesignConstants labels, read once ("" for a null or an absent column); one per row of the design relations.
    std::vector<std::string> labels(static_cast<std::size_t>(table.rowCount()));
    if (table.hasColumn("constantLabel"))
    {
        const base::Utf8View texts = table.texts("constantLabel", NullPolicy::Allow);
        const base::ValidityView valid = table.validity("constantLabel");
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            if (valid.isValid(row))
                labels[static_cast<std::size_t>(row)] = std::string(texts[row]);
        }
    }

    std::vector<DesignRelation> relations;
    std::vector<std::vector<std::pair<std::int32_t, RelationArgument>>> indexedArguments;
    std::map<std::pair<std::string, std::int32_t>, std::size_t> relationOf;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        const std::string card = cards[row];
        const bool isLinear =
            card == "DesignPropRelation1" || card == "DesignMatRelation1" || card == "DesignElemRelation1";
        const bool isEquation =
            card == "DesignPropRelation2" || card == "DesignMatRelation2" || card == "DesignElemRelation2";
        if (!isLinear && !isEquation)
            refuse(table, row,
                   "expected a relationCard DesignPropRelation1, DesignPropRelation2, DesignMatRelation1, "
                   "DesignMatRelation2, "
                   "DesignElemRelation1 or DesignElemRelation2, "
                   "found '" +
                       card + "'");
        const auto [found, isNew] = relationOf.emplace(std::make_pair(card, ids[r]), relations.size());
        if (isNew)
        {
            const std::optional<TargetKind> kind = findTargetKind(targetTypes[row]);
            if (!kind)
                refuse(table, row,
                       "expected a targetType property, material or element, found '" + targetTypes[row] + "'");
            DesignRelation relation;
            relation.card = card;
            relation.id = ids[r];
            relation.targetKind = *kind;
            relation.targetCard = targetCards[row];
            relation.targetId = targetIds[r];
            relation.field = fields[row];
            relation.fieldItem = items[row].value_or(-1);
            relation.form = isLinear ? RelationForm::Linear : RelationForm::Equation;
            relation.constant = constants[row].value_or(0.0);
            relation.equationId = equations[row].value_or(0);
            relation.lowerBound = fieldMin[row];
            relation.upperBound = fieldMax[row];
            if (isEquation && !equations[row])
                refuse(table, row,
                       "expected the equation_id of " + card + " " + std::to_string(ids[r]) + ", found null");
            relations.push_back(std::move(relation));
            indexedArguments.emplace_back();
        }
        DesignRelation& relation = relations[found->second];
        if (isLinear)
        {
            if (!variables[row] || !coefficients[row])
                refuse(table, row, "expected the design_variable and coefficient of a " + card + " term, found null");
            relation.terms.push_back({*variables[row], *coefficients[row]});
            continue;
        }
        RelationArgument argument;
        if (variables[row])
            argument.variableId = *variables[row];
        else if (!labels[r].empty())
            argument.constantLabel = labels[r];
        else
            refuse(table, row,
                   "expected a design_variable or a constant_label for an argument of " + card + " " +
                       std::to_string(ids[r]) + ", found neither");
        const std::int32_t position =
            argumentIndices[row].value_or(static_cast<std::int32_t>(indexedArguments[found->second].size()));
        indexedArguments[found->second].emplace_back(position, std::move(argument));
    }
    for (std::size_t index = 0; index < relations.size(); ++index)
    {
        auto& arguments = indexedArguments[index];
        std::stable_sort(arguments.begin(), arguments.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t k = 0; k < arguments.size(); ++k)
        {
            if (arguments[k].first != static_cast<std::int32_t>(k))
                throw base::ContractError(base::concatenate(
                    {"table '", table.tableName(), "': expected the argument_index of ", relations[index].card, " ",
                     std::to_string(relations[index].id), " to run from 0 by steps of 1, found ",
                     std::to_string(arguments[k].first), " at position ", std::to_string(k),
                     "; the producer breaks the design table contract (src/design/README.md): take the snapshot "
                     "again with a O_o_Reader that follows it"}));
            relations[index].arguments.push_back(std::move(arguments[k].second));
        }
    }
    return relations;
}

std::vector<std::pair<std::string, double>> readConstants(const ContractReader& table)
{
    const base::Utf8View labels = table.texts("label");
    const std::span<const double> values = table.values<double>("value");
    std::vector<std::pair<std::string, double>> constants;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
        constants.emplace_back(std::string(labels[row]), values[static_cast<std::size_t>(row)]);
    return constants;
}

} // namespace

base::TableContract designVariablesContract()
{
    return {"designVariablesTable",
            {required("id", INT32), text("label"), required("value", FLOAT64), required("lower", FLOAT64),
             required("upper", FLOAT64), required("delta", FLOAT64), optional("discreteValuesId", INT32)}};
}

base::TableContract designLinksContract()
{
    return {"designLinksTable",
            {required("dependentVariable", INT32), required("constant", FLOAT64), required("multiplier", FLOAT64),
             required("independentVariable", INT32), required("coefficient", FLOAT64)}};
}

base::TableContract designRelationsContract()
{
    return {"designRelationsTable",
            {text("relationCard"), required("relationId", INT32), text("targetType"), text("targetCard"),
             required("targetId", INT32), text("field"), required("designVariable", INT32),
             required("coefficient", FLOAT64), required("constant", FLOAT64), required("equationId", INT32),
             optional("fieldItem", INT32), optional("argumentIndex", INT32), optional("constantLabel", UTF8),
             optional("fieldMin", FLOAT64), optional("fieldMax", FLOAT64)}};
}

base::TableContract designConstantsContract()
{
    return {"designConstantsTable", {required("label", UTF8), required("value", FLOAT64)}};
}

base::TableContract designShapesContract()
{
    return {"shapeVariablesTable",
            {required("designVariable", INT32), required("nodeId", INT32), required("frameId", INT32),
             required("coefficient", FLOAT64), required("directionX", FLOAT64), required("directionY", FLOAT64),
             required("directionZ", FLOAT64)}};
}

std::vector<ShapeTerm> readShapeTable(const ContractReader& shapes)
{
    const std::span<const std::int32_t> variables = shapes.values<std::int32_t>("designVariable");
    const std::span<const std::int32_t> grids = shapes.values<std::int32_t>("nodeId");
    const std::span<const std::int32_t> systems = shapes.values<std::int32_t>("frameId");
    const std::span<const double> coefficients = shapes.values<double>("coefficient");
    // O_o_Reader writes NaN (not a null) for a vector it cannot resolve; a null is read as NaN too, so that
    // ShapeRelations::check reports it.
    const NullableColumn<double> x(shapes, "directionX");
    const NullableColumn<double> y(shapes, "directionY");
    const NullableColumn<double> z(shapes, "directionZ");
    constexpr double UNRESOLVED = std::numeric_limits<double>::quiet_NaN();
    std::vector<ShapeTerm> terms;
    for (std::int64_t row = 0; row < shapes.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        terms.push_back({variables[r],
                         grids[r],
                         systems[r],
                         coefficients[r],
                         {x[row].value_or(UNRESOLVED), y[row].value_or(UNRESOLVED), z[row].value_or(UNRESOLVED)}});
    }
    return terms;
}

DesignModelParts readDesignTables(const ContractReader* variables, const ContractReader* links,
                                  const ContractReader* relations, const ContractReader* constants,
                                  std::vector<ExpressionProgram> equations)
{
    DesignModelParts parts;
    if (variables != nullptr)
        parts.variables = readVariables(*variables);
    if (links != nullptr)
        parts.links = readLinks(*links);
    if (relations != nullptr)
        parts.relations = readRelations(*relations);
    if (constants != nullptr)
        parts.constants = readConstants(*constants);
    parts.equations = std::move(equations);
    return parts;
}

} // namespace oos::design
