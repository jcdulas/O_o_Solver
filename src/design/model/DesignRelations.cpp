#include "oos/design/model/DesignRelations.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace oos::design
{

namespace
{

// PERFORMANCE: the DesignEquation programs by equation ID and the DesignConstants constants by label, indexed once per
// construction or check instead of a linear search per relation and argument (R relations bound to C constants cost R ·
// C: thousands of DesignPropRelation2 of a dozen constants each made the declarations quadratic, 2.9 ms per relation at
// 4 544). The first entry of an ID or label wins, as the linear search did (a label given twice is reported by
// checkDesignModel). Measured: 6 400 DesignPropRelation2 of 12 constants each in one declarations() block: 33.3 s
// before (5.2 ms per relation, almost all the block's rebuild), 0.34 s after (the rebuild 0.12 s).
class EquationLookup
{
public:
    EquationLookup(const std::vector<ExpressionProgram>& equations,
                   const std::vector<std::pair<std::string, double>>& constants)
    {
        m_equations.reserve(equations.size());
        for (const ExpressionProgram& program : equations)
            m_equations.emplace(program.equationId(), &program);
        m_constants.reserve(constants.size());
        for (const auto& [name, value] : constants)
            m_constants.emplace(std::string_view(name), &value);
    }

    [[nodiscard]] const ExpressionProgram* findEquation(std::int32_t id) const
    {
        const auto found = m_equations.find(id);
        return found == m_equations.end() ? nullptr : found->second;
    }

    [[nodiscard]] const double* findConstant(const std::string& label) const
    {
        const auto found = m_constants.find(std::string_view(label));
        return found == m_constants.end() ? nullptr : found->second;
    }

private:
    std::unordered_map<std::int32_t, const ExpressionProgram*> m_equations;
    std::unordered_map<std::string_view, const double*> m_constants;
};

// Adds `derivative` · d(variable)/d(x) to a gradient over the columns of x.
void accumulate(std::vector<std::pair<std::size_t, double>>& gradient,
                const std::vector<std::pair<std::size_t, double>>& variableDerivatives, double derivative)
{
    for (const auto& [column, dv] : variableDerivatives)
    {
        auto found =
            std::find_if(gradient.begin(), gradient.end(), [&](const auto& entry) { return entry.first == column; });
        if (found == gradient.end())
            gradient.emplace_back(column, derivative * dv);
        else
            found->second += derivative * dv;
    }
}

void sortGradient(std::vector<std::pair<std::size_t, double>>& gradient)
{
    std::sort(gradient.begin(), gradient.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
}

} // namespace

std::string_view targetKindName(TargetKind kind) noexcept
{
    switch (kind)
    {
    case TargetKind::Property:
        return "property";
    case TargetKind::Material:
        return "material";
    case TargetKind::Element:
        return "element";
    }
    return "property";
}

std::optional<TargetKind> findTargetKind(std::string_view name) noexcept
{
    for (const TargetKind kind : {TargetKind::Property, TargetKind::Material, TargetKind::Element})
    {
        if (targetKindName(kind) == name)
            return kind;
    }
    return std::nullopt;
}

std::string DesignRelation::label() const
{
    std::string text = card;
    text.append(" ").append(std::to_string(id)).append(" (").append(targetCard).append(" ");
    text.append(std::to_string(targetId)).append(" ").append(field);
    if (fieldItem >= 0)
        text.append("[").append(std::to_string(fieldItem)).append("]");
    return text.append(")");
}

DesignRelations::DesignRelations(std::vector<DesignRelation> relations, const std::vector<ExpressionProgram>& equations,
                                 const std::vector<std::pair<std::string, double>>& constants)
    : m_relations(std::move(relations)), m_bindings(m_relations.size())
{
    const EquationLookup lookup(equations, constants);
    for (std::size_t index = 0; index < m_relations.size(); ++index)
    {
        const DesignRelation& relation = m_relations[index];
        if (relation.form != RelationForm::Equation)
            continue;
        const ExpressionProgram* program = lookup.findEquation(relation.equationId);
        if (program == nullptr || program->arguments().size() != relation.arguments.size())
            continue; // check() reports it
        std::vector<std::pair<std::string, double>> bound;
        std::vector<std::int32_t> freeIds;
        bool isResolved = true;
        for (std::size_t position = 0; position < relation.arguments.size(); ++position)
        {
            const RelationArgument& argument = relation.arguments[position];
            if (!argument.isConstant())
            {
                freeIds.push_back(argument.variableId);
                continue;
            }
            const double* value = lookup.findConstant(argument.constantLabel);
            if (value == nullptr)
                isResolved = false;
            else
                bound.emplace_back(program->arguments()[position], *value);
        }
        if (!isResolved)
            continue;
        m_bindings[index].evaluator = std::make_shared<const ExpressionEvaluator>(*program, bound);
        m_bindings[index].freeVariableIds = std::move(freeIds);
    }
}

void DesignRelations::check(const DesignVariables& variables, const std::vector<ExpressionProgram>& equations,
                            const std::vector<std::pair<std::string, double>>& constants,
                            std::vector<base::Diagnostic>& diagnostics) const
{
    using FieldKey = std::tuple<int, std::string, std::int32_t, std::string, std::int32_t>;
    std::map<FieldKey, std::size_t> fieldOwner;
    std::map<std::pair<std::string, std::int32_t>, std::size_t> idOwner;
    const EquationLookup lookup(equations, constants);
    const auto requireVariable = [&](const DesignRelation& relation, std::int32_t id, const char* field)
    {
        if (!variables.findIndex(id))
            diagnostics.push_back(
                designProblem(design_code::MISSING_REFERENCE, relation.card, relation.id, field,
                              base::concatenate({"expected the ID of a DesignVar, found ", std::to_string(id),
                                                 ", which the design model does not define"})));
    };
    for (std::size_t index = 0; index < m_relations.size(); ++index)
    {
        const DesignRelation& relation = m_relations[index];
        if (relation.id != 0 && !idOwner.emplace(std::make_pair(relation.card, relation.id), index).second)
            diagnostics.push_back(designProblem(design_code::INVALID_RELATION, relation.card, relation.id, "id",
                                                "expected a relation ID used once per card, found it twice"));
        const FieldKey key{static_cast<int>(relation.targetKind), relation.targetCard, relation.targetId,
                           relation.field, relation.fieldItem};
        const auto [owner, isFirst] = fieldOwner.emplace(key, index);
        if (!isFirst)
            diagnostics.push_back(designProblem(
                design_code::FIELD_DRIVEN_TWICE, relation.card, relation.id, "fieldName",
                base::concatenate({"expected each designed field to be driven by one relation, found ",
                                   relation.label(), " and ", m_relations[owner->second].label(),
                                   " on the same field; give one relation with several terms (a field driven by "
                                   "several variables) instead"})));
        switch (relation.form)
        {
        case RelationForm::Linear:
            if (relation.terms.empty())
                diagnostics.push_back(designProblem(design_code::INVALID_RELATION, relation.card, relation.id,
                                                    "designVarIds",
                                                    "expected at least one design variable, found none"));
            for (const RelationTerm& term : relation.terms)
                requireVariable(relation, term.variableId, "designVarIds");
            break;
        case RelationForm::Equation:
        {
            const ExpressionProgram* program = lookup.findEquation(relation.equationId);
            if (program == nullptr)
            {
                diagnostics.push_back(designProblem(design_code::MISSING_REFERENCE, relation.card, relation.id,
                                                    "equationId",
                                                    base::concatenate({"expected the ID of a DesignEquation, found ",
                                                                       std::to_string(relation.equationId),
                                                                       ", which the design model does not define"})));
                break;
            }
            if (program->arguments().size() != relation.arguments.size())
                diagnostics.push_back(designProblem(
                    design_code::INVALID_RELATION, relation.card, relation.id, "equationId",
                    base::concatenate({"expected as many DesignVar and DesignConstants arguments as ", program->label(),
                                       " has (", std::to_string(program->arguments().size()), "), found ",
                                       std::to_string(relation.arguments.size())})));
            for (const RelationArgument& argument : relation.arguments)
            {
                if (!argument.isConstant())
                    requireVariable(relation, argument.variableId, "designVarIds");
                else if (lookup.findConstant(argument.constantLabel) == nullptr)
                    diagnostics.push_back(designProblem(
                        design_code::MISSING_REFERENCE, relation.card, relation.id, "constantLabels",
                        base::concatenate({"expected the label of a DesignConstants constant, found '",
                                           argument.constantLabel, "', which the design model does not define"})));
            }
            break;
        }
        case RelationForm::Supplied:
            if (relation.arguments.empty())
                diagnostics.push_back(designProblem(design_code::INVALID_RELATION, relation.card, relation.id,
                                                    "designVarIds",
                                                    "expected at least one design variable, found none"));
            for (const RelationArgument& argument : relation.arguments)
            {
                if (argument.isConstant())
                    diagnostics.push_back(
                        designProblem(design_code::INVALID_RELATION, relation.card, relation.id, "constantLabels",
                                      "expected design variables only for a relation given by a "
                                      "function, found a DesignConstants label; pass constants to the "
                                      "function itself"));
                else
                    requireVariable(relation, argument.variableId, "designVarIds");
            }
            break;
        }
    }
}

std::vector<FieldValue> DesignRelations::evaluate(const VariableValues& values, const DesignVariables& variables,
                                                  std::span<const SuppliedField> supplied) const
{
    const auto derivativesOf = [&](std::int32_t id) -> const std::vector<std::pair<std::size_t, double>>&
    { return values.derivatives[*variables.findIndex(id)]; };
    const auto valueOf = [&](std::int32_t id) { return values.values[*variables.findIndex(id)]; };

    std::vector<const SuppliedField*> suppliedOf(m_relations.size(), nullptr);
    for (const SuppliedField& field : supplied)
    {
        if (field.relationIndex >= m_relations.size() ||
            m_relations[field.relationIndex].form != RelationForm::Supplied)
            throw std::invalid_argument(base::concatenate({"design point: a supplied value names relation ",
                                                           std::to_string(field.relationIndex),
                                                           ", which is not a relation given by a function; supply "
                                                           "values for the relations of function= only"}));
        suppliedOf[field.relationIndex] = &field;
    }

    std::vector<FieldValue> fields(m_relations.size());
    for (std::size_t index = 0; index < m_relations.size(); ++index)
    {
        const DesignRelation& relation = m_relations[index];
        FieldValue& field = fields[index];
        switch (relation.form)
        {
        case RelationForm::Linear:
        {
            // DesignPropRelation1: field = C0 + Σ COEFi · DVIDi, summed in the written order.
            double sum = relation.constant;
            for (const RelationTerm& term : relation.terms)
            {
                sum += term.coefficient * valueOf(term.variableId);
                accumulate(field.gradient, derivativesOf(term.variableId), term.coefficient);
            }
            field.value = sum;
            break;
        }
        case RelationForm::Equation:
        {
            const EquationBinding& binding = m_bindings[index];
            std::vector<double> point;
            point.reserve(binding.freeVariableIds.size());
            for (const std::int32_t id : binding.freeVariableIds)
                point.push_back(valueOf(id));
            const EvaluationResult result = binding.evaluator->evaluate(point, 1, true);
            if (!result.isValid())
            {
                std::string texts;
                for (const base::Diagnostic& diagnostic : result.diagnostics)
                    texts.append(texts.empty() ? "" : "; ").append(base::formatDiagnostic(diagnostic));
                throw base::OptimisationError(base::concatenate(
                    {relation.label(), ": its equation cannot be evaluated at this design point: ", texts,
                     "; keep the design variables inside the domain of the equation (bounds of the DESVARs)"}));
            }
            field.value = result.values[0];
            for (std::size_t k = 0; k < binding.freeVariableIds.size(); ++k)
                accumulate(field.gradient, derivativesOf(binding.freeVariableIds[k]), result.derivatives[k]);
            break;
        }
        case RelationForm::Supplied:
        {
            const SuppliedField* given = suppliedOf[index];
            if (given == nullptr || given->gradient.size() != relation.arguments.size())
                throw std::invalid_argument(base::concatenate(
                    {relation.label(), ": expected a supplied value and a gradient of ",
                     std::to_string(relation.arguments.size()), " derivatives (one per variable), found ",
                     given == nullptr ? std::string("none") : std::to_string(given->gradient.size())}));
            field.value = given->value;
            for (std::size_t k = 0; k < relation.arguments.size(); ++k)
                accumulate(field.gradient, derivativesOf(relation.arguments[k].variableId), given->gradient[k]);
            break;
        }
        }
        sortGradient(field.gradient);
    }
    return fields;
}

} // namespace oos::design
