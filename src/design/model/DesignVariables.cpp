#include "oos/design/model/DesignVariables.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

const char* const CARD = "DesignVar";

std::string text(double value)
{
    return std::to_string(value);
}

} // namespace

DesignVariables::DesignVariables(std::vector<DesignVariable> variables) : m_variables(std::move(variables))
{
    m_indexById.reserve(m_variables.size());
    for (std::size_t index = 0; index < m_variables.size(); ++index)
        m_indexById.emplace_back(m_variables[index].id, index);
    std::stable_sort(m_indexById.begin(), m_indexById.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
}

std::optional<std::size_t> DesignVariables::findIndex(std::int32_t id) const noexcept
{
    const auto found = std::lower_bound(m_indexById.begin(), m_indexById.end(), id,
                                        [](const auto& entry, std::int32_t value) { return entry.first < value; });
    if (found == m_indexById.end() || found->first != id)
        return std::nullopt;
    return found->second;
}

void DesignVariables::check(std::vector<base::Diagnostic>& diagnostics) const
{
    for (std::size_t k = 1; k < m_indexById.size(); ++k)
    {
        if (m_indexById[k].first == m_indexById[k - 1].first)
            diagnostics.push_back(designProblem(design_code::INVALID_VARIABLE, CARD, m_indexById[k].first, "id",
                                                "expected a DesignVar ID used once, found it twice; renumber one of "
                                                "the design variables"));
    }
    for (const DesignVariable& variable : m_variables)
    {
        if (variable.id <= 0)
            diagnostics.push_back(
                designProblem(design_code::INVALID_VARIABLE, CARD, variable.id, "id",
                              base::concatenate({"expected a positive ID, found ", std::to_string(variable.id)})));
        if (variable.discreteValues)
            diagnostics.push_back(designProblem(
                design_code::UNSUPPORTED_CARD, CARD, variable.id, "DesignDiscreteValues",
                base::concatenate({"DesignDiscreteValues ", std::to_string(*variable.discreteValues),
                                   " (discrete design variable values) is not supported yet; "
                                   "found on DesignVar ",
                                   std::to_string(variable.id),
                                   ". Leave DesignDiscreteValues blank to design the variable continuously"})));
        if (!(variable.lower <= variable.upper))
            diagnostics.push_back(designProblem(
                design_code::INVALID_VARIABLE, CARD, variable.id, "lowerBound",
                base::concatenate({"expected DesignVar.lowerBound <= DesignVar.upperBound, found "
                                   "DesignVar.lowerBound = ",
                                   text(variable.lower), " and DesignVar.upperBound = ", text(variable.upper),
                                   "; correct the bounds"})));
        else if (!(variable.lower <= variable.initialValue && variable.initialValue <= variable.upper))
            diagnostics.push_back(designProblem(
                design_code::INVALID_VARIABLE, CARD, variable.id, "initialValue",
                base::concatenate(
                    {"expected DesignVar.initialValue between DesignVar.lowerBound = ", text(variable.lower),
                     " and DesignVar.upperBound = ", text(variable.upper), ", found ", text(variable.initialValue),
                     "; move DesignVar.initialValue inside the bounds"})));
        if (variable.moveLimit && !(*variable.moveLimit > 0.0))
            diagnostics.push_back(
                designProblem(design_code::INVALID_VARIABLE, CARD, variable.id, "designVarMoveLimit",
                              base::concatenate({"expected DesignVar.designVarMoveLimit > 0 or blank, found ",
                                                 text(*variable.moveLimit)})));
    }
}

} // namespace oos::design
