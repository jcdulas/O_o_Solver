#include "oos/design/model/DesignLinks.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

const char* const CARD = "DesignVarLink";

} // namespace

DesignLinks::DesignLinks(std::vector<DesignLink> links, const DesignVariables& variables)
    : m_links(std::move(links)), m_linkOfVariable(variables.size(), -1)
{
    for (std::size_t link = 0; link < m_links.size(); ++link)
    {
        const std::optional<std::size_t> dependent = variables.findIndex(m_links[link].dependentId);
        if (dependent && m_linkOfVariable[*dependent] < 0)
            m_linkOfVariable[*dependent] = static_cast<std::int64_t>(link);
    }
    for (std::size_t index = 0; index < variables.size(); ++index)
    {
        if (m_linkOfVariable[index] < 0)
            m_independent.push_back(index);
    }
}

void DesignLinks::check(const DesignVariables& variables, std::vector<base::Diagnostic>& diagnostics) const
{
    std::vector<std::int32_t> dependentCount(variables.size(), 0);
    for (const DesignLink& link : m_links)
    {
        const std::optional<std::size_t> dependent = variables.findIndex(link.dependentId);
        if (!dependent)
        {
            diagnostics.push_back(designProblem(
                design_code::MISSING_REFERENCE, CARD, link.id, "dependentVarId",
                base::concatenate({"expected the ID of a DesignVar, found ", std::to_string(link.dependentId),
                                   ", which the design model does not define"})));
            continue;
        }
        if (++dependentCount[*dependent] == 2)
            diagnostics.push_back(designProblem(
                design_code::INVALID_LINK, CARD, link.id, "dependentVarId",
                base::concatenate(
                    {"expected DesignVar ", std::to_string(link.dependentId),
                     " to be the dependent variable of one DesignVarLink, found it in two; merge the links"})));
        if (link.terms.empty())
            diagnostics.push_back(designProblem(design_code::INVALID_LINK, CARD, link.id, "independentIds",
                                                "expected at least one independent variable, found none"));
        for (const LinkTerm& term : link.terms)
        {
            const std::optional<std::size_t> independent = variables.findIndex(term.variableId);
            if (!independent)
                diagnostics.push_back(designProblem(
                    design_code::MISSING_REFERENCE, CARD, link.id, "independentIds",
                    base::concatenate({"expected the ID of a DesignVar, found ", std::to_string(term.variableId),
                                       ", which the design model does not define"})));
            else if (m_linkOfVariable[*independent] >= 0)
                diagnostics.push_back(designProblem(
                    design_code::INVALID_LINK, CARD, link.id, "independentIds",
                    base::concatenate(
                        {"expected an independent variable, found DesignVar ", std::to_string(term.variableId),
                         ", which is the dependent variable of a DesignVarLink (a DesignVarLink.dependentVarId "
                         "cannot be an DesignVarLink.independentIds); "
                         "link to its own independent variables instead"})));
        }
    }
}

VariableValues DesignLinks::expand(std::span<const double> x, const DesignVariables& variables) const
{
    if (x.size() != m_independent.size())
        throw std::invalid_argument(
            base::concatenate({"design point: expected one value per independent design "
                               "variable (",
                               std::to_string(m_independent.size()), "), found ", std::to_string(x.size())}));
    VariableValues result;
    result.values.assign(variables.size(), 0.0);
    result.derivatives.assign(variables.size(), {});
    std::vector<std::int64_t> columnOf(variables.size(), -1);
    for (std::size_t column = 0; column < m_independent.size(); ++column)
    {
        const std::size_t index = m_independent[column];
        columnOf[index] = static_cast<std::int64_t>(column);
        result.values[index] = x[column];
        result.derivatives[index].emplace_back(column, 1.0);
    }
    for (std::size_t index = 0; index < variables.size(); ++index)
    {
        const std::int64_t linkIndex = m_linkOfVariable[index];
        if (linkIndex < 0)
            continue;
        // DesignVarLink: DDVID = C0 + CMULT · Σ Ci · IDVi; d(DDVID)/d(IDVi) = CMULT · Ci (terms on the same IDVi summed
        // in the written order).
        const DesignLink& link = m_links[static_cast<std::size_t>(linkIndex)];
        double sum = 0.0;
        std::vector<std::pair<std::size_t, double>>& derivative = result.derivatives[index];
        for (const LinkTerm& term : link.terms)
        {
            const std::size_t column =
                static_cast<std::size_t>(columnOf[*variables.findIndex(term.variableId)]); // checked by check()
            sum += term.coefficient * x[column];
            bool isMerged = false;
            for (auto& [existing, value] : derivative)
            {
                if (existing == column)
                {
                    value += link.multiplier * term.coefficient;
                    isMerged = true;
                }
            }
            if (!isMerged)
                derivative.emplace_back(column, link.multiplier * term.coefficient);
        }
        result.values[index] = link.constant + link.multiplier * sum;
    }
    return result;
}

} // namespace oos::design
