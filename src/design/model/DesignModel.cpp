#include "oos/design/model/DesignModel.hpp"

#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace oos::design
{

namespace
{

// The DesignConstants labels must be unique: a diagnostic per earlier entry of the same label,
// as the pairwise comparison gave, counted by label (the pairs made thousands of private constants quadratic).
void checkConstants(const std::vector<std::pair<std::string, double>>& constants,
                    std::vector<base::Diagnostic>& diagnostics)
{
    std::unordered_map<std::string_view, std::size_t> seen;
    seen.reserve(constants.size());
    for (const auto& [label, value] : constants)
    {
        const std::size_t earlier = seen[std::string_view(label)]++;
        for (std::size_t k = 0; k < earlier; ++k)
            diagnostics.push_back(
                designProblem(design_code::INVALID_RELATION, "DesignConstants", 0, label,
                              "expected each DesignConstants label once, found '" + label + "' twice; rename one"));
    }
}

} // namespace

std::vector<base::Diagnostic> checkDesignModel(const DesignModelParts& parts)
{
    std::vector<base::Diagnostic> diagnostics;
    const DesignVariables variables(parts.variables);
    variables.check(diagnostics);
    const DesignLinks links(parts.links, variables);
    links.check(variables, diagnostics);
    const DesignRelations relations(parts.relations, parts.equations, parts.constants);
    relations.check(variables, parts.equations, parts.constants, diagnostics);
    const ShapeRelations shapes(parts.shapes, variables);
    shapes.check(variables, diagnostics);
    checkConstants(parts.constants, diagnostics);
    return diagnostics;
}

void throwDesignDiagnostics(const std::string& what, const std::vector<base::Diagnostic>& diagnostics)
{
    if (diagnostics.empty())
        return;
    bool isUnsupported = false;
    std::string message = what;
    message.append(": ").append(std::to_string(diagnostics.size())).append(" problem(s): ");
    for (std::size_t k = 0; k < diagnostics.size(); ++k)
    {
        const base::Diagnostic& diagnostic = diagnostics[k];
        isUnsupported = isUnsupported || diagnostic.code == design_code::UNSUPPORTED_CARD ||
                        diagnostic.code == design_code::UNSUPPORTED_TARGET ||
                        diagnostic.code == design_code::UNSUPPORTED_RESPONSE;
        message.append(k == 0 ? "" : "; ").append(base::formatDiagnostic(diagnostic));
    }
    if (isUnsupported)
        throw base::UnsupportedCardError(message);
    throw base::OptimisationError(message);
}

DesignModel::DesignModel(DesignModelParts parts)
{
    throwDesignDiagnostics("the design model cannot be used", checkDesignModel(parts));
    m_variables = DesignVariables(std::move(parts.variables));
    m_links = DesignLinks(std::move(parts.links), m_variables);
    m_relations = DesignRelations(std::move(parts.relations), parts.equations, parts.constants);
    m_shapes = ShapeRelations(std::move(parts.shapes), m_variables);
    m_equations = std::move(parts.equations);
    m_constants = std::move(parts.constants);
}

std::vector<double> DesignModel::initialValues() const
{
    std::vector<double> x;
    x.reserve(independentCount());
    for (const std::size_t index : independentIndices())
        x.push_back(m_variables[index].initialValue);
    return x;
}

DesignValues DesignModel::evaluate(std::span<const double> x, std::span<const SuppliedField> supplied) const
{
    DesignValues values;
    values.variables = m_links.expand(x, m_variables);
    values.fields = m_relations.evaluate(values.variables, m_variables, supplied);
    values.shapes = m_shapes.evaluate(values.variables, m_variables);
    return values;
}

} // namespace oos::design
