#include "oos/design/model/ShapeRelations.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

constexpr const char* DVGRID = "DesignNodeMove";

[[nodiscard]] bool isFinite(const std::array<double, 3>& vector) noexcept
{
    return std::isfinite(vector[0]) && std::isfinite(vector[1]) && std::isfinite(vector[2]);
}

[[nodiscard]] std::string gridText(const ShapeTerm& term)
{
    return base::concatenate(
        {"grid point ", std::to_string(term.gridId), " (DesignNodeMove.frameId ", std::to_string(term.frameId), ")"});
}

} // namespace

ShapeRelations::ShapeRelations(std::vector<ShapeTerm> terms, const DesignVariables& variables)
    : m_terms(std::move(terms))
{
    // Per variable (in the variables' order), per grid point (in increasing ID): Σ COEFF·N, the terms added in their
    // order (several DesignNodeMove entries of one variable and grid point add as vectors).
    std::map<std::size_t, std::map<std::int32_t, std::array<double, 3>>> moves;
    for (const ShapeTerm& term : m_terms)
    {
        const std::optional<std::size_t> variable = variables.findIndex(term.variableId);
        if (!variable)
            continue;
        std::array<double, 3>& move = moves[*variable][term.gridId];
        for (std::size_t k = 0; k < 3; ++k)
            move[k] += term.coefficient * term.direction[k];
    }
    for (const auto& [variable, grids] : moves)
    {
        ShapeBasis basis;
        basis.variable = variable;
        basis.variableId = variables[variable].id;
        for (const auto& [grid, move] : grids)
            basis.moves.push_back({grid, move});
        m_bases.push_back(std::move(basis));
    }
}

void ShapeRelations::check(const DesignVariables& variables, std::vector<base::Diagnostic>& diagnostics) const
{
    for (const ShapeTerm& term : m_terms)
    {
        if (!variables.findIndex(term.variableId))
            diagnostics.push_back(
                designProblem(design_code::MISSING_REFERENCE, DVGRID, term.variableId, "designVarId",
                              base::concatenate({"expected a DesignVar ", std::to_string(term.variableId),
                                                 " for the DesignNodeMove of ", gridText(term),
                                                 ", found none; define the design variable"})));
        if (!isFinite(term.direction))
            diagnostics.push_back(designProblem(
                design_code::INVALID_SHAPE, DVGRID, term.variableId, "directionX",
                base::concatenate({"expected the vector of the DesignNodeMove of ", gridText(term),
                                   " in the basic system, found none: O_o_Reader could not resolve it (the grid point "
                                   "or the system DesignNodeMove.frameId does not exist); correct "
                                   "DesignNodeMove.nodeId or DesignNodeMove.frameId"})));
        else if (term.direction[0] == 0.0 && term.direction[1] == 0.0 && term.direction[2] == 0.0)
            diagnostics.push_back(designProblem(
                design_code::INVALID_SHAPE, DVGRID, term.variableId, "directionX",
                base::concatenate({"expected a vector N with at least one Ni other than 0.0 for ", gridText(term),
                                   ", found (0, 0, 0); give the direction of the move"})));
        if (!std::isfinite(term.coefficient))
            diagnostics.push_back(designProblem(
                design_code::INVALID_SHAPE, DVGRID, term.variableId, "coefficient",
                base::concatenate({"expected a finite DesignNodeMove.coefficient for the DesignNodeMove of ",
                                   gridText(term), ", found ", std::to_string(term.coefficient)})));
    }
}

std::vector<FieldValue> ShapeRelations::evaluate(const VariableValues& values, const DesignVariables& variables) const
{
    std::vector<FieldValue> shapes;
    shapes.reserve(m_bases.size());
    for (const ShapeBasis& basis : m_bases)
    {
        // DesignNodeMove: the change of the variable from its initial value moves the grid points, not the value of the
        // variable itself.
        FieldValue shape;
        shape.value = values.values[basis.variable] - variables[basis.variable].initialValue;
        shape.gradient = values.derivatives[basis.variable];
        shapes.push_back(std::move(shape));
    }
    return shapes;
}

} // namespace oos::design
