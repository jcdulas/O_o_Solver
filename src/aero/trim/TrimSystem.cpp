#include "oos/aero/trim/TrimSystem.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/dense/DenseLu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::aero
{

namespace
{

std::string labelList(const TrimCase& trimCase, const std::vector<std::size_t>& variables)
{
    std::string text;
    for (const std::size_t v : variables)
        text.append(text.empty() ? "" : ", ").append(trimCase.settings[v].label);
    return text.empty() ? std::string("none") : text;
}

} // namespace

std::vector<double> TrimSystem::solve(const TrimCase& trimCase, const linalg::DenseMatrix& equations,
                                      std::span<const double> constant, base::MemoryBudget& budget)
{
    const std::size_t rows = equations.rows();
    const std::size_t variableCount = trimCase.settings.size();
    if (equations.columns() != variableCount || (!constant.empty() && constant.size() != rows))
        throw std::invalid_argument(base::concatenate(
            {"TrimSystem::solve: expected ", std::to_string(variableCount), " columns and ", std::to_string(rows),
             " constants, found ", std::to_string(equations.columns()), " and ", std::to_string(constant.size())}));
    const std::string name = base::concatenate(
        {"AeroTrim ", std::to_string(trimCase.setId), " (subcase ", std::to_string(trimCase.subcase), ")"});

    // Â: the columns of the linked variables moved onto their independent ones.
    linalg::DenseMatrix reduced(budget, rows, variableCount, "trim equations");
    for (std::size_t x = 0; x < variableCount; ++x)
        if (trimCase.settings[x].state != TrimVariableState::Linked)
            for (std::size_t k = 0; k < rows; ++k)
                reduced(k, x) = equations(k, x);
    for (const AeroLink& link : trimCase.links)
    {
        const std::size_t dependent = *trimCase.find(link.dependent);
        for (const LinkTerm& term : link.terms)
        {
            const std::size_t independent = *trimCase.find(term.label);
            for (std::size_t k = 0; k < rows; ++k)
                reduced(k, independent) -= term.coefficient * equations(k, dependent);
        }
    }

    std::vector<std::size_t> free;
    for (std::size_t x = 0; x < variableCount; ++x)
        if (trimCase.settings[x].state == TrimVariableState::Free)
            free.push_back(x);
    if (free.size() != rows)
        throw base::SolverError(base::concatenate(
            {name, ": expected as many free trim variables as supported DOFs (FreeBodySupport, FreeBodySupportSet: ",
             std::to_string(rows), "), found ", std::to_string(free.size()), " (", labelList(trimCase, free),
             "); fix or free variables on the AeroTrim, or change the FreeBodySupport (a trim with more free variables "
             "than "
             "supports, a trim with redundant control surfaces, is not supported)"}));

    std::vector<double> values(variableCount, 0.0);
    for (std::size_t x = 0; x < variableCount; ++x)
        if (trimCase.settings[x].state == TrimVariableState::Fixed)
            values[x] = trimCase.settings[x].value;
    if (rows > 0)
    {
        // The right-hand side −(Â_X u_X + b), and Â_F scaled by rows and columns.
        std::vector<double> rowScale(rows, 0.0);
        std::vector<double> columnScale(rows, 0.0);
        for (std::size_t k = 0; k < rows; ++k)
            for (std::size_t f = 0; f < rows; ++f)
                rowScale[k] = std::max(rowScale[k], std::abs(reduced(k, free[f])));
        for (std::size_t k = 0; k < rows; ++k)
            rowScale[k] = rowScale[k] > 0.0 ? 1.0 / rowScale[k] : 1.0;
        for (std::size_t f = 0; f < rows; ++f)
        {
            for (std::size_t k = 0; k < rows; ++k)
                columnScale[f] = std::max(columnScale[f], std::abs(rowScale[k] * reduced(k, free[f])));
            columnScale[f] = columnScale[f] > 0.0 ? 1.0 / columnScale[f] : 1.0;
        }
        linalg::DenseMatrix system(budget, rows, rows, "trim system");
        linalg::DenseMatrix rightHandSide(budget, rows, 1, "trim right-hand side");
        for (std::size_t k = 0; k < rows; ++k)
        {
            double sum = constant.empty() ? 0.0 : constant[k];
            for (std::size_t x = 0; x < variableCount; ++x)
                if (trimCase.settings[x].state == TrimVariableState::Fixed)
                    sum += reduced(k, x) * values[x];
            rightHandSide(k, 0) = -rowScale[k] * sum;
            for (std::size_t f = 0; f < rows; ++f)
                system(k, f) = rowScale[k] * reduced(k, free[f]) * columnScale[f];
        }
        const linalg::DenseLu factor(std::move(system), budget);
        if (factor.isSingular() || factor.pivotRatio() < SINGULAR_PIVOT_RATIO)
            throw base::SolverError(base::concatenate(
                {name, ": the free trim variables (", labelList(trimCase, free),
                 ") cannot balance the supported rigid-body motions (the trim equations are singular: a variable "
                 "without aerodynamic or inertial effect on them, or two with the same, or a FreeBodySupport DOF "
                 "holding a "
                 "motion none of them acts on); fix one of them on the AeroTrim, or free another, or support the "
                 "motions the free variables balance"}));
        factor.solve(rightHandSide.view());
        for (std::size_t f = 0; f < rows; ++f)
            values[free[f]] = rightHandSide(f, 0) * columnScale[f];
    }
    for (const AeroLink& link : trimCase.links)
    {
        double value = 0.0;
        for (const LinkTerm& term : link.terms)
            value -= term.coefficient * values[*trimCase.find(term.label)];
        values[*trimCase.find(link.dependent)] = value;
    }
    return values;
}

} // namespace oos::aero
