#include "oos/design/optimiser/LinearProgram.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace oos::design
{

namespace
{

// A tableau entry below this (relative to the largest entry of the problem) is not a pivot: the pivots of a sizing
// LP are O(1) after scaling, and an entry 1e-11 times the largest is round-off of the eliminations.
constexpr double PIVOT_TOLERANCE = 1.0e-11;
// A reduced cost above −COST_TOLERANCE · (1 + max |c|) does not improve the objective: round-off of the eliminations.
constexpr double COST_TOLERANCE = 1.0e-11;
// The sum of the artificial variables at the end of phase 1 above this (relative to the right-hand sides) means that
// no point satisfies the rows.
constexpr double FEASIBILITY_TOLERANCE = 1.0e-9;
// Bland's rule terminates; this bound only guards against a loop made by round-off.
constexpr std::size_t PIVOT_LIMIT_PER_COLUMN = 50;

class Tableau
{
public:
    Tableau(base::MemoryBudget& budget, std::size_t rows, std::size_t columns)
        : m_rows(rows),
          m_width(columns + 1),
          m_values(budget, (rows + 1) * (columns + 1), "simplex tableau"),
          m_basis(rows, 0)
    {
    }

    [[nodiscard]] double& at(std::size_t row, std::size_t column) noexcept { return m_values[row * m_width + column]; }
    [[nodiscard]] double& rhs(std::size_t row) noexcept { return at(row, m_width - 1); }
    [[nodiscard]] double& cost(std::size_t column) noexcept { return at(m_rows, column); }
    [[nodiscard]] std::size_t rows() const noexcept { return m_rows; }
    [[nodiscard]] std::size_t columns() const noexcept { return m_width - 1; }
    [[nodiscard]] std::vector<std::size_t>& basis() noexcept { return m_basis; }

    void pivot(std::size_t row, std::size_t column)
    {
        const double inverse = 1.0 / at(row, column);
        double* pivotRow = &m_values[row * m_width];
        for (std::size_t j = 0; j < m_width; ++j)
            pivotRow[j] *= inverse;
        pivotRow[column] = 1.0;
        for (std::size_t i = 0; i <= m_rows; ++i)
        {
            if (i == row)
                continue;
            double* target = &m_values[i * m_width];
            const double factor = target[column];
            if (factor == 0.0)
                continue;
            for (std::size_t j = 0; j < m_width; ++j)
                target[j] -= factor * pivotRow[j];
            target[column] = 0.0;
        }
        m_basis[row] = column;
    }

    // Bland's rule: the first improving column below `enteringLimit`, the leaving row of the smallest ratio (ties: the
    // smallest basic column). Returns false when optimal; throws nothing; `isUnbounded` set when a column improves
    // without a limiting row.
    bool step(std::size_t enteringLimit, double costTolerance, double pivotTolerance, bool& isUnbounded)
    {
        std::optional<std::size_t> entering;
        for (std::size_t j = 0; j < enteringLimit; ++j)
            if (cost(j) < -costTolerance)
            {
                entering = j;
                break;
            }
        if (!entering)
            return false;
        std::optional<std::size_t> leaving;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < m_rows; ++i)
        {
            const double entry = at(i, *entering);
            if (entry <= pivotTolerance)
                continue;
            const double ratio = rhs(i) / entry;
            if (ratio < best || (ratio == best && leaving && m_basis[i] < m_basis[*leaving]))
            {
                best = ratio;
                leaving = i;
            }
        }
        if (!leaving)
        {
            isUnbounded = true;
            return false;
        }
        pivot(*leaving, *entering);
        return true;
    }

private:
    std::size_t m_rows;
    std::size_t m_width;
    base::BudgetedBuffer<double> m_values;
    std::vector<std::size_t> m_basis;
};

} // namespace

LinearProgramSolution LinearProgram::solve(std::span<const double> c, std::span<const double> a,
                                           std::span<const double> b, std::span<const double> lower,
                                           std::span<const double> upper, base::MemoryBudget& budget)
{
    const std::size_t n = c.size();
    const std::size_t m = b.size();
    if (a.size() != m * n || lower.size() != n || upper.size() != n)
        throw base::OptimisationError(
            base::concatenate({"linear program: expected A of ", std::to_string(m), " × ", std::to_string(n),
                               " values and ", std::to_string(n), " bounds, found ", std::to_string(a.size()),
                               " values and ", std::to_string(lower.size()), " and ", std::to_string(upper.size()),
                               " bounds; give A row by row and one lower and one upper bound per variable"}));
    double largest = 1.0;
    for (std::size_t j = 0; j < n; ++j)
    {
        if (!std::isfinite(lower[j]) || !std::isfinite(upper[j]) || lower[j] > upper[j])
            throw base::OptimisationError(
                base::concatenate({"linear program: expected finite bounds lower ≤ upper on variable ",
                                   std::to_string(j), ", found [", std::to_string(lower[j]), ", ",
                                   std::to_string(upper[j]), "]; give finite bounds, the lower one first"}));
        largest = std::max(largest, upper[j] - lower[j]);
    }
    for (const double value : a)
        largest = std::max(largest, std::abs(value));
    double largestCost = 0.0;
    for (const double value : c)
        largestCost = std::max(largestCost, std::abs(value));

    // Rows: A·y ≤ b − A·lower, then y ≤ upper − lower.
    const std::size_t rows = m + n;
    std::vector<double> rhs(rows);
    for (std::size_t i = 0; i < m; ++i)
    {
        double shifted = b[i];
        for (std::size_t j = 0; j < n; ++j)
            shifted -= a[i * n + j] * lower[j];
        rhs[i] = shifted;
        largest = std::max(largest, std::abs(shifted));
    }
    for (std::size_t k = 0; k < n; ++k)
        rhs[m + k] = upper[k] - lower[k];
    const auto artificialCount =
        static_cast<std::size_t>(std::count_if(rhs.begin(), rhs.end(), [](double value) { return value < 0.0; }));
    const std::size_t structural = n + rows; // y and the slacks: the columns that may enter
    Tableau tableau(budget, rows, structural + artificialCount);
    std::size_t artificial = structural;
    for (std::size_t i = 0; i < rows; ++i)
    {
        const double sign = rhs[i] < 0.0 ? -1.0 : 1.0;
        if (i < m)
            for (std::size_t j = 0; j < n; ++j)
                tableau.at(i, j) = sign * a[i * n + j];
        else
            tableau.at(i, i - m) = sign;
        tableau.at(i, n + i) = sign;
        tableau.rhs(i) = sign * rhs[i];
        if (sign < 0.0)
        {
            tableau.at(i, artificial) = 1.0;
            tableau.basis()[i] = artificial++;
        }
        else
            tableau.basis()[i] = n + i;
    }

    const double pivotTolerance = PIVOT_TOLERANCE * largest;
    const std::size_t pivotLimit = PIVOT_LIMIT_PER_COLUMN * (rows + tableau.columns());
    LinearProgramSolution solution;
    bool isUnbounded = false;
    const auto run = [&](double costTolerance)
    {
        while (tableau.step(structural, costTolerance, pivotTolerance, isUnbounded))
            if (++solution.pivots > pivotLimit)
                throw base::OptimisationError(
                    base::concatenate({"linear program: no optimum after ", std::to_string(pivotLimit),
                                       " pivots (round-off cycling on a problem of ", std::to_string(m), " rows and ",
                                       std::to_string(n), " variables); scale the problem"}));
    };

    // Phase 1: minimise the sum of the artificial variables; the cost row is 1 on them, reduced by their rows.
    if (artificialCount > 0)
    {
        for (std::size_t j = structural; j < tableau.columns(); ++j)
            tableau.cost(j) = 1.0;
        for (std::size_t i = 0; i < rows; ++i)
            if (tableau.basis()[i] >= structural)
                for (std::size_t j = 0; j <= tableau.columns(); ++j)
                    tableau.at(rows, j) -= tableau.at(i, j);
        run(COST_TOLERANCE);
        if (-tableau.rhs(rows) > FEASIBILITY_TOLERANCE * largest)
        {
            solution.status = LinearProgramStatus::Infeasible;
            return solution;
        }
        // The artificial variables left in the basis are at zero: pivot them out where a structural column allows it
        // (a row with none is redundant and stays as it is, its artificial variable at zero).
        for (std::size_t i = 0; i < rows; ++i)
        {
            if (tableau.basis()[i] < structural)
                continue;
            for (std::size_t j = 0; j < structural; ++j)
                if (std::abs(tableau.at(i, j)) > pivotTolerance)
                {
                    tableau.pivot(i, j);
                    break;
                }
        }
    }

    // Phase 2: the cost row of cᵀy, reduced by the basic rows.
    for (std::size_t j = 0; j <= tableau.columns(); ++j)
        tableau.cost(j) = j < n ? c[j] : 0.0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        const std::size_t basic = tableau.basis()[i];
        if (basic < n && c[basic] != 0.0)
        {
            const double weight = c[basic];
            for (std::size_t j = 0; j <= tableau.columns(); ++j)
                tableau.at(rows, j) -= weight * tableau.at(i, j);
        }
    }
    run(COST_TOLERANCE * (1.0 + largestCost));
    if (isUnbounded)
    {
        solution.status = LinearProgramStatus::Unbounded;
        return solution;
    }

    solution.x.assign(lower.begin(), lower.end());
    for (std::size_t i = 0; i < rows; ++i)
        if (tableau.basis()[i] < n)
            solution.x[tableau.basis()[i]] = lower[tableau.basis()[i]] + std::max(tableau.rhs(i), 0.0);
    for (std::size_t j = 0; j < n; ++j)
    {
        solution.x[j] = std::clamp(solution.x[j], lower[j], upper[j]);
        solution.objective += c[j] * solution.x[j];
    }
    solution.multipliers.resize(m);
    for (std::size_t i = 0; i < m; ++i)
        solution.multipliers[i] = std::max(tableau.cost(n + i), 0.0);
    return solution;
}

} // namespace oos::design
