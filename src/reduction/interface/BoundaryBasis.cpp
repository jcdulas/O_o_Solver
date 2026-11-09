#include "oos/reduction/interface/BoundaryBasis.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

bool hasNonzero(const BasisCoefficients& coefficients, std::size_t row)
{
    for (std::size_t c = 0; c < coefficients.caseCount(); ++c)
        if (coefficients(row, c) != 0.0)
            return true;
    return false;
}

} // namespace

BoundaryBasis::BoundaryBasis(const InterfaceProblem& interface, std::span<const BoundaryUnitPatterns> units,
                             std::span<const BoundaryLocalLoads> blocks, std::size_t caseCount,
                             const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                             const base::CancellationToken* cancellation, const BoundarySolveOptions& solve)
    : m_interface(&interface)
{
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    const auto equations = static_cast<std::size_t>(interface.equationCount());
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        const BoundaryUnitPatterns& unit = units[u];
        if (unit.coefficients == nullptr || unit.patterns.rows != interface.unitLocalDofs(u).size() ||
            unit.patterns.columns != unit.coefficients->size() || unit.coefficients->caseCount() != caseCount)
            throw std::invalid_argument(
                base::concatenate({"BoundaryBasis: unit ", std::to_string(u), " gives ",
                                   std::to_string(unit.patterns.rows), " x ", std::to_string(unit.patterns.columns),
                                   " patterns for ", std::to_string(interface.unitLocalDofs(u).size()),
                                   " boundary DOFs; give one row per row of its K*, one coefficient row per pattern, ",
                                   std::to_string(caseCount), " cases"}));
    }
    bool hasEnforced = false;
    for (std::size_t b = 0; b < blocks.size(); ++b)
    {
        const BoundaryLocalLoads& block = blocks[b];
        const bool isEnforcedGiven = block.enforced.columns > 0;
        if (block.coefficients == nullptr || block.loads.rows != localDofs ||
            block.loads.columns != block.coefficients->size() || block.coefficients->caseCount() != caseCount ||
            (isEnforcedGiven && (block.enforced.rows != localDofs || block.enforced.columns != block.loads.columns)))
            throw std::invalid_argument(base::concatenate(
                {"BoundaryBasis: block ", std::to_string(b), " gives ", std::to_string(block.loads.rows), " x ",
                 std::to_string(block.loads.columns), " loads for ", std::to_string(localDofs),
                 " local DOFs; give one row per local DOF, one coefficient row per vector, ", std::to_string(caseCount),
                 " cases"}));
        hasEnforced = hasEnforced || isEnforcedGiven;
    }

    // The columns: the patterns and vectors some case takes.
    for (std::size_t u = 0; u < units.size(); ++u)
        for (std::size_t k = 0; k < units[u].patterns.columns; ++k)
            if (hasNonzero(*units[u].coefficients, k))
                m_columns.push_back({static_cast<std::int64_t>(u), -1, static_cast<std::int64_t>(k)});
    for (std::size_t b = 0; b < blocks.size(); ++b)
        for (std::size_t v = 0; v < blocks[b].loads.columns; ++v)
            if (hasNonzero(*blocks[b].coefficients, v))
                m_columns.push_back({-1, static_cast<std::int64_t>(b), static_cast<std::int64_t>(v)});
    m_distinctCount = m_columns.size();
    const std::size_t distinct = m_distinctCount;

    // Their loads on the local DOFs, their enforced values, and β.
    linalg::DenseMatrix loads(budget, localDofs, distinct, "boundary basis loads");
    linalg::DenseMatrix enforced(budget, localDofs, hasEnforced ? distinct : 0, "boundary basis enforced values");
    linalg::DenseMatrix beta(budget, distinct, caseCount, "boundary basis coefficients");
    for (std::size_t q = 0; q < distinct; ++q)
    {
        const BoundaryColumn& column = m_columns[q];
        const auto vector = static_cast<std::size_t>(column.vector);
        const BasisCoefficients* coefficients = nullptr;
        if (column.unit >= 0)
        {
            const BoundaryUnitPatterns& unit = units[static_cast<std::size_t>(column.unit)];
            const std::span<const std::int64_t> locals = interface.unitLocalDofs(static_cast<std::size_t>(column.unit));
            for (std::size_t i = 0; i < locals.size(); ++i)
                loads(static_cast<std::size_t>(locals[i]), q) += unit.patterns(i, vector);
            coefficients = unit.coefficients;
        }
        else
        {
            const BoundaryLocalLoads& block = blocks[static_cast<std::size_t>(column.block)];
            for (std::size_t i = 0; i < localDofs; ++i)
                loads(i, q) = block.loads(i, vector);
            if (block.enforced.columns > 0)
                for (std::size_t i = 0; i < localDofs; ++i)
                    enforced(i, q) = block.enforced(i, vector);
            coefficients = block.coefficients;
        }
        for (std::size_t c = 0; c < caseCount; ++c)
            beta(q, c) = (*coefficients)(vector, c);
    }

    std::size_t solved = distinct;
    if (distinct > caseCount || solve.solvesCases)
    {
        // The cases themselves: their right-hand sides are the combinations of the columns.
        const BasisCoefficients combination(std::move(beta));
        linalg::DenseMatrix caseLoads(budget, localDofs, caseCount, "boundary basis case loads");
        combination.combine(loads.view(), 0, caseLoads.view(), kernels);
        linalg::DenseMatrix caseEnforced(budget, localDofs, hasEnforced ? caseCount : 0,
                                         "boundary basis case enforced values");
        if (hasEnforced)
            combination.combine(enforced.view(), 0, caseEnforced.view(), kernels);
        loads = std::move(caseLoads);
        enforced = std::move(caseEnforced);
        beta = linalg::DenseMatrix(budget, caseCount, caseCount, "boundary basis coefficients");
        for (std::size_t c = 0; c < caseCount; ++c)
            beta(c, c) = 1.0;
        m_columns.clear();
        m_isCaseBasis = true;
        solved = caseCount;
    }
    m_coefficients = BasisCoefficients(std::move(beta));

    // The right-hand sides on the equations, the solves, the residuals and the local displacements.
    m_reducedLoads = std::move(loads);
    interface.reduceLoads(m_reducedLoads.view());
    m_rightHandSides = linalg::DenseMatrix(budget, equations, solved, "boundary basis right-hand sides");
    interface.equationLoads(m_reducedLoads.view(), enforced.view(), m_rightHandSides.view());
    m_enforced = std::move(enforced);
    if (solve.mode == BoundarySolveMode::RightHandSidesOnly)
        return;
    const auto refuseGiven = [&](std::size_t expectedRows, const char* rowName)
    {
        if (solve.given.rows != expectedRows || solve.given.columns != solved)
            throw std::invalid_argument(base::concatenate(
                {"BoundaryBasis: the given solutions are ", std::to_string(solve.given.rows), " x ",
                 std::to_string(solve.given.columns), "; expected ", std::to_string(expectedRows), " ", rowName, " by ",
                 std::to_string(solved), m_isCaseBasis ? " cases" : " columns of the boundary basis",
                 "; give the values of the interface this analysis assembles"}));
    };
    linalg::DenseMatrix solutions(budget, equations, solved, "boundary basis solutions");
    if (solve.mode == BoundarySolveMode::GivenLocal)
    {
        // Local displacements taken as they are: the equations' values read from them for the residuals.
        refuseGiven(localDofs, "local DOFs");
        m_displacements = linalg::DenseMatrix(budget, localDofs, solved, "boundary basis displacements");
        for (std::size_t q = 0; q < solved; ++q)
        {
            for (std::size_t i = 0; i < localDofs; ++i)
                m_displacements(i, q) = solve.given(i, q);
            for (std::size_t e = 0; e < equations; ++e)
                solutions(e, q) = solve.given(
                    static_cast<std::size_t>(interface.localDofOfEquation(static_cast<std::int32_t>(e))), q);
        }
    }
    else if (solve.mode == BoundarySolveMode::GivenEquations)
    {
        refuseGiven(equations, "equations");
        for (std::size_t q = 0; q < solved; ++q)
            for (std::size_t e = 0; e < equations; ++e)
                solutions(e, q) = solve.given(e, q);
    }
    else
    {
        for (std::size_t q = 0; q < solved; ++q)
            for (std::size_t e = 0; e < equations; ++e)
                solutions(e, q) = m_rightHandSides(e, q);
        interface.solve(solutions.view(), cancellation);
    }
    m_residuals = linalg::DenseMatrix(budget, equations, solved, "boundary basis residuals");
    interface.residuals(solutions.view(), m_rightHandSides.view(), m_residuals.view());
    if (solve.mode != BoundarySolveMode::GivenLocal)
    {
        m_displacements = linalg::DenseMatrix(budget, localDofs, solved, "boundary basis displacements");
        interface.expand(solutions.view(), m_enforced.view(), m_displacements.view());
    }
    if (solve.keepsEquationSolutions)
        m_equationSolutions = std::move(solutions);
}

UnitBasisRows BoundaryBasis::unitRows(std::size_t unit, base::MemoryBudget& budget) const
{
    const std::span<const std::int64_t> locals = m_interface->unitLocalDofs(unit);
    UnitBasisRows rows;
    rows.values = linalg::DenseMatrix(budget, locals.size(), columnCount(), "unit boundary basis rows");
    for (std::size_t q = 0; q < columnCount(); ++q)
        for (std::size_t i = 0; i < locals.size(); ++i)
            rows.values(i, q) = m_displacements(static_cast<std::size_t>(locals[i]), q);
    if (!m_isCaseBasis)
    {
        rows.patternOfColumn.assign(columnCount(), -1);
        for (std::size_t q = 0; q < m_columns.size(); ++q)
            if (m_columns[q].unit == static_cast<std::int64_t>(unit))
                rows.patternOfColumn[q] = m_columns[q].vector;
    }
    return rows;
}

} // namespace oos::reduction
