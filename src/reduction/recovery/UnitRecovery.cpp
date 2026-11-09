#include "oos/reduction/recovery/UnitRecovery.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace oos::reduction
{

namespace
{

constexpr std::int64_t NO_COLUMN = -1;

// c = a·b, each element one chain of terms in increasing order (0 − a·b, the sign changed: exact), as
// BasisCoefficients::combine.
void multiplyInto(linalg::ConstDenseMatrixView a, linalg::ConstDenseMatrixView b, linalg::DenseMatrixView c,
                  const linalg::DenseKernels& kernels)
{
    for (std::size_t j = 0; j < c.columns; ++j)
        for (std::size_t i = 0; i < c.rows; ++i)
            c(i, j) = 0.0;
    if (c.rows == 0 || c.columns == 0 || a.columns == 0)
        return;
    kernels.subtractProduct(c, a, b);
    for (std::size_t j = 0; j < c.columns; ++j)
        for (std::size_t i = 0; i < c.rows; ++i)
            c(i, j) = -c(i, j);
}

std::string unitText(const StaticCondensation& condensation)
{
    return base::concatenate({"UnitRecovery (unit ", std::to_string(condensation.system().unitId()), "): "});
}

} // namespace

const char* nameOf(RecoveryRegime regime) noexcept
{
    switch (regime)
    {
    case RecoveryRegime::BasisColumns:
        return "basis columns";
    case RecoveryRegime::LoadCases:
        return "load cases";
    case RecoveryRegime::BoundaryDofs:
        return "boundary DOFs";
    }
    return "";
}

UnitRecovery::UnitRecovery(const StaticCondensation& condensation, const UnitPatterns& patterns,
                           const BasisCoefficients& alpha, const UnitBoundaryBasis& boundary,
                           const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                           const base::CancellationToken* cancellation)
    : m_condensation(&condensation), m_patterns(patterns), m_alpha(&alpha), m_boundary(boundary)
{
    const UnitSystem& unit = condensation.system();
    const auto boundaryDofs = static_cast<std::size_t>(unit.boundaryDofCount());
    const auto interior = static_cast<std::size_t>(unit.interiorDofCount());
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto localDofs = static_cast<std::size_t>(unit.localDofCount());
    const std::size_t patternCount = alpha.size();
    const std::size_t cases = alpha.caseCount();
    const std::size_t columns = boundary.values.columns;
    const bool hasEnforced = patterns.enforced.columns > 0;
    if ((patternCount > 0 && (patterns.loads.rows != localDofs || patterns.loads.columns != patternCount)) ||
        (hasEnforced && (patterns.enforced.rows != localDofs || patterns.enforced.columns != patternCount)) ||
        patternCount != condensation.patternCount())
        throw std::invalid_argument(base::concatenate(
            {unitText(condensation), "patterns of ", std::to_string(patterns.loads.rows), " x ",
             std::to_string(patterns.loads.columns), " (enforced ", std::to_string(patterns.enforced.rows), " x ",
             std::to_string(patterns.enforced.columns), ") with ", std::to_string(patternCount),
             " coefficient rows; the condensation took ", std::to_string(condensation.patternCount()), " patterns on ",
             std::to_string(localDofs), " local DOFs: give the same patterns"}));
    if (boundary.values.rows != boundaryDofs || boundary.coefficients.rows != columns ||
        boundary.coefficients.columns != cases ||
        (!boundary.patternOfColumn.empty() && boundary.patternOfColumn.size() != columns))
        throw std::invalid_argument(base::concatenate(
            {unitText(condensation), "a boundary basis of ", std::to_string(boundary.values.rows), " x ",
             std::to_string(columns), " values and ", std::to_string(boundary.coefficients.rows), " x ",
             std::to_string(boundary.coefficients.columns), " coefficients (",
             std::to_string(boundary.patternOfColumn.size()), " pattern names) for ", std::to_string(boundaryDofs),
             " boundary DOFs and ", std::to_string(cases),
             " cases; give one value row per boundary DOF, one coefficient row per column, one column per case"}));

    // The patterns the basis columns carry: their coefficients must be the patterns' own (one right-hand side, one
    // load).
    std::vector<std::int64_t> columnOfPattern(patternCount, NO_COLUMN); // bounded by the unit's load basis
    for (std::size_t q = 0; q < boundary.patternOfColumn.size(); ++q)
    {
        const std::int64_t pattern = boundary.patternOfColumn[q];
        if (pattern == NO_COLUMN)
            continue;
        if (pattern < 0 || static_cast<std::size_t>(pattern) >= patternCount ||
            columnOfPattern[static_cast<std::size_t>(pattern)] != NO_COLUMN)
            throw std::invalid_argument(base::concatenate(
                {unitText(condensation), "basis column ", std::to_string(q), " names pattern ", std::to_string(pattern),
                 "; a column names one of the ", std::to_string(patternCount), " patterns, each at most once"}));
        columnOfPattern[static_cast<std::size_t>(pattern)] = static_cast<std::int64_t>(q);
        for (std::size_t c = 0; c < cases; ++c)
            if (boundary.coefficients(q, c) != alpha(static_cast<std::size_t>(pattern), c))
                throw std::invalid_argument(base::concatenate(
                    {unitText(condensation), "basis column ", std::to_string(q), " names pattern ",
                     std::to_string(pattern), " but its coefficient of case ", std::to_string(c),
                     " differs from the pattern's; a column carries a pattern's load only with its coefficients"}));
    }
    // The patterns that load some case (a pattern with zero coefficients contributes nothing).
    std::vector<std::size_t> loading; // bounded by the unit's load basis
    for (std::size_t p = 0; p < patternCount; ++p)
    {
        bool isUsed = false;
        for (std::size_t c = 0; c < cases && !isUsed; ++c)
            isUsed = alpha(p, c) != 0.0;
        if (isUsed)
            loading.push_back(p);
    }
    std::vector<std::size_t> uncovered;
    for (const std::size_t p : loading)
        if (columnOfPattern[p] == NO_COLUMN)
            uncovered.push_back(p);

    m_statistics.boundaryDofCount = static_cast<std::int64_t>(boundaryDofs);
    m_statistics.caseCount = static_cast<std::int64_t>(cases);
    m_statistics.basisColumnCount = static_cast<std::int64_t>(columns);
    m_statistics.patternCount = static_cast<std::int64_t>(patternCount);
    const std::size_t basisCost = columns + uncovered.size();
    const std::size_t caseCost = cases;
    const std::size_t boundaryCost = boundaryDofs + loading.size();
    if (basisCost <= caseCost && basisCost <= boundaryCost)
    {
        m_statistics.regime = RecoveryRegime::BasisColumns;
        m_statistics.boundarySolveCount = static_cast<std::int64_t>(columns);
        m_statistics.patternSolveCount = static_cast<std::int64_t>(uncovered.size());
    }
    else if (caseCost <= boundaryCost)
    {
        m_statistics.regime = RecoveryRegime::LoadCases;
        m_statistics.boundarySolveCount = static_cast<std::int64_t>(cases);
    }
    else
    {
        m_statistics.regime = RecoveryRegime::BoundaryDofs;
        m_statistics.boundarySolveCount = static_cast<std::int64_t>(boundaryDofs);
        m_statistics.patternSolveCount = static_cast<std::int64_t>(loading.size());
    }

    // The loads of the patterns on the equations (f_o,k: the interior rows; the boundary rows are not read).
    linalg::DenseMatrix patternLoads(budget, equations, patternCount, "unit recovery pattern loads");
    if (patternCount > 0)
        unit.equationLoads(patterns.loads, patterns.enforced, patternLoads.view(), budget);
    if (m_statistics.regime == RecoveryRegime::LoadCases)
    {
        m_patternLoads = std::move(patternLoads);
        return;
    }

    // The right-hand sides of the basis: interior loads and boundary values, solved together.
    const bool isBasis = m_statistics.regime == RecoveryRegime::BasisColumns;
    const std::size_t ownColumns = isBasis ? uncovered.size() : loading.size();
    const std::size_t boundaryColumns = isBasis ? columns : boundaryDofs;
    const std::size_t basis = boundaryColumns + ownColumns;
    // Basis order: BasisColumns, the Q columns then the uncovered patterns; BoundaryDofs, the patterns then the
    // boundary DOFs.
    const std::size_t firstPatternColumn = isBasis ? columns : 0;
    const std::size_t firstBoundaryColumn = isBasis ? 0 : ownColumns;
    m_solutions = linalg::DenseMatrix(budget, equations, basis, "unit recovery basis solutions");
    linalg::DenseMatrix trailing(budget, boundaryDofs, basis, "unit recovery basis boundary values");
    const std::vector<std::size_t>& ownPatterns = isBasis ? uncovered : loading;
    const auto copyPatternLoad = [&](std::size_t pattern, std::size_t column)
    {
        for (std::size_t i = 0; i < interior; ++i)
            m_solutions(i, column) = patternLoads(i, pattern);
    };
    for (std::size_t k = 0; k < ownPatterns.size(); ++k)
        copyPatternLoad(ownPatterns[k], firstPatternColumn + k);
    if (isBasis)
    {
        for (std::size_t q = 0; q < columns; ++q)
        {
            for (std::size_t i = 0; i < boundaryDofs; ++i)
                trailing(i, q) = boundary.values(i, q);
            const std::int64_t pattern = boundary.patternOfColumn.empty() ? NO_COLUMN : boundary.patternOfColumn[q];
            if (pattern != NO_COLUMN)
                copyPatternLoad(static_cast<std::size_t>(pattern), q);
        }
    }
    else
    {
        // H = −K_oo⁻¹·K_oa: a unit boundary displacement per column.
        for (std::size_t i = 0; i < boundaryDofs; ++i)
            trailing(i, firstBoundaryColumn + i) = 1.0;
    }
    condensation.recover(m_solutions.view(), trailing.view(), kernels, cancellation);

    // The enforced values of each basis column (the pattern's own; none for the boundary columns).
    if (hasEnforced)
    {
        m_enforced = linalg::DenseMatrix(budget, localDofs, basis, "unit recovery basis enforced values");
        const auto copyEnforced = [&](std::size_t pattern, std::size_t column)
        {
            for (std::size_t i = 0; i < localDofs; ++i)
                m_enforced(i, column) = patterns.enforced(i, pattern);
        };
        for (std::size_t k = 0; k < ownPatterns.size(); ++k)
            copyEnforced(ownPatterns[k], firstPatternColumn + k);
        if (isBasis)
            for (std::size_t q = 0; q < columns; ++q)
                if (!boundary.patternOfColumn.empty() && boundary.patternOfColumn[q] != NO_COLUMN)
                    copyEnforced(static_cast<std::size_t>(boundary.patternOfColumn[q]), q);
    }

    // The coefficients of every case on the basis: β and the α of the uncovered patterns, or α and u_a = V·β.
    linalg::DenseMatrix coefficients(budget, basis, cases, "unit recovery coefficients");
    for (std::size_t c = 0; c < cases; ++c)
        for (std::size_t k = 0; k < ownPatterns.size(); ++k)
            coefficients(firstPatternColumn + k, c) = alpha(ownPatterns[k], c);
    if (isBasis)
    {
        for (std::size_t c = 0; c < cases; ++c)
            for (std::size_t q = 0; q < columns; ++q)
                coefficients(q, c) = boundary.coefficients(q, c);
    }
    else
    {
        multiplyInto(boundary.values, boundary.coefficients,
                     coefficients.view().block(firstBoundaryColumn, 0, boundaryDofs, cases), kernels);
    }
    m_coefficients = BasisCoefficients(std::move(coefficients));
}

const BasisCoefficients& UnitRecovery::basisCoefficients() const
{
    if (!hasBasis())
        throw std::logic_error(base::concatenate({unitText(*m_condensation),
                                                  "the cases are solved one by one (regime 'load cases'): there is no "
                                                  "basis; recover the cases by batches"}));
    return m_coefficients;
}

linalg::DenseMatrix UnitRecovery::basisDisplacements(base::MemoryBudget& budget) const
{
    const UnitSystem& unit = m_condensation->system();
    const BasisCoefficients& coefficients = basisCoefficients();
    linalg::DenseMatrix displacements(budget, static_cast<std::size_t>(unit.localDofCount()), coefficients.size(),
                                      "unit recovery basis displacements");
    unit.expandSolution(m_solutions.view(), m_enforced.view(), displacements.view());
    return displacements;
}

void UnitRecovery::recoverCases(std::size_t firstCase, linalg::DenseMatrixView displacementsLocal,
                                const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                const base::CancellationToken* cancellation) const
{
    const UnitSystem& unit = m_condensation->system();
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto localDofs = static_cast<std::size_t>(unit.localDofCount());
    const std::size_t batch = displacementsLocal.columns;
    if (displacementsLocal.rows != localDofs || firstCase > caseCount() || batch > caseCount() - firstCase)
        throw std::invalid_argument(base::concatenate(
            {unitText(*m_condensation), "a batch of ", std::to_string(batch), " cases from case ",
             std::to_string(firstCase), " into ", std::to_string(displacementsLocal.rows), " rows; expected ",
             std::to_string(localDofs), " rows (the local DOFs) and a batch within the ", std::to_string(caseCount()),
             " cases"}));
    if (batch == 0)
        return;
    linalg::DenseMatrix solution(budget, equations, batch, "unit recovery batch");
    linalg::DenseMatrix enforced(budget, localDofs, hasEnforcedValues() ? batch : 0,
                                 "unit recovery batch enforced values");
    recoverEquations(firstCase, solution.view(), enforced.view(), kernels, budget, cancellation);
    unit.expandSolution(solution.view(), enforced.view(), displacementsLocal);
}

void UnitRecovery::recoverEquations(std::size_t firstCase, linalg::DenseMatrixView equationValues,
                                    linalg::DenseMatrixView enforcedLocal, const linalg::DenseKernels& kernels,
                                    base::MemoryBudget& budget, const base::CancellationToken* cancellation) const
{
    const UnitSystem& unit = m_condensation->system();
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto localDofs = static_cast<std::size_t>(unit.localDofCount());
    const auto boundaryDofs = static_cast<std::size_t>(unit.boundaryDofCount());
    const std::size_t batch = equationValues.columns;
    const bool hasEnforced = hasEnforcedValues();
    if (equationValues.rows != equations || firstCase > caseCount() || batch > caseCount() - firstCase ||
        (hasEnforced && (enforcedLocal.rows != localDofs || enforcedLocal.columns != batch)))
        throw std::invalid_argument(base::concatenate(
            {unitText(*m_condensation), "a batch of ", std::to_string(batch), " cases from case ",
             std::to_string(firstCase), " into ", std::to_string(equationValues.rows), " equation rows; expected ",
             std::to_string(equations), " rows (the equations), the enforced values on the ", std::to_string(localDofs),
             " local DOFs when the patterns enforce some, and a batch within the ", std::to_string(caseCount()),
             " cases"}));
    if (batch == 0)
        return;
    if (m_statistics.regime == RecoveryRegime::LoadCases)
    {
        // Each case's own right-hand side: Σ_k α_k·f_k on the interior, u_a = V·β on the boundary.
        m_alpha->combine(m_patternLoads.view(), firstCase, equationValues, kernels);
        linalg::DenseMatrix boundaryValues(budget, boundaryDofs, batch, "unit recovery batch boundary values");
        multiplyInto(m_boundary.values,
                     m_boundary.coefficients.block(0, firstCase, m_boundary.coefficients.rows, batch),
                     boundaryValues.view(), kernels);
        m_condensation->recover(equationValues, boundaryValues.view(), kernels, cancellation);
        if (hasEnforced)
            m_alpha->combine(m_patterns.enforced, firstCase, enforcedLocal, kernels);
    }
    else
    {
        m_coefficients.combine(m_solutions.view(), firstCase, equationValues, kernels);
        if (hasEnforced)
            m_coefficients.combine(m_enforced.view(), firstCase, enforcedLocal, kernels);
    }
}

} // namespace oos::reduction
