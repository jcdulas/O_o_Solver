#include "oos/reduction/patterns/UnitLoadBasis.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/units/StampHasher.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

namespace
{

constexpr std::int64_t NOT_IN_UNIT = -1;

// The compact rows of a family that load the unit, renumbered in their order.
struct UnitRows
{
    base::BudgetedBuffer<std::int64_t> unitRowOfModelRow; // NOT_IN_UNIT for the others
    base::BudgetedBuffer<std::int64_t> modelRowOfUnitRow;
    std::int64_t count = 0;
};

UnitRows unitRowsOf(const UnitFamilyLoads& loads, base::MemoryBudget& budget)
{
    const std::int64_t rowCount = loads.isCombinationOfSets ? loads.loadSets->rowCount() : loads.cases->rowCount();
    UnitRows rows;
    rows.unitRowOfModelRow =
        base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(rowCount), "unit load rows");
    for (std::int64_t row = 0; row < rowCount; ++row)
    {
        const bool isUnitRow = loads.isUnitRow.empty() || loads.isUnitRow[static_cast<std::size_t>(row)] != 0;
        rows.unitRowOfModelRow[static_cast<std::size_t>(row)] = isUnitRow ? rows.count++ : NOT_IN_UNIT;
    }
    rows.modelRowOfUnitRow =
        base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(rows.count), "unit load rows");
    for (std::int64_t row = 0; row < rowCount; ++row)
        if (const std::int64_t unitRow = rows.unitRowOfModelRow[static_cast<std::size_t>(row)]; unitRow >= 0)
            rows.modelRowOfUnitRow[static_cast<std::size_t>(unitRow)] = row;
    return rows;
}

// The matrix restricted to the unit's rows, the columns as they are.
assembly::CompactLoadMatrix restrictedTo(const assembly::CompactLoadMatrix& matrix, const UnitRows& rows,
                                         base::MemoryBudget& budget)
{
    std::int64_t nonzeros = 0;
    std::size_t largest = 0;
    for (std::int64_t column = 0; column < matrix.columnCount(); ++column)
    {
        std::size_t count = 0;
        for (const std::int64_t row : matrix.rowsOf(column))
            count += rows.unitRowOfModelRow[static_cast<std::size_t>(row)] >= 0 ? 1 : 0;
        nonzeros += static_cast<std::int64_t>(count);
        largest = std::max(largest, count);
    }
    assembly::CompactLoadMatrix restricted(budget, rows.count, matrix.columnCount(), nonzeros);
    base::BudgetedBuffer<std::int64_t> columnRows(budget, largest, "unit load column");
    base::BudgetedBuffer<double> columnValues(budget, largest, "unit load column");
    base::BudgetedBuffer<double> columnPrecisions(budget, largest, "unit load column");
    for (std::int64_t column = 0; column < matrix.columnCount(); ++column)
    {
        const std::span<const std::int64_t> modelRows = matrix.rowsOf(column);
        const std::span<const double> values = matrix.valuesOf(column);
        const std::span<const double> precisions = matrix.precisionsOf(column);
        std::size_t kept = 0;
        for (std::size_t k = 0; k < modelRows.size(); ++k)
        {
            const std::int64_t unitRow = rows.unitRowOfModelRow[static_cast<std::size_t>(modelRows[k])];
            if (unitRow < 0)
                continue;
            columnRows[kept] = unitRow;
            columnValues[kept] = values[k];
            columnPrecisions[kept] = precisions[k];
            ++kept;
        }
        restricted.appendColumn({columnRows.data(), kept}, {columnValues.data(), kept},
                                {columnPrecisions.data(), kept});
    }
    return restricted;
}

std::uint64_t bitsOf(double value) noexcept
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// The name of a compact row in the stamp: its DOF for a loaded-DOF family, its row otherwise (a card value or a load
// set: the rows of those families are numbered by the cards, not by the cases).
std::int64_t rowName(const UnitFamilyLoads& loads, std::int64_t modelRow)
{
    return loads.dofOfRow.empty() ? modelRow
                                  : static_cast<std::int64_t>(loads.dofOfRow[static_cast<std::size_t>(modelRow)]);
}

// Mixes the compact values of a column of the unit (its rows by name and its values' bits).
void mixColumn(StampHasher& hasher, const UnitFamilyLoads& loads, const assembly::CompactLoadMatrix& restricted,
               const UnitRows& rows, std::int64_t column)
{
    const std::span<const std::int64_t> unitRows = restricted.rowsOf(column);
    const std::span<const double> values = restricted.valuesOf(column);
    for (std::size_t k = 0; k < unitRows.size(); ++k)
    {
        hasher.mix(rowName(loads, rows.modelRowOfUnitRow[static_cast<std::size_t>(unitRows[k])]));
        hasher.mix(bitsOf(values[k]));
    }
}

void requireConsistent(const UnitFamilyLoads& loads, std::size_t caseCount)
{
    const std::string family(assembly::nameOf(loads.family));
    if (loads.isCombinationOfSets)
    {
        // The cases as combinations of the load sets: the sets, their factors and the rows' flags.
        if (loads.loadSets == nullptr || loads.setCoefficients == nullptr ||
            loads.setCoefficients->columns() != caseCount ||
            loads.setCoefficients->rows() != static_cast<std::size_t>(loads.loadSets->columnCount()))
            throw std::invalid_argument(base::concatenate(
                {"UnitLoadBasis: the ", family,
                 " family's cases are combinations of its load sets but the sets or their factors for ",
                 std::to_string(caseCount), " cases are missing; give both"}));
        const auto rowCount = static_cast<std::size_t>(loads.loadSets->rowCount());
        if ((!loads.isUnitRow.empty() && loads.isUnitRow.size() != rowCount) ||
            (loads.offersLoadedDofs && loads.dofOfRow.size() != rowCount))
            throw std::invalid_argument(
                base::concatenate({"UnitLoadBasis: the ", family, " family flags or names ", "rows other than its ",
                                   std::to_string(rowCount), " compact rows; give one per row"}));
        return;
    }
    if (loads.cases == nullptr)
        throw std::invalid_argument(
            base::concatenate({"UnitLoadBasis: the ", family, " family has no cases matrix; give one per family"}));
    if (static_cast<std::size_t>(loads.cases->columnCount()) != caseCount ||
        loads.cases->filledColumnCount() != loads.cases->columnCount())
        throw std::invalid_argument(base::concatenate(
            {"UnitLoadBasis: the ", family, " family has ", std::to_string(loads.cases->filledColumnCount()), " of ",
             std::to_string(loads.cases->columnCount()), " case columns filled for ", std::to_string(caseCount),
             " load cases; give one filled column per load case"}));
    const auto rowCount = static_cast<std::size_t>(loads.cases->rowCount());
    if (!loads.isUnitRow.empty() && loads.isUnitRow.size() != rowCount)
        throw std::invalid_argument(base::concatenate(
            {"UnitLoadBasis: the ", family, " family flags ", std::to_string(loads.isUnitRow.size()),
             " rows of the unit for ", std::to_string(rowCount), " compact rows; give one flag per row or none"}));
    if (loads.offersLoadedDofs && loads.dofOfRow.size() != rowCount)
        throw std::invalid_argument(
            base::concatenate({"UnitLoadBasis: the ", family, " family offers the loaded DOFs but names ",
                               std::to_string(loads.dofOfRow.size()), " DOFs for ", std::to_string(rowCount),
                               " compact rows; give the DOF of every row"}));
    if ((loads.loadSets == nullptr) != (loads.setCoefficients == nullptr))
        throw std::invalid_argument(
            base::concatenate({"UnitLoadBasis: the ", family,
                               " family gives load sets without their factors or the reverse; give both "
                               "or neither"}));
    if (loads.loadSets != nullptr && loads.loadSets->rowCount() != loads.cases->rowCount())
        throw std::invalid_argument(base::concatenate({"UnitLoadBasis: the ", family,
                                                       " family's load sets and cases have different compact rows; "
                                                       "describe both on the same rows"}));
}

} // namespace

UnitLoadBasis UnitLoadBasis::choose(std::span<const UnitFamilyLoads> families, std::size_t caseCount,
                                    base::MemoryBudget& budget, base::ThreadPool* pool)
{
    for (std::size_t f = 0; f < families.size(); ++f)
    {
        requireConsistent(families[f], caseCount);
        for (std::size_t g = 0; g < f; ++g)
            if (families[g].family == families[f].family)
                throw std::invalid_argument(
                    base::concatenate({"UnitLoadBasis: the ", std::string(assembly::nameOf(families[f].family)),
                                       " family is given twice; give each family once"}));
    }

    UnitLoadBasis basis;
    StampHasher hasher;
    std::vector<linalg::DenseMatrix> familyCoefficients; // one per family given
    familyCoefficients.reserve(families.size());
    for (const UnitFamilyLoads& loads : families)
    {
        const UnitRows rows = unitRowsOf(loads, budget);
        const assembly::CompactLoadMatrix cases =
            loads.isCombinationOfSets ? assembly::CompactLoadMatrix{} : restrictedTo(*loads.cases, rows, budget);
        assembly::CompactLoadMatrix sets;
        if (loads.loadSets != nullptr)
            sets = restrictedTo(*loads.loadSets, rows, budget);

        assembly::LoadBasis::Input input;
        input.family = loads.family;
        input.cases = loads.isCombinationOfSets ? nullptr : &cases;
        input.isCombinationOfSets = loads.isCombinationOfSets;
        input.caseCount = caseCount;
        input.loadSets = loads.loadSets != nullptr ? &sets : nullptr;
        input.setCoefficients = loads.setCoefficients;
        input.offersLoadedDofs = loads.offersLoadedDofs;
        input.tolerance = loads.tolerance;
        input.pool = pool;
        const assembly::LoadBasis familyBasis = assembly::LoadBasis::choose(input, budget);
        if (familyBasis.kind() != assembly::LoadBasisKind::Empty)
            basis.m_reports.push_back(familyBasis.report());

        // A loaded-DOF basis in the order of its DOFs, whatever the order in which the cases first load them: its
        // patterns, their stamp and the columns of f* are then the same for any load values on the same DOFs.
        const std::span<const assembly::BasisVector> chosen = familyBasis.vectors();
        std::vector<std::size_t> order(chosen.size()); // bounded by the unit's load basis
        std::iota(order.begin(), order.end(), std::size_t{0});
        if (familyBasis.kind() == assembly::LoadBasisKind::LoadedDofs)
            std::sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b)
                      {
                          return rowName(loads, rows.modelRowOfUnitRow[static_cast<std::size_t>(chosen[a].index)]) <
                                 rowName(loads, rows.modelRowOfUnitRow[static_cast<std::size_t>(chosen[b].index)]);
                      });
        hasher.mix(static_cast<std::uint64_t>(loads.family));
        hasher.mix(static_cast<std::uint64_t>(familyBasis.kind()));
        for (const std::size_t k : order)
        {
            const assembly::BasisVector& vector = chosen[k];
            UnitBasisVector unitVector{loads.family, vector};
            hasher.mix(static_cast<std::uint64_t>(vector.source));
            switch (vector.source)
            {
            case assembly::BasisVector::Source::LoadedRow:
                // A unit load on the DOF of the row: the pattern does not depend on any value.
                unitVector.source.index = rows.modelRowOfUnitRow[static_cast<std::size_t>(vector.index)];
                hasher.mix(rowName(loads, unitVector.source.index));
                break;
            case assembly::BasisVector::Source::LoadSet:
                mixColumn(hasher, loads, sets, rows, vector.index);
                break;
            case assembly::BasisVector::Source::Case:
                mixColumn(hasher, loads, cases, rows, vector.index);
                break;
            }
            basis.m_vectors.push_back(unitVector);
        }
        const linalg::DenseMatrix& alpha = familyBasis.coefficients();
        linalg::DenseMatrix copy(budget, alpha.rows(), alpha.columns(), "unit load basis coefficients");
        for (std::size_t c = 0; c < alpha.columns(); ++c)
            for (std::size_t i = 0; i < alpha.rows(); ++i)
                copy(i, c) = alpha(order[i], c);
        familyCoefficients.push_back(std::move(copy));
    }

    linalg::DenseMatrix alpha(budget, basis.m_vectors.size(), caseCount, "unit load basis coefficients");
    std::size_t first = 0;
    for (const linalg::DenseMatrix& family : familyCoefficients)
    {
        for (std::size_t c = 0; c < family.columns(); ++c)
            for (std::size_t i = 0; i < family.rows(); ++i)
                alpha(first + i, c) = family(i, c);
        first += family.rows();
    }
    basis.m_coefficients = BasisCoefficients(std::move(alpha));
    basis.m_patternStamp = hasher.stamp();
    return basis;
}

} // namespace oos::reduction
