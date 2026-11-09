#include "oos/input/tables/SubcaseTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace oos::input
{

namespace
{

constexpr std::array<std::string_view, 7> SELECTION_COLUMNS{
    "spc", "mpc", "load", "dynamicLoadSelection", "method", "temperatureLoad", "temperatureInitial"};

} // namespace

SubcaseTable::SubcaseTable(base::ImportedTable subcases) : m_reader(readTable(TableRole::Subcases, std::move(subcases)))
{
    (void)ids();
}

std::int64_t SubcaseTable::findRow(std::int32_t subcase) const
{
    // A handful of subcases (thousands at most): a linear search is faster than building an index.
    const std::span<const std::int32_t> subcaseIds = ids();
    for (std::size_t row = 0; row < subcaseIds.size(); ++row)
    {
        if (subcaseIds[row] == subcase)
            return static_cast<std::int64_t>(row);
    }
    return -1;
}

namespace
{

void requireSelectionColumn(const std::string& command)
{
    bool isSelection = false;
    for (const std::string_view column : SELECTION_COLUMNS)
        isSelection = isSelection || column == command;
    if (!isSelection)
        throw std::invalid_argument(base::concatenate(
            {"subcasesTable: expected a selection column (spc, mpc, load, dynamicLoadSelection, method, "
             "temperatureLoad, temperatureInitial), found '",
             command, "'"}));
}

} // namespace

std::optional<std::int32_t> SubcaseTable::selectedSet(std::int32_t subcase, const std::string& command) const
{
    requireSelectionColumn(command);
    const std::int64_t row = findRow(subcase);
    if (row < 0)
        throw std::invalid_argument(
            base::concatenate({"subcase ", std::to_string(subcase),
                               " is not in the Case Control of the model; give a subcase of the model"}));
    return selectedSetAtRow(row, command);
}

std::optional<std::int32_t> SubcaseTable::selectedSetAtRow(std::int64_t row, const std::string& command) const
{
    requireSelectionColumn(command);
    if (row < 0 || row >= rowCount())
        throw std::invalid_argument(base::concatenate(
            {"subcasesTable: expected a row in [0, ", std::to_string(rowCount()), "), found ", std::to_string(row)}));
    if (!m_reader.validity(command).isValid(row))
        return std::nullopt;
    return m_reader.values<std::int32_t>(command, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

std::optional<std::int32_t> SubcaseTable::contactTableAtRow(std::int64_t row) const
{
    if (row < 0 || row >= rowCount())
        throw std::invalid_argument(base::concatenate(
            {"subcasesTable: expected a row in [0, ", std::to_string(rowCount()), "), found ", std::to_string(row)}));
    if (!hasContactColumn() || !m_reader.validity("contactSelection").isValid(row))
        return std::nullopt;
    return m_reader.values<std::int32_t>("contactSelection", base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

std::optional<std::int32_t> SubcaseTable::deformationSetAtRow(std::int64_t row) const
{
    if (row < 0 || row >= rowCount())
        throw std::invalid_argument(base::concatenate(
            {"subcasesTable: expected a row in [0, ", std::to_string(rowCount()), "), found ", std::to_string(row)}));
    if (!m_reader.hasColumn("deform") || !m_reader.validity("deform").isValid(row))
        return std::nullopt;
    const auto index = static_cast<std::size_t>(row);
    if (m_reader.columnType("deform") == "int32")
        return m_reader.values<std::int32_t>("deform", base::NullPolicy::Allow)[index];
    const double set = m_reader.values<double>("deform", base::NullPolicy::Allow)[index];
    if (!std::isfinite(set))
        return std::nullopt;
    // A set ID is a positive integer; the added column carries it exactly as a double.
    if (set < 1.0 || set > static_cast<double>(std::numeric_limits<std::int32_t>::max()) || set != std::floor(set))
        throw base::ContractError(
            base::concatenate({"subcasesTable, row ", std::to_string(row),
                               ": expected a DeformationLoadSelection set ID (an integer > 0) in column "
                               "deform, found ",
                               std::to_string(set), "; take the snapshot again"}));
    return static_cast<std::int32_t>(set);
}

SubcaseTable::Combination SubcaseTable::combinationAtRow(std::int64_t row) const
{
    if (!m_reader.hasColumn("combinedSubcases") || !m_reader.hasColumn("coefficients"))
        return {};
    const base::ValidityView hasSubcases = m_reader.validity("combinedSubcases");
    const base::ValidityView hasCoefficients = m_reader.validity("coefficients");
    if (!hasSubcases.isValid(row) || !hasCoefficients.isValid(row))
        return {};
    Combination combination{m_reader.lists<std::int32_t>("combinedSubcases", base::NullPolicy::Allow)[row],
                            m_reader.lists<double>("coefficients", base::NullPolicy::Allow)[row]};
    if (combination.subcases.size() != combination.coefficients.size())
        throw base::ContractError(base::concatenate(
            {"subcasesTable, row ", std::to_string(row), ": expected as many coefficients as combined subcases, found ",
             std::to_string(combination.coefficients.size()), " for ", std::to_string(combination.subcases.size()),
             "; the table does not follow the data contract of O_o_Reader: "
             "take the snapshot again with a O_o_Reader that follows it"}));
    return combination;
}

} // namespace oos::input
