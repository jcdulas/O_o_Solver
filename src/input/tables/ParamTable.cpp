#include "oos/input/tables/ParamTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <string>
#include <utility>

namespace oos::input
{

namespace
{

constexpr base::NullPolicy ALLOW = base::NullPolicy::Allow;

// The subcase number paramsTable gives the PARAMs of the global Case Control section.
constexpr std::int32_t GLOBAL_SECTION = 0;

// A PARAM written with no value (kind "blank"): the solver reads it as not given (the default applies).
constexpr std::string_view BLANK_KIND = "blank";

} // namespace

ParamTable::ParamTable(base::ImportedTable params) : m_reader(readTable(TableRole::Params, std::move(params)))
{
    // Every row must be readable: check the kinds once, so that find() never throws on a row.
    const base::DictionaryView kinds = m_reader.dictionaryTexts("kind");
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        if (kinds[row] != BLANK_KIND)
            (void)valueAt(row);
    }
}

ParamValue ParamTable::valueAt(std::int64_t row) const
{
    const auto index = static_cast<std::size_t>(row);
    const std::string_view kind = m_reader.dictionaryTexts("kind")[row];
    ParamValue value;
    if (kind == "integer")
    {
        value.kind = ParamValue::Kind::Integer;
        value.integer = m_reader.values<std::int32_t>("intValue", ALLOW)[index];
    }
    else if (kind == "real" || kind == "complex")
    {
        // A complex PARAM (two reals) is read by its real part: the parameters of the static, modal and
        // buckling solutions take no complex value.
        value.kind = ParamValue::Kind::Real;
        value.real = m_reader.values<double>("realValue", ALLOW)[index];
    }
    else if (kind == "text")
    {
        value.kind = ParamValue::Kind::Text;
        value.text = m_reader.texts("textValue", ALLOW)[row];
    }
    else
    {
        throw base::ContractError(
            base::concatenate({"paramsTable, row ", std::to_string(row),
                               ": expected the kind 'integer', 'real', 'text', 'complex' or 'blank', found '", kind,
                               "'; the table does not follow the data contract of O_o_Reader (paramsTable): take "
                               "the snapshot again with a O_o_Reader that follows it"}));
    }
    return value;
}

std::optional<ParamValue> ParamTable::find(std::string_view name, std::optional<std::int32_t> subcase) const
{
    const base::DictionaryView names = m_reader.dictionaryTexts("name");
    const base::DictionaryView kinds = m_reader.dictionaryTexts("kind");
    const base::ValidityView hasSubcase = m_reader.validity("subcase");
    const std::span<const std::int32_t> subcases = m_reader.values<std::int32_t>("subcase", ALLOW);
    // A PARAM of the subcase's Case Control wins over one of the global Case Control section, which wins over
    // the Bulk Data one.
    std::int64_t bulkRow = -1;
    std::int64_t globalRow = -1;
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        if (names[row] != name || kinds[row] == BLANK_KIND)
            continue;
        const std::int32_t section = subcases[static_cast<std::size_t>(row)];
        if (!hasSubcase.isValid(row))
            bulkRow = row;
        else if (subcase && section == *subcase && section != GLOBAL_SECTION)
            return valueAt(row);
        else if (section == GLOBAL_SECTION)
            globalRow = row;
    }
    if (globalRow >= 0)
        return valueAt(globalRow);
    if (bulkRow < 0)
        return std::nullopt;
    return valueAt(bulkRow);
}

std::optional<std::int32_t> ParamTable::findSubcaseGiving(std::string_view name) const
{
    const base::DictionaryView names = m_reader.dictionaryTexts("name");
    const base::DictionaryView kinds = m_reader.dictionaryTexts("kind");
    const base::ValidityView hasSubcase = m_reader.validity("subcase");
    const std::span<const std::int32_t> subcases = m_reader.values<std::int32_t>("subcase", ALLOW);
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        const std::int32_t section = subcases[static_cast<std::size_t>(row)];
        if (names[row] == name && kinds[row] != BLANK_KIND && hasSubcase.isValid(row) && section != GLOBAL_SECTION)
            return section;
    }
    return std::nullopt;
}

} // namespace oos::input
