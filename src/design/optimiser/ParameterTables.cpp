#include "oos/design/optimiser/ParameterTables.hpp"

#include "../common/TableColumns.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

using namespace table_columns;

constexpr const char* INT64 = "int64";

} // namespace

base::TableContract designParametersContract()
{
    return {"designParametersTable",
            {text("name"), required("integerValue", INT64), required("realValue", FLOAT64), optionalText("textValue")}};
}

OptimisationParameters readDesignParameters(const base::ContractReader& table)
{
    const TextColumn names(table, "name");
    const NullableColumn<std::int64_t> integers(table, "integerValue");
    const NullableColumn<double> reals(table, "realValue");
    const TextColumn texts(table, "textValue");
    std::optional<base::ValidityView> textValidity;
    if (table.hasColumn("textValue"))
        textValidity.emplace(table.validity("textValue"));
    // O_o_Reader gives a row per parameter of the format's catalogue: is_default true for the catalogue's default,
    // and no value at all for a default the program computes (DABOBJ, OPTCOD). OptimisationParameters holds the
    // defaults itself: only the values a DesignOptimParams gives are read.
    std::optional<base::BooleanView> isDefault;
    if (table.hasColumn("isDefault"))
        isDefault.emplace(table.booleans("isDefault"));
    std::vector<std::pair<std::string, ParameterValue>> entries;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        if (isDefault && (*isDefault)[row])
            continue;
        const bool hasText = textValidity && textValidity->isValid(row);
        const int count = (integers[row] ? 1 : 0) + (reals[row] ? 1 : 0) + (hasText ? 1 : 0);
        if (count == 0)
            continue;
        if (count != 1)
            refuseRow(table, row,
                      base::concatenate({"expected one of integer_value, real_value, text_value for DesignOptimParams ",
                                         names[row], ", found ", std::to_string(count)}));
        if (integers[row])
            entries.emplace_back(names[row], *integers[row]);
        else if (reals[row])
            entries.emplace_back(names[row], *reals[row]);
        else
            entries.emplace_back(names[row], texts[row]);
    }
    return OptimisationParameters::fromEntries(entries);
}

} // namespace oos::design
