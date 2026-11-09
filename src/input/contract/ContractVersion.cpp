#include "oos/input/contract/ContractVersion.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <charconv>
#include <string>

namespace oos::input
{

bool isAcceptedContractVersion(std::int64_t version) noexcept
{
    return std::find(ACCEPTED_DATA_CONTRACT_VERSIONS.begin(), ACCEPTED_DATA_CONTRACT_VERSIONS.end(), version) !=
           ACCEPTED_DATA_CONTRACT_VERSIONS.end();
}

std::string acceptedContractVersionsText()
{
    std::string text = "{";
    for (const std::int64_t version : ACCEPTED_DATA_CONTRACT_VERSIONS)
        text.append(text.size() > 1 ? ", " : "").append(std::to_string(version));
    return text.append("}");
}

void requireAcceptedContractVersion(std::int64_t version, std::string_view where)
{
    // A contract newer than the last one checked is read: O_o_Reader's contract changes add far more than they change,
    // and the column contracts still refuse a table whose columns no longer fit.
    if (isAcceptedContractVersion(version) || version > ACCEPTED_DATA_CONTRACT_VERSIONS.back())
        return;
    const std::string accepted = acceptedContractVersionsText();
    throw base::ContractError(
        base::concatenate({where, ": expected a O_o_Reader data_contract_version in ", accepted,
                           " (the versions this O_o_Solver was "
                           "built and tested against; a later one is also read), found ",
                           std::to_string(version), "; install a O_o_Reader whose data_contract_version is in ",
                           accepted, ", or a O_o_Solver that reads version ", std::to_string(version)}));
}

std::int64_t requireAcceptedTableVersion(const base::ImportedTable& table, std::string_view tableName)
{
    const std::string text = table.metadataValue(std::string(DATA_CONTRACT_VERSION_KEY));
    if (text.empty())
        return 0;
    std::int64_t version = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), version);
    const std::string where = base::concatenate({"table '", tableName, "'"});
    if (error != std::errc() || end != text.data() + text.size())
        throw base::ContractError(base::concatenate(
            {where, ": expected an integer data_contract_version in the metadata key ", DATA_CONTRACT_VERSION_KEY,
             ", found '", text,
             "'; the table was not exported by a O_o_Reader this O_o_Solver reads: take the snapshot again with a "
             "compatible O_o_Reader"}));
    requireAcceptedContractVersion(version, where);
    return version;
}

} // namespace oos::input
