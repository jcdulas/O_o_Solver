#include "oos/input/requests/OutputRequests.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <utility>

namespace oos::input
{

namespace
{

constexpr base::NullPolicy ALLOW = base::NullPolicy::Allow;

// The section of caseSetsTable that holds the sets the output requests name (the others: plot, post, xyplot).
constexpr std::string_view CASE_SECTION = "case";
// The subcase number caseSetsTable gives the SETs of the global section.
constexpr std::int32_t GLOBAL_SECTION = 0;

} // namespace

OutputRequests::OutputRequests(base::ImportedTable requests, base::ImportedTable caseSets)
    : m_requests(readTable(TableRole::OutputRequests, std::move(requests))),
      m_caseSets(readTable(TableRole::CaseSets, std::move(caseSets)))
{
    m_names = m_requests.dictionaryTexts("request");
    m_setWords = m_requests.dictionaryTexts("set", ALLOW);
    m_hasSetWord = m_requests.validity("set");
    m_hasSetId = m_requests.validity("setId");
    m_describers = m_requests.textLists("describers", ALLOW);
    for (std::int64_t row = 0; row < m_requests.rowCount(); ++row)
        (void)scopeAt(row);
    const std::span<const std::int32_t> subcases = m_requests.values<std::int32_t>("subcase");
    m_rowsBySubcase.resize(static_cast<std::size_t>(m_requests.rowCount()));
    for (std::size_t row = 0; row < m_rowsBySubcase.size(); ++row)
        m_rowsBySubcase[row] = static_cast<std::int64_t>(row);
    std::stable_sort(m_rowsBySubcase.begin(), m_rowsBySubcase.end(), [&](std::int64_t a, std::int64_t b)
                     { return subcases[static_cast<std::size_t>(a)] < subcases[static_cast<std::size_t>(b)]; });
}

OutputRequest::Scope OutputRequests::scopeAt(std::int64_t row) const
{
    // In an output request, a number names a set (setId), All / None the words; ModalEffectiveMassOutput takes Yes and
    // No. A request written without a selection applies to everything, as All.
    if (m_hasSetId.isValid(row))
        return OutputRequest::Scope::Set;
    const std::string_view word = m_hasSetWord.isValid(row) ? m_setWords[row] : std::string_view();
    if (word.empty() || word == "All" || word == "Yes")
        return OutputRequest::Scope::All;
    if (word == "None" || word == "No")
        return OutputRequest::Scope::None;
    throw base::ContractError(base::concatenate(
        {"outputRequestsTable, row ", std::to_string(row),
         ": expected the selection All, None, Yes, No or a SetDefinition ID, "
         "found '",
         word,
         "'; the table does not follow the data contract of O_o_Reader (outputRequestsTable): take the snapshot "
         "again with a O_o_Reader that follows it"}));
}

std::vector<OutputRequest> OutputRequests::requestsOf(std::int32_t subcase) const
{
    const std::span<const std::int32_t> subcases = m_requests.values<std::int32_t>("subcase");
    const std::span<const std::int32_t> setIds = m_requests.values<std::int32_t>("setId", ALLOW);
    const base::ValidityView hasDescribers = m_requests.validity("describers");
    const auto first =
        std::lower_bound(m_rowsBySubcase.begin(), m_rowsBySubcase.end(), subcase, [&](std::int64_t row, std::int32_t id)
                         { return subcases[static_cast<std::size_t>(row)] < id; });
    std::vector<OutputRequest> result;
    for (auto found = first; found != m_rowsBySubcase.end() && subcases[static_cast<std::size_t>(*found)] == subcase;
         ++found)
    {
        const std::int64_t row = *found;
        const auto index = static_cast<std::size_t>(row);
        OutputRequest request;
        request.request = std::string(m_names[row]);
        if (hasDescribers.isValid(row))
        {
            for (std::int64_t item = 0; item < m_describers.count(row); ++item)
                request.describers.emplace_back(m_describers.at(row, item));
        }
        request.scope = scopeAt(row);
        request.setId = request.scope == OutputRequest::Scope::Set ? setIds[index] : 0;
        result.push_back(std::move(request));
    }
    return result;
}

std::optional<std::span<const std::int32_t>> OutputRequests::setMembers(std::int32_t setId, std::int32_t subcase) const
{
    const std::span<const std::int32_t> setIds = m_caseSets.values<std::int32_t>("setId");
    const std::span<const std::int32_t> subcases = m_caseSets.values<std::int32_t>("subcase");
    const std::span<const std::int32_t> ids = m_caseSets.values<std::int32_t>("id", ALLOW);
    const base::ValidityView hasId = m_caseSets.validity("id");
    const base::DictionaryView sections = m_caseSets.dictionaryTexts("section");
    const std::int64_t rows = m_caseSets.rowCount();
    std::optional<std::span<const std::int32_t>> global;
    std::int64_t row = 0;
    while (row < rows)
    {
        const auto index = static_cast<std::size_t>(row);
        // The rows of one SET are consecutive: [row, end) shares its subcase, set ID and section.
        std::int64_t end = row + 1;
        while (end < rows && setIds[static_cast<std::size_t>(end)] == setIds[index] &&
               subcases[static_cast<std::size_t>(end)] == subcases[index] && sections[end] == sections[row])
            ++end;
        bool isIdSet = setIds[index] == setId && sections[row] == CASE_SECTION;
        for (std::int64_t member = row; isIdSet && member < end; ++member)
            isIdSet = hasId.isValid(member);
        if (isIdSet)
        {
            const std::span<const std::int32_t> members = ids.subspan(index, static_cast<std::size_t>(end - row));
            if (subcases[index] == subcase)
                return members;
            if (subcases[index] == GLOBAL_SECTION)
                global = members;
        }
        row = end;
    }
    return global;
}

} // namespace oos::input
