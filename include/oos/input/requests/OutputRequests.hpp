#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::input
{

// One output request of a subcase (DISPLACEMENT(SORT1,PRINT) = 10): the request, its describers as the deck writes
// them, and what it applies to.
struct OutputRequest
{
    enum class Scope
    {
        All,  // = ALL
        None, // = NONE: the request is switched off for this subcase
        Set   // = n: the members of Case Control SET n
    };
    std::string request;                 // the neutral command name: "DisplacementOutput", "StressOutput"
    std::vector<std::string> describers; // "SortByPoint", "PrintListing", ... in the order of the deck
    Scope scope = Scope::All;
    std::int32_t setId = 0; // for Scope::Set
};

// The output requests of the Case Control and their sets (O_o_Reader outputRequestsTable and caseSetsTable,
// data contract 31): the requests of a subcase and the members of a set; the solution drivers map them to result
// kinds.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported tables (buffers borrowed).
class OutputRequests
{
public:
    OutputRequests(base::ImportedTable requests, base::ImportedTable caseSets);

    // The effective requests of a subcase (the global requests already inherited by O_o_Reader), in table order.
    [[nodiscard]] std::vector<OutputRequest> requestsOf(std::int32_t subcase) const;

    // The members of Case Control SET `setId` as seen from `subcase`: the set defined in that subcase if there is
    // one, otherwise the set of the global section (subcase 0: a SET defined in a subcase is local to it);
    // nothing when no such set of IDs exists. The members are the `id` values of the rows of the set, which
    // caseSetsTable gives one after the other in the order of the set.
    [[nodiscard]] std::optional<std::span<const std::int32_t>> setMembers(std::int32_t setId,
                                                                          std::int32_t subcase) const;

    [[nodiscard]] const base::ContractReader& requestReader() const noexcept { return m_requests; }
    [[nodiscard]] const base::ContractReader& caseSetReader() const noexcept { return m_caseSets; }

private:
    [[nodiscard]] OutputRequest::Scope scopeAt(std::int64_t row) const;

    base::ContractReader m_requests;
    base::ContractReader m_caseSets;
    // The columns read once and the rows by subcase (a deck of thousands of subcases asks for the requests of each).
    base::DictionaryView m_names;
    base::DictionaryView m_setWords; // ALL, NONE, YES, NO; null for a SET or no selection
    base::ValidityView m_hasSetWord;
    base::ValidityView m_hasSetId;
    base::TextListView m_describers;
    std::vector<std::int64_t> m_rowsBySubcase; // the rows sorted by subcase, table order within; bounded by the rows
};

} // namespace oos::input
