#include "oos/input/tables/LoadTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace oos::input
{

namespace
{

// The tables that expand the combinations list a combination's rows under its SID with the set of each card in
// sourceSetId; the tables taken with expandCombinations=False list each set's own cards, setId = sourceSetId on every
// row. A table of the first form given with loadCombinationsTable would bring every combination twice.
void requireOwnRows(const base::ContractReader& table)
{
    const std::span<const std::int32_t> setIds = table.values<std::int32_t>("setId", base::NullPolicy::Allow);
    const std::span<const std::int32_t> sources = table.values<std::int32_t>("sourceSetId", base::NullPolicy::Allow);
    for (std::size_t row = 0; row < setIds.size(); ++row)
    {
        if (setIds[row] == sources[row])
            continue;
        throw base::ContractError(
            base::concatenate({"model snapshot: expected ", table.tableName(),
                               " to list each load set once, its own cards, with "
                               "loadCombinationsTable (O_o_Reader's expandCombinations=False), found row ",
                               std::to_string(row), " of set ", std::to_string(setIds[row]),
                               " whose card belongs to set ", std::to_string(sources[row]),
                               " (a combination expanded); take every load table with "
                               "expandCombinations=False, or none of them and no loadCombinationsTable"}));
    }
}

} // namespace

LoadTable::LoadTable(base::ImportedTable loads, base::ImportedTable pressures, base::ImportedTable lineLoads,
                     base::ImportedTable temperatures, std::optional<base::ImportedTable> inertialLoads,
                     std::optional<base::ImportedTable> combinations, bool isPressureFields)
    : m_isPressureFields(isPressureFields),
      m_loads(readTable(TableRole::Loads, std::move(loads))),
      m_pressures(readTable(isPressureFields ? TableRole::PressureFields : TableRole::Pressures, std::move(pressures))),
      m_lineLoads(readTable(TableRole::LineLoads, std::move(lineLoads))),
      m_temperatures(readTable(TableRole::Temperatures, std::move(temperatures)))
{
    if (inertialLoads)
        m_inertialLoads.emplace(readTable(TableRole::InertialLoads, std::move(*inertialLoads)));
    if (!combinations)
        return;
    m_combinations.emplace(readTable(TableRole::LoadCombinations, std::move(*combinations)));
    m_combinationLoads = m_combinations->values<std::int32_t>("load");
    m_combinationSets = m_combinations->values<std::int32_t>("loadSet");
    m_combinationFactors = m_combinations->values<double>("combinedFactor");
    // termsOf finds the rows of an entry by a binary search: O_o_Reader sorts the table by (load, loadSet).
    const auto unsorted = std::is_sorted_until(m_combinationLoads.begin(), m_combinationLoads.end());
    if (unsorted != m_combinationLoads.end())
        throw base::ContractError(base::concatenate(
            {"model snapshot: expected loadCombinationsTable sorted by load (O_o_Reader's default order), found "
             "load ",
             std::to_string(*unsorted), " after load ", std::to_string(*(unsorted - 1)), " at row ",
             std::to_string(unsorted - m_combinationLoads.begin()), "; take the table without sort="}));
    requireOwnRows(m_loads);
    if (m_inertialLoads)
        requireOwnRows(*m_inertialLoads);
}

const base::ContractReader& LoadTable::inertialLoads() const
{
    if (!m_inertialLoads)
        throw missingTableError(TableRole::InertialLoads);
    return *m_inertialLoads;
}

const base::ContractReader& LoadTable::combinations() const
{
    if (!m_combinations)
        throw missingTableError(TableRole::LoadCombinations);
    return *m_combinations;
}

bool LoadTable::isCombination(std::int32_t setId) const
{
    return std::binary_search(m_combinationLoads.begin(), m_combinationLoads.end(), setId);
}

std::vector<LoadTerm> LoadTable::termsOf(std::int32_t setId) const
{
    std::vector<LoadTerm> terms{LoadTerm{setId, 1.0}};
    const auto [first, last] = std::equal_range(m_combinationLoads.begin(), m_combinationLoads.end(), setId);
    for (auto it = first; it != last; ++it)
    {
        const auto row = static_cast<std::size_t>(it - m_combinationLoads.begin());
        terms.push_back({m_combinationSets[row], m_combinationFactors[row]});
    }
    return terms;
}

} // namespace oos::input
