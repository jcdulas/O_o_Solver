#include "oos/input/tables/UnitTable.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

UnitTable::UnitTable(base::ImportedTable components, base::ImportedTable componentElements,
                     std::optional<base::ImportedTable> boundaries, std::optional<base::ImportedTable> loadGroups)
    : m_components(readTable(TableRole::Components, std::move(components))),
      m_componentElements(readTable(TableRole::ComponentElements, std::move(componentElements)))
{
    if (boundaries)
        m_boundaries.emplace(readTable(TableRole::Boundaries, std::move(*boundaries)));
    if (loadGroups)
        m_loadGroups.emplace(readTable(TableRole::LoadGroups, std::move(*loadGroups)));
    (void)componentIds();
    (void)revisions();
}

std::optional<std::int64_t> UnitTable::revisionOf(std::int64_t componentId) const
{
    const std::span<const std::int64_t> ids = componentIds();
    for (std::size_t row = 0; row < ids.size(); ++row)
    {
        if (ids[row] == componentId)
            return revisions()[row];
    }
    return std::nullopt;
}

const base::ContractReader& UnitTable::boundaries() const
{
    if (!m_boundaries)
        throw missingTableError(TableRole::Boundaries);
    return *m_boundaries;
}

const base::ContractReader& UnitTable::loadGroups() const
{
    if (!m_loadGroups)
        throw missingTableError(TableRole::LoadGroups);
    return *m_loadGroups;
}

} // namespace oos::input
