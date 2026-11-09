#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>

namespace oos::input
{

// The units of the model as O_o_Reader gives them: the connected components with their revision
// stamps (connectedComponents), the component of each element (connectedComponents(detail="elements")), and, when
// the snapshot was taken with them, the boundary nodes of each component and load group (componentBoundaries) and the
// load case groups with their load revision stamps (loadCaseGroups). A revision stamp is a 64-bit hash of everything
// a component's condensation depends on: two snapshots whose stamps are equal for a component have the same content
// for it (SnapshotDiff relies on this).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported tables (buffers borrowed).
class UnitTable
{
public:
    UnitTable(base::ImportedTable components, base::ImportedTable componentElements,
              std::optional<base::ImportedTable> boundaries, std::optional<base::ImportedTable> loadGroups);

    [[nodiscard]] std::int64_t componentCount() const noexcept { return m_components.rowCount(); }
    [[nodiscard]] std::span<const std::int64_t> componentIds() const
    {
        return m_components.values<std::int64_t>("componentId");
    }
    [[nodiscard]] std::span<const std::int64_t> revisions() const
    {
        return m_components.values<std::int64_t>("revision");
    }
    // The revision stamp of a component, or nothing when there is no such component.
    [[nodiscard]] std::optional<std::int64_t> revisionOf(std::int64_t componentId) const;

    [[nodiscard]] std::span<const std::int32_t> elementIds() const
    {
        return m_componentElements.values<std::int32_t>("elementId");
    }
    [[nodiscard]] std::span<const std::int64_t> elementComponents() const
    {
        return m_componentElements.values<std::int64_t>("componentId");
    }

    [[nodiscard]] bool hasBoundaries() const noexcept { return m_boundaries.has_value(); }
    // Throws ContractError when the snapshot was taken without the boundaries.
    [[nodiscard]] const base::ContractReader& boundaries() const;
    [[nodiscard]] bool hasLoadGroups() const noexcept { return m_loadGroups.has_value(); }
    // Throws ContractError when the snapshot was taken without the load case groups.
    [[nodiscard]] const base::ContractReader& loadGroups() const;

    [[nodiscard]] const base::ContractReader& components() const noexcept { return m_components; }
    [[nodiscard]] const base::ContractReader& componentElements() const noexcept { return m_componentElements; }

private:
    base::ContractReader m_components;
    base::ContractReader m_componentElements;
    std::optional<base::ContractReader> m_boundaries;
    std::optional<base::ContractReader> m_loadGroups;
};

} // namespace oos::input
