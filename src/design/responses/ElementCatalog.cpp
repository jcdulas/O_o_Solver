#include "oos/design/responses/ElementCatalog.hpp"

#include "oos/elements/core/ElementRegistry.hpp"

#include <algorithm>

namespace oos::design
{

namespace
{

bool tableHasId(const input::CardTable* table, std::int32_t id)
{
    if (table == nullptr)
        return false;
    const std::span<const std::int32_t> ids = table->ids();
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

} // namespace

ElementCatalog::ElementCatalog(const input::ModelSnapshot& snapshot) : m_snapshot(&snapshot)
{
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    const std::vector<input::ElementBatch>& batches = snapshot.elementBatches();
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const input::ElementBatch& batch = batches[b];
        const std::span<const std::int32_t> ids = batch.ids();
        const bool hasProperty = batch.hasProperty();
        const std::span<const std::int32_t> propIds =
            hasProperty ? batch.propertyIds() : std::span<const std::int32_t>{};
        const bool hasTypes = hasProperty && batch.hasPropertyRows() && batch.reader().hasColumn("propType");
        std::vector<std::string> candidates; // the property cards of the family, bounded by a family's list
        if (const elements::ElementFamily* family = registry.find(batch.card()))
            candidates = family->propertyCards();
        for (std::int64_t row = 0; row < batch.rowCount(); ++row)
        {
            const auto r = static_cast<std::size_t>(row);
            ElementEntry entry{ids[r], batch.card(), b, row, hasProperty ? propIds[r] : 0, {}};
            if (hasTypes)
            {
                entry.propertyCard = std::string(batch.propertyTypes()[row]);
            }
            else if (hasProperty)
            {
                for (const std::string& card : candidates)
                    if (tableHasId(snapshot.findPropertyTable(card), entry.propId))
                    {
                        entry.propertyCard = card;
                        break;
                    }
            }
            m_entries.push_back(std::move(entry));
        }
    }
    std::stable_sort(m_entries.begin(), m_entries.end(),
                     [](const ElementEntry& a, const ElementEntry& b) { return a.elementId < b.elementId; });
    for (std::size_t i = 0; i < m_entries.size(); ++i)
    {
        m_byId.emplace(m_entries[i].elementId, i);
        if (m_entries[i].propId != 0)
            m_byProperty[{m_entries[i].propertyCard, m_entries[i].propId}].push_back(i);
    }
}

const ElementEntry* ElementCatalog::find(std::int32_t elementId) const noexcept
{
    const auto found = m_byId.find(elementId);
    return found == m_byId.end() ? nullptr : &m_entries[found->second];
}

std::vector<const ElementEntry*> ElementCatalog::elementsOfProperty(std::string_view propertyCard,
                                                                    std::int32_t propId) const
{
    std::vector<const ElementEntry*> found;
    const auto entries = m_byProperty.find({std::string(propertyCard), propId});
    if (entries != m_byProperty.end())
        for (const std::size_t index : entries->second)
            found.push_back(&m_entries[index]);
    return found;
}

bool ElementCatalog::hasProperty(std::string_view propertyCard, std::int32_t propId) const
{
    return tableHasId(m_snapshot->findPropertyTable(propertyCard), propId);
}

} // namespace oos::design
