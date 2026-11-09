#include "oos/input/tables/ElementBatch.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

ElementBatch::ElementBatch(std::string card, base::ImportedTable table)
    : m_card(std::move(card)),
      m_idColumn(ELEMENT_ID_COLUMN),
      m_propertyColumn(ELEMENT_PROPERTY_COLUMN),
      m_reader(std::move(table), elementBatchContract(m_card))
{
    (void)ids();
}

std::span<const std::int32_t> ElementBatch::ids() const
{
    return m_reader.values<std::int32_t>(m_idColumn);
}

bool ElementBatch::hasProperty() const
{
    return m_reader.hasColumn(m_propertyColumn);
}

std::span<const std::int32_t> ElementBatch::propertyIds() const
{
    return m_reader.values<std::int32_t>(m_propertyColumn);
}

std::span<const std::int32_t> ElementBatch::propertyRows() const
{
    // A row is null when the property does not exist (O_o_Reader in tolerant mode): the caller reports it.
    return m_reader.values<std::int32_t>("propIdRow", base::NullPolicy::Allow);
}

base::DictionaryView ElementBatch::propertyTypes() const
{
    return m_reader.dictionaryTexts("propType", base::NullPolicy::Allow);
}

ConnectivityView ElementBatch::connectivity(const std::string& column) const
{
    // Nulls are allowed: an absent midside grid point (CQUAD8), or the row of a grid point that does not exist
    // (O_o_Reader in tolerant mode), is a problem of the model the element code reports as a diagnostic.
    const base::NullPolicy nulls = base::NullPolicy::Allow;
    if (m_reader.columnType(column).starts_with("fixed_list"))
        return ConnectivityView(m_reader.fixedLists<std::int32_t>(column, nulls));
    return ConnectivityView(m_reader.lists<std::int32_t>(column, nulls));
}

} // namespace oos::input
