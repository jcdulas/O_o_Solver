#include "oos/input/tables/CardTable.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

CardTable::CardTable(std::string card, std::string idColumn, base::ImportedTable table)
    : m_card(std::move(card)),
      m_idColumn(std::move(idColumn)),
      m_reader(std::move(table), cardTableContract(m_card, m_idColumn))
{
    // An ID is never missing: refuse nulls now rather than at the first read.
    (void)ids();
}

} // namespace oos::input
