#include "oos/input/tables/MaterialTable.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

MaterialTable::MaterialTable(base::ImportedTable materials)
    : m_reader(readTable(TableRole::Materials, std::move(materials)))
{
    (void)ids();
}

} // namespace oos::input
