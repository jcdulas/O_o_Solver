#include "oos/input/tables/RigidTable.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

RigidTable::RigidTable(base::ImportedTable rigidElements)
    : m_reader(readTable(TableRole::RigidElements, std::move(rigidElements)))
{
    (void)ids();
}

} // namespace oos::input
