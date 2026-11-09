#include "oos/input/tables/ConstraintTable.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

ConstraintTable::ConstraintTable(base::ImportedTable spc, base::ImportedTable mpc)
    : m_spc(readTable(TableRole::Spc, std::move(spc))), m_mpc(readTable(TableRole::Mpc, std::move(mpc)))
{
}

} // namespace oos::input
