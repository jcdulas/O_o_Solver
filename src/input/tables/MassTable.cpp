#include "oos/input/tables/MassTable.hpp"

#include "oos/input/contract/TableContracts.hpp"

#include <utility>

namespace oos::input
{

MassTable::MassTable(base::ImportedTable nonStructuralMass)
    : m_reader(readTable(TableRole::NonStructuralMass, std::move(nonStructuralMass)))
{
}

} // namespace oos::input
