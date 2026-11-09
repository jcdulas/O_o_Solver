#include "oos/solutions/sol101/CondensedFactorisedSet.hpp"

#include <stdexcept>
#include <utility>

namespace oos::solutions
{

CondensedFactorisedSet::CondensedFactorisedSet(CondensedFactorisedSetParts parts) : m_parts(std::move(parts))
{
    if (!m_parts.index || !m_parts.dofMap || !m_parts.units || !m_parts.interface || !m_parts.gm)
        throw std::invalid_argument(
            "CondensedFactorisedSet: expected the index, the DOF map, the units, the interface problem and Gm of the "
            "condensed run; give every part");
    if (m_parts.isConstrained.size() != static_cast<std::size_t>(m_parts.dofMap->gSetSize()))
        throw std::invalid_argument("CondensedFactorisedSet: expected one constrained flag per g-set DOF");
}

} // namespace oos::solutions
