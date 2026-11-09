#include "oos/solutions/common/FactorisedSet.hpp"

#include <stdexcept>
#include <utility>

namespace oos::solutions
{

FactorisedSet::FactorisedSet(FactorisedSetParts parts) : m_parts(std::move(parts))
{
    if (!m_parts.index || !m_parts.dofMap || !m_parts.gm || !m_parts.solveSets || !m_parts.pattern || !m_parts.kgg ||
        !m_parts.solver)
        throw std::invalid_argument(
            "FactorisedSet: expected every part of a factorised constraint set (index, DOF map, "
            "Gm, sets, pattern, Kgg, factor), found one missing");
}

} // namespace oos::solutions
