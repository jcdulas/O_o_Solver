#include "oos/reduction/condense/SymmetricBlockAccess.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

std::int32_t SymmetricBlockAccess::nodeOf(std::int64_t dof) const noexcept
{
    return static_cast<std::int32_t>(std::upper_bound(m_firstDof.begin(), m_firstDof.end(), dof) - m_firstDof.begin()) -
           1;
}

std::int64_t SymmetricBlockAccess::positionOf(std::int64_t i, std::int64_t j, bool& isStored) const
{
    const std::int32_t a = nodeOf(i);
    const std::int32_t b = nodeOf(j);
    isStored = a >= b;
    const std::int32_t rowNode = isStored ? a : b;
    const std::int32_t columnNode = isStored ? b : a;
    const std::int64_t rowComponent = (isStored ? i : j) - m_firstDof[static_cast<std::size_t>(rowNode)];
    const std::int64_t columnComponent = (isStored ? j : i) - m_firstDof[static_cast<std::size_t>(columnNode)];
    const std::int64_t block = m_pattern->findBlock(rowNode, columnNode);
    if (block < 0)
        return -1;
    return m_pattern->blockValueOffset(block) + rowComponent +
           static_cast<std::int64_t>(m_pattern->nodeDofCount(rowNode)) * columnComponent;
}

void SymmetricBlockAccess::requireCoupled(std::int64_t position, std::int64_t i, std::int64_t j)
{
    if (position < 0)
        throw std::logic_error(base::concatenate(
            {"SymmetricBlockAccess: the term (", std::to_string(i), ", ", std::to_string(j),
             ") is outside the pattern; the node graph must hold the couplings of the relations and of the spiders "
             "(an internal error: report it)"}));
}

double SymmetricBlockAccess::read(std::int64_t i, std::int64_t j) const
{
    bool isStored = false;
    const std::int64_t position = positionOf(i, j, isStored);
    return position < 0 ? 0.0 : m_values[static_cast<std::size_t>(position)];
}

void SymmetricBlockAccess::add(std::int64_t i, std::int64_t j, double value) const
{
    bool isStored = false;
    const std::int64_t position = positionOf(i, j, isStored);
    requireCoupled(position, i, j);
    if (isStored)
        m_values[static_cast<std::size_t>(position)] += value;
}

void SymmetricBlockAccess::setSymmetric(std::int64_t i, std::int64_t j, double value) const
{
    bool isStored = false;
    const std::int64_t position = positionOf(i, j, isStored);
    requireCoupled(position, i, j);
    m_values[static_cast<std::size_t>(position)] = value;
    const std::int64_t mirror = positionOf(j, i, isStored);
    m_values[static_cast<std::size_t>(mirror)] = value;
}

} // namespace oos::reduction
