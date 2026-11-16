#include "oos/elements/core/EntityLayout.hpp"

#include <algorithm>
#include <stdexcept>

namespace oos::elements
{

std::int32_t EntityLayout::add(int dofCount, base::Vec3 position, EntityName name)
{
    if (dofCount < 1 || dofCount > MAX_ENTITY_DOFS)
        throw std::invalid_argument("EntityLayout: expected 1 to 255 DOFs for an entity, found " +
                                    std::to_string(dofCount));
    const auto row = static_cast<std::int32_t>(m_positions.size());
    const int nodes = (dofCount + ENTITY_NODE_DOFS - 1) / ENTITY_NODE_DOFS;
    m_dofOffsets.push_back(m_dofOffsets.back() + dofCount);
    m_nodeOffsets.push_back(m_nodeOffsets.back() + nodes);
    m_positions.push_back(position);
    m_names.push_back(name);
    return row;
}

std::int32_t EntityLayout::rowOfDof(std::int64_t offset) const noexcept
{
    // The last entity whose first DOF is at or before the offset.
    const auto after = std::upper_bound(m_dofOffsets.begin(), m_dofOffsets.end(), offset);
    return static_cast<std::int32_t>(after - m_dofOffsets.begin()) - 1;
}

std::string EntityLayout::describe(std::int32_t row) const
{
    const EntityName& entity = name(row);
    std::string grids;
    for (const std::int32_t id : entity.gridIds)
    {
        if (id == 0)
            continue;
        if (!grids.empty())
            grids.append(", ");
        grids.append(std::to_string(id));
    }
    switch (entity.kind)
    {
    case EntityKind::Edge:
        return "the edge between grid points " + grids;
    case EntityKind::Face:
        return "the face of grid points " + grids;
    case EntityKind::Interior:
        return "the interior of element " + std::to_string(entity.elementId);
    case EntityKind::LineEnd:
        return "the end at grid point " + grids + " of element " + std::to_string(entity.elementId);
    }
    return "an entity";
}

} // namespace oos::elements
