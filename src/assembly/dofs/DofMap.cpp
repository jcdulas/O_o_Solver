#include "oos/assembly/dofs/DofMap.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <string>

namespace oos::assembly
{

DofMap::DofMap(const input::NodeTable& nodes, const input::PointIndex& points)
    : m_nodes(&nodes),
      m_points(&points),
      m_gridCount(static_cast<std::int32_t>(nodes.rowCount())),
      m_scalarCount(static_cast<std::int32_t>(nodes.scalarPointIds().size())),
      m_gridIds(nodes.ids()),
      m_scalarIds(nodes.scalarPointIds()),
      m_positions(nodes.positionsInBasic())
{
    // The extent: the largest side of the grid points' box.
    if (m_gridCount > 0)
    {
        base::Vec3 low = positionInBasic(0);
        base::Vec3 high = low;
        for (std::int32_t row = 1; row < m_gridCount; ++row)
        {
            const base::Vec3 position = positionInBasic(row);
            for (std::size_t a = 0; a < 3; ++a)
            {
                low[a] = std::min(low[a], position[a]);
                high[a] = std::max(high[a], position[a]);
            }
        }
        for (std::size_t a = 0; a < 3; ++a)
            m_extent = std::max(m_extent, high[a] - low[a]);
    }
    const std::int64_t dofCount = 6 * nodes.rowCount() + static_cast<std::int64_t>(nodes.scalarPointIds().size());
    if (dofCount > base::MAX_DOF_COUNT)
        throw base::ContractError(
            base::concatenate({"the model has ", std::to_string(dofCount),
                               " degrees of freedom (6 per grid point, 1 per scalar point); "
                               "at most ",
                               std::to_string(base::MAX_DOF_COUNT), " are supported: split the model into parts"}));
    if (nodes.hasCdRotations())
        return;
    const std::span<const std::int32_t> displacementFrameIds = nodes.displacementFrameIds();
    for (std::size_t row = 0; row < displacementFrameIds.size(); ++row)
    {
        if (displacementFrameIds[row] != 0)
            throw base::ContractError(base::concatenate(
                {"grid point ", std::to_string(nodes.ids()[row]),
                 " has the displacement system Node.displacementFrameId ", std::to_string(displacementFrameIds[row]),
                 " but the snapshot holds no Node.displacementFrameId rotations; take the snapshot with "
                 "displacementRotations()"}));
    }
}

std::optional<base::DofIndex> DofMap::findDof(base::EntityId id, int component) const noexcept
{
    const std::optional<input::PointRef> point = m_points->find(id);
    if (!point)
        return std::nullopt;
    if (point->kind == input::PointKind::Scalar)
        return component == 0 ? std::optional<base::DofIndex>(scalarDof(point->row)) : std::nullopt;
    if (component < 1 || component > 6)
        return std::nullopt;
    return gridDof(point->row, component);
}

DofLabel DofMap::labelOf(base::DofIndex dof) const noexcept
{
    const base::DofIndex gridDofs = 6 * m_gridCount;
    if (dof >= gridDofs)
        return {input::PointKind::Scalar, dof - gridDofs, 0};
    return {input::PointKind::Grid, dof / 6, static_cast<std::uint8_t>(dof % 6 + 1)};
}

base::EntityId DofMap::pointIdOf(base::DofIndex dof) const
{
    const DofLabel label = labelOf(dof);
    return label.kind == input::PointKind::Scalar ? scalarId(label.row) : gridId(label.row);
}

linalg::NodeDof DofMap::nodeDofOf(base::DofIndex dof) const noexcept
{
    const DofLabel label = labelOf(dof);
    if (label.kind == input::PointKind::Scalar)
        return {m_gridCount + label.row, 0};
    return {label.row, label.component - 1};
}

base::BudgetedBuffer<std::uint8_t> DofMap::nodeDofCounts(base::MemoryBudget& budget) const
{
    base::BudgetedBuffer<std::uint8_t> counts(budget, static_cast<std::size_t>(nodeCount()), "node DOF counts");
    for (std::int32_t node = 0; node < nodeCount(); ++node)
        counts[static_cast<std::size_t>(node)] = node < m_gridCount ? 6 : 1;
    return counts;
}

base::Mat3 DofMap::cdToBasic(std::int32_t gridRow) const
{
    if (!m_nodes->hasCdRotations())
        return base::Mat3::identity();
    base::Mat3 rotation;
    const std::span<const double> values = m_nodes->cdRotations().subspan(9 * static_cast<std::size_t>(gridRow), 9);
    for (std::size_t k = 0; k < 9; ++k)
        rotation.values[k] = values[k];
    return rotation;
}

base::Vec3 DofMap::positionInBasic(std::int32_t gridRow) const
{
    const std::span<const double> xyz = m_positions[gridRow];
    return {xyz[0], xyz[1], xyz[2]};
}

} // namespace oos::assembly
