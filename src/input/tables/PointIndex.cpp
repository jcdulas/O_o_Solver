#include "oos/input/tables/PointIndex.hpp"

namespace oos::input
{

PointIndex::PointIndex(const NodeTable& nodes, base::MemoryBudget& budget)
    : m_grids(nodes.ids(), budget, "grid point IDs"), m_scalars(nodes.scalarPointIds(), budget, "scalar point IDs")
{
}

std::optional<PointRef> PointIndex::find(std::int32_t id) const noexcept
{
    if (const std::optional<std::int32_t> row = m_grids.find(id))
        return PointRef{PointKind::Grid, *row};
    if (const std::optional<std::int32_t> row = m_scalars.find(id))
        return PointRef{PointKind::Scalar, *row};
    return std::nullopt;
}

} // namespace oos::input
