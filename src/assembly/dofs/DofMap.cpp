#include "oos/assembly/dofs/DofMap.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace oos::assembly
{

namespace
{

// The layout of a model without entities.
const elements::EntityLayout& noEntities()
{
    static const elements::EntityLayout empty;
    return empty;
}

} // namespace

DofMap::DofMap(const input::NodeTable& nodes, const input::PointIndex& points,
               const elements::FormulationModel* formulation, elements::EntityLayout::Kind layoutKind)
    : m_nodes(&nodes),
      m_points(&points),
      m_formulation(formulation),
      m_layoutKind(layoutKind),
      m_entities(formulation != nullptr ? &formulation->layout(layoutKind) : &noEntities()),
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
    const std::int64_t dofCount =
        6 * nodes.rowCount() + static_cast<std::int64_t>(nodes.scalarPointIds().size()) + m_entities->totalDofCount();
    if (dofCount > base::MAX_DOF_COUNT)
        throw base::ContractError(
            base::concatenate({"the model has ", std::to_string(dofCount),
                               " degrees of freedom (6 per grid point, 1 per scalar point, those of the "
                               "formulation's edges, faces and interiors); "
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
    if (dof >= firstEntityDof())
    {
        const std::int64_t offset = dof - firstEntityDof();
        const std::int32_t row = m_entities->rowOfDof(offset);
        return {input::PointKind::Entity, row, static_cast<std::uint8_t>(offset - m_entities->dofOffset(row))};
    }
    if (dof >= gridDofs)
        return {input::PointKind::Scalar, dof - gridDofs, 0};
    return {input::PointKind::Grid, dof / 6, static_cast<std::uint8_t>(dof % 6 + 1)};
}

base::EntityId DofMap::pointIdOf(base::DofIndex dof) const
{
    const DofLabel label = labelOf(dof);
    switch (label.kind)
    {
    case input::PointKind::Scalar:
        return scalarId(label.row);
    case input::PointKind::Entity:
        return 0;
    case input::PointKind::Grid:
        break;
    }
    return gridId(label.row);
}

std::string DofMap::describe(base::DofIndex dof) const
{
    const DofLabel label = labelOf(dof);
    switch (label.kind)
    {
    case input::PointKind::Scalar:
        return "scalar point " + std::to_string(scalarId(label.row));
    case input::PointKind::Entity:
        return "DOF " + std::to_string(label.component + 1) + " of " + m_entities->describe(label.row);
    case input::PointKind::Grid:
        break;
    }
    return "grid point " + std::to_string(gridId(label.row)) + " component " + std::to_string(label.component);
}

linalg::NodeDof DofMap::nodeDofOf(base::DofIndex dof) const noexcept
{
    const DofLabel label = labelOf(dof);
    return nodeDofOf(elements::ElementDof{label.kind, label.component, label.row});
}

linalg::NodeDof DofMap::nodeDofOf(const elements::ElementDof& dof) const noexcept
{
    switch (dof.kind)
    {
    case input::PointKind::Scalar:
        return {m_gridCount + dof.row, 0};
    case input::PointKind::Entity:
    {
        const int nodeDofs = elements::EntityLayout::ENTITY_NODE_DOFS;
        return {firstEntityNode() + m_entities->firstNode(dof.row) + dof.component / nodeDofs,
                dof.component % nodeDofs};
    }
    case input::PointKind::Grid:
        break;
    }
    return {dof.row, dof.component - 1};
}

base::BudgetedBuffer<std::uint8_t> DofMap::nodeDofCounts(base::MemoryBudget& budget) const
{
    base::BudgetedBuffer<std::uint8_t> counts(budget, static_cast<std::size_t>(nodeCount()), "node DOF counts");
    for (std::int32_t node = 0; node < firstEntityNode(); ++node)
        counts[static_cast<std::size_t>(node)] = node < m_gridCount ? 6 : 1;
    // Each entity in nodes of ENTITY_NODE_DOFS DOFs, the last one with the rest.
    const int nodeDofs = elements::EntityLayout::ENTITY_NODE_DOFS;
    for (std::int32_t row = 0; row < m_entities->count(); ++row)
    {
        int left = m_entities->dofCount(row);
        std::int32_t node = firstEntityNode() + m_entities->firstNode(row);
        while (left > 0)
        {
            const int here = left < nodeDofs ? left : nodeDofs;
            counts[static_cast<std::size_t>(node)] = static_cast<std::uint8_t>(here);
            left -= here;
            ++node;
        }
    }
    return counts;
}

namespace
{

// The entity of a node of the entities (counted from the first one): the last entity whose first node is at or
// before it.
std::int32_t entityOfNode(const elements::EntityLayout& entities, std::int32_t entityNode) noexcept
{
    std::int32_t low = 0;
    std::int32_t high = entities.count() - 1;
    while (low < high)
    {
        const std::int32_t middle = (low + high + 1) / 2;
        if (entities.firstNode(middle) <= entityNode)
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

} // namespace

base::EntityId DofMap::nodeIdOf(std::int32_t node) const noexcept
{
    if (node < m_gridCount)
        return gridId(node);
    if (node < firstEntityNode())
        return scalarId(node - m_gridCount);
    return entityNodeId(node - firstEntityNode());
}

std::optional<std::int32_t> DofMap::findNode(base::EntityId id) const noexcept
{
    if (id < 0)
    {
        const std::int32_t entityNode = -1 - id;
        if (entityNode >= m_entities->nodeCount())
            return std::nullopt;
        return firstEntityNode() + entityNode;
    }
    const std::optional<input::PointRef> point = m_points->find(id);
    if (!point)
        return std::nullopt;
    return point->kind == input::PointKind::Grid ? point->row : m_gridCount + point->row;
}

DofMap::NodeDofs DofMap::dofsOfNode(std::int32_t node) const noexcept
{
    if (node < m_gridCount)
        return {gridDof(node, 1), 6};
    if (node < firstEntityNode())
        return {scalarDof(node - m_gridCount), 1};
    const std::int32_t entityNode = node - firstEntityNode();
    const std::int32_t row = entityOfNode(*m_entities, entityNode);
    const int nodeDofs = elements::EntityLayout::ENTITY_NODE_DOFS;
    const int before = nodeDofs * (entityNode - m_entities->firstNode(row));
    const int left = m_entities->dofCount(row) - before;
    return {entityDof(row, before), left < nodeDofs ? left : nodeDofs};
}

void DofMap::addAffineEntityField(const base::Mat3& gradient, const base::Vec3& translation,
                                  std::span<double> valuesG) const
{
    if (m_formulation == nullptr || m_entities->empty())
        return;
    std::vector<double> field(static_cast<std::size_t>(m_entities->totalDofCount()), 0.0);
    m_formulation->affineField(m_layoutKind, gradient, translation, field);
    const std::span<double> entityValues = valuesG.subspan(static_cast<std::size_t>(firstEntityDof()), field.size());
    for (std::size_t i = 0; i < field.size(); ++i)
        entityValues[i] += field[i];
}

base::Vec3 DofMap::nodePosition(std::int32_t node) const
{
    if (node < m_gridCount)
        return positionInBasic(node);
    return m_entities->position(entityOfNode(*m_entities, node - firstEntityNode()));
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
