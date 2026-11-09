#include "oos/elements/core/SnapshotIndex.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::elements
{

namespace
{

input::IdIndex materialIndex(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget)
{
    if (!snapshot.hasTable(input::TableRole::Materials))
        return {};
    return {snapshot.materials().ids(), budget, "material IDs"};
}

input::IdIndex coordinateSystemIndex(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget)
{
    if (!snapshot.hasTable(input::TableRole::CoordinateSystems))
        return {};
    return {snapshot.coordinateSystems().ids(), budget, "coordinate system IDs"};
}

std::size_t materialCount(const input::ModelSnapshot& snapshot)
{
    return snapshot.hasTable(input::TableRole::Materials) ? snapshot.materials().ids().size() : 0;
}

std::size_t coordinateSystemCount(const input::ModelSnapshot& snapshot)
{
    return snapshot.hasTable(input::TableRole::CoordinateSystems) ? snapshot.coordinateSystems().ids().size() : 0;
}

} // namespace

SnapshotIndex::SnapshotIndex(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget)
    : m_snapshot(&snapshot),
      m_lookups(std::make_shared<const Lookups>(Lookups{input::PointIndex(snapshot.nodes(), budget),
                                                        materialIndex(snapshot, budget),
                                                        coordinateSystemIndex(snapshot, budget)}))
{
}

SnapshotIndex::SnapshotIndex(const input::ModelSnapshot& snapshot, const SnapshotIndex& sameIds)
    : m_snapshot(&snapshot), m_lookups(sameIds.m_lookups)
{
    const input::ModelSnapshot& other = sameIds.snapshot();
    const bool isSameShape = snapshot.nodes().ids().size() == other.nodes().ids().size() &&
                             snapshot.nodes().scalarPointIds().size() == other.nodes().scalarPointIds().size() &&
                             materialCount(snapshot) == materialCount(other) &&
                             coordinateSystemCount(snapshot) == coordinateSystemCount(other);
    if (!isSameShape)
        throw std::invalid_argument(base::concatenate(
            {"SnapshotIndex: expected an overlay of the indexed snapshot (the same grid points, scalar points, "
             "materials and coordinate systems), found ",
             std::to_string(snapshot.nodes().ids().size()), " grid points and ",
             std::to_string(materialCount(snapshot)), " materials against ", std::to_string(other.nodes().ids().size()),
             " and ", std::to_string(materialCount(other)), "; build its own index"}));
}

std::optional<std::int32_t> SnapshotIndex::findMaterialRow(std::int32_t matId) const noexcept
{
    return m_lookups->materials.find(matId);
}

std::optional<input::CoordinateFrame> SnapshotIndex::findCoordinateFrame(std::int32_t frameId) const
{
    if (frameId == 0)
        return input::CoordinateFrame{};
    const std::optional<std::int32_t> row = m_lookups->coordinateSystems.find(frameId);
    if (!row)
        return std::nullopt;
    return m_snapshot->coordinateSystems().frameAt(*row);
}

} // namespace oos::elements
