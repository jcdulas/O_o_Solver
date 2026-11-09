#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"
#include "oos/input/tables/IdIndex.hpp"
#include "oos/input/tables/PointIndex.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <typeindex>
#include <typeinfo>

namespace oos::elements
{

// The lookups by ID that the element families need and the snapshot's row-index columns do not give: grid and scalar
// points (for the cards without nodesRows), materials (a property names its material by MID),
// coordinate systems (CBUSH CID and OCID, PointMassMatrix and PointMassInertia CID). Built once per snapshot and shared by every
// batch view; shared too by the index of an overlay of the snapshot that changes no ID (a design perturbation), see the
// second constructor.
//
// Thread-safety: immutable once built; concurrent lookups are safe (computedOnce builds its values under a lock).
// Ownership: shares its indices (budgeted) with the indices made from it; borrows the snapshot, which must outlive it.
class SnapshotIndex
{
public:
    // Throws as IdIndex (duplicate IDs, budget).
    SnapshotIndex(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget);

    // The index of `snapshot`, an overlay of the snapshot of `sameIds` that changes no ID column and no row count (a
    // design point's field moved by a sensitivity perturbation: ColumnOverlay changes values of real columns only),
    // sharing the lookups of `sameIds` instead of building them again: a lookup by ID gives the same row in both.
    // Throws std::invalid_argument when the node, scalar point, material or coordinate system tables differ in rows.
    SnapshotIndex(const input::ModelSnapshot& snapshot, const SnapshotIndex& sameIds);

    [[nodiscard]] const input::ModelSnapshot& snapshot() const noexcept { return *m_snapshot; }
    [[nodiscard]] const input::PointIndex& points() const noexcept { return m_lookups->points; }
    // The row of a material in the materials table, or nothing (no such material, or no materials table).
    [[nodiscard]] std::optional<std::int32_t> findMaterialRow(std::int32_t matId) const noexcept;
    // The frame of a coordinate system: the basic system for CID 0, the row of coordsTable otherwise; nothing for a
    // system the model does not define (or a snapshot taken without coordsTable).
    [[nodiscard]] std::optional<input::CoordinateFrame> findCoordinateFrame(std::int32_t frameId) const;

    // A value an element family computes from the whole snapshot, once, on its first use, and then shares between
    // all its batch views (the shared nodal directors of the higher-order shells: shells/HigherOrderGeometry). One
    // value per type T: `build()` returns the T. The value belongs to this index, not to the indices made from it
    // with the second constructor (an overlay may move the grid points). Concurrent callers wait for the one build.
    // Not budgeted: such values are a few numbers per grid point of the elements that use them.
    template <typename T, typename Build>
    [[nodiscard]] const T& computedOnce(Build&& build) const
    {
        const std::lock_guard<std::mutex> lock(m_computed->mutex);
        std::shared_ptr<const void>& value = m_computed->values[std::type_index(typeid(T))];
        if (!value)
            value = std::make_shared<const T>(build());
        return *static_cast<const T*>(value.get());
    }

private:
    struct ComputedValues
    {
        std::mutex mutex;
        std::map<std::type_index, std::shared_ptr<const void>> values;
    };

    struct Lookups
    {
        input::PointIndex points;
        input::IdIndex materials;
        input::IdIndex coordinateSystems;
    };

    const input::ModelSnapshot* m_snapshot;
    std::shared_ptr<const Lookups> m_lookups;
    std::shared_ptr<ComputedValues> m_computed = std::make_shared<ComputedValues>();
};

} // namespace oos::elements
