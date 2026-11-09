#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/IdIndex.hpp"
#include "oos/input/tables/NodeTable.hpp"

#include <cstdint>
#include <optional>

namespace oos::input
{

// The two kinds of points an element connects: a grid point (six DOFs in its displacement system CD) or a
// scalar point (SPOINT, one DOF without a frame). Grid points and scalar points share one ID space.
enum class PointKind : std::uint8_t
{
    Grid,
    Scalar
};

// A point of the model by its row: the row of a grid point in NodeTable, or of a scalar point in the scalar point
// table of the NodeTable.
struct PointRef
{
    PointKind kind = PointKind::Grid;
    std::int32_t row = 0;

    friend constexpr bool operator==(const PointRef&, const PointRef&) = default;
};

// The row of a grid point or scalar point by its ID, for the element cards whose table gives grid point IDs and no
// row-index column (CELASi, CMASSi, CBUSH, CONM1, CONM2). Built once per snapshot.
//
// Thread-safety: immutable once built; concurrent lookups are safe. Ownership: owns its indices (budgeted); the
// budget must outlive it. Movable, not copyable.
class PointIndex
{
public:
    // Throws as IdIndex (duplicate IDs, budget).
    PointIndex(const NodeTable& nodes, base::MemoryBudget& budget);

    // The point of an ID: a grid point when the node table holds it, else a scalar point; nothing for an unknown ID.
    [[nodiscard]] std::optional<PointRef> find(std::int32_t id) const noexcept;
    [[nodiscard]] std::optional<std::int32_t> findGrid(std::int32_t id) const noexcept { return m_grids.find(id); }
    [[nodiscard]] std::optional<std::int32_t> findScalar(std::int32_t id) const noexcept { return m_scalars.find(id); }

private:
    IdIndex m_grids;
    IdIndex m_scalars;
};

} // namespace oos::input
