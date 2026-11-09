#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/input/tables/NodeTable.hpp"
#include "oos/input/tables/PointIndex.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"

#include <cstdint>
#include <optional>

namespace oos::assembly
{

// Two grid points closer than this share of the model's extent (DofMap::extent) are at one place for the geometry of
// the relations: an RigidInterpolation leg on its reference point takes no lever arm. Far above the round-off of
// coordinates (1e-16 of the extent, a point given in metres to 1e-6 and written in mm: 1e-12 mm off, the
// engine hub of a test model), far below any meshed length (a 40 m aircraft: 4e-6 mm).
inline constexpr double GEOMETRIC_COINCIDENCE_RATIO = 1.0e-10;

// A DOF of the g-set named as the user reads it: the point (grid or scalar, by its row in the snapshot) and the
// component (1 to 6 for a grid point, 0 for a scalar point, as the input format numbers them).
struct DofLabel
{
    input::PointKind kind = input::PointKind::Grid;
    std::int32_t row = 0;
    std::uint8_t component = 1;

    friend constexpr bool operator==(const DofLabel&, const DofLabel&) = default;
};

// The g-set of a model: six DOFs per grid point, in its displacement
// system CD (translations 1, 2, 3, rotations 4, 5, 6), and one per scalar point. The numbering is node-major and
// matches the node blocks of linalg: the grid point of row r is node r and its component c is g-DOF 6·r + c − 1; the
// scalar point of row s is node gridCount + s and g-DOF 6·gridCount + s (the order of input::PointIndex and of the test
// support's DenseSystem). A DofSelection built from nodeDofCounts() in the natural node order gives the same DOFs.
//
// The displacement systems come from the snapshot's CD rotations (O_o_Reader displacementRotations: the columns of each
// 3 × 3 are the CD axes in the basic system, evaluated at the point, so a cylindrical or spherical CD is already
// resolved).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: borrows the node table and the point index, which
// must outlive it.
class DofMap
{
public:
    // Throws ContractError when the model has more DOFs than a DofIndex holds, or when a grid point has a
    // CD system other than 0 and the snapshot was taken without the CD rotations.
    DofMap(const input::NodeTable& nodes, const input::PointIndex& points);

    [[nodiscard]] std::int32_t gridCount() const noexcept { return m_gridCount; }
    [[nodiscard]] std::int32_t scalarCount() const noexcept { return m_scalarCount; }
    // The nodes of the node-block pattern: the grid points, then the scalar points.
    [[nodiscard]] std::int32_t nodeCount() const noexcept { return m_gridCount + m_scalarCount; }
    [[nodiscard]] base::DofIndex gSetSize() const noexcept { return 6 * m_gridCount + m_scalarCount; }

    // The g-DOF of component 1 to 6 of the grid point of a row.
    [[nodiscard]] base::DofIndex gridDof(std::int32_t row, int component) const noexcept
    {
        return 6 * row + component - 1;
    }
    [[nodiscard]] base::DofIndex scalarDof(std::int32_t row) const noexcept { return 6 * m_gridCount + row; }
    [[nodiscard]] base::DofIndex dofOf(const elements::ElementDof& dof) const noexcept
    {
        return dof.kind == input::PointKind::Scalar ? scalarDof(dof.row) : gridDof(dof.row, dof.component);
    }
    [[nodiscard]] base::DofIndex dofOf(const DofLabel& label) const noexcept
    {
        return label.kind == input::PointKind::Scalar ? scalarDof(label.row) : gridDof(label.row, label.component);
    }
    // The g-DOF of a point ID and a component as a card writes them (1 to 6 for a grid point; 0 for a scalar point,
    // where a blank component also reads 0). Nothing for an unknown ID or a component the point does not have.
    [[nodiscard]] std::optional<base::DofIndex> findDof(base::EntityId id, int component) const noexcept;

    [[nodiscard]] DofLabel labelOf(base::DofIndex dof) const noexcept;
    // The ID of the point of a DOF (grid point ID or scalar point ID).
    [[nodiscard]] base::EntityId pointIdOf(base::DofIndex dof) const;
    [[nodiscard]] base::EntityId gridId(std::int32_t row) const noexcept
    {
        return m_gridIds[static_cast<std::size_t>(row)];
    }
    [[nodiscard]] base::EntityId scalarId(std::int32_t row) const noexcept
    {
        return m_scalarIds[static_cast<std::size_t>(row)];
    }
    // The node and component (from 0) of a DOF in the node-block pattern of linalg.
    [[nodiscard]] linalg::NodeDof nodeDofOf(base::DofIndex dof) const noexcept;

    // The DOF count of each node (6 for a grid point, 1 for a scalar point), for linalg::DofSelection.
    [[nodiscard]] base::BudgetedBuffer<std::uint8_t> nodeDofCounts(base::MemoryBudget& budget) const;

    // The rotation from the CD system of a grid point to the basic system: v_basic = R·v_cd (identity for CD 0).
    [[nodiscard]] base::Mat3 cdToBasic(std::int32_t gridRow) const;
    [[nodiscard]] base::Vec3 positionInBasic(std::int32_t gridRow) const;
    // The largest side of the box of the grid points in the basic system (0 without grid points), and the distance
    // below which two points are at one place for the relations' geometry (GEOMETRIC_COINCIDENCE_RATIO of it).
    [[nodiscard]] double extent() const noexcept { return m_extent; }
    [[nodiscard]] double coincidenceDistance() const noexcept { return GEOMETRIC_COINCIDENCE_RATIO * m_extent; }

    [[nodiscard]] const input::NodeTable& nodes() const noexcept { return *m_nodes; }
    [[nodiscard]] const input::PointIndex& points() const noexcept { return *m_points; }

private:
    const input::NodeTable* m_nodes;
    const input::PointIndex* m_points;
    std::int32_t m_gridCount = 0;
    std::int32_t m_scalarCount = 0;
    // PERFORMANCE: the ID and position columns, found once instead of by name at every call (gridId, scalarId and
    // positionInBasic are called for every grid point of every output table and load resultant of every case); they
    // view the node table, which outlives the map (the snapshot owns it).
    // Measured: a model of 100 components, 197 640 DOFs, 600 cases:
    // the condensed run 11.1 s before, 3.7 s with this and the two per-case changes of the same merge
    // (LoadResultant::of, NodalResultBuilder::build), the same results.
    std::span<const std::int32_t> m_gridIds;
    std::span<const std::int32_t> m_scalarIds;
    base::FixedListView<double> m_positions;
    double m_extent = 0.0;
};

} // namespace oos::assembly
