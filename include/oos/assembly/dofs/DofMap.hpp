#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/EntityLayout.hpp"
#include "oos/elements/core/FormulationModel.hpp"
#include "oos/input/tables/NodeTable.hpp"
#include "oos/input/tables/PointIndex.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace oos::assembly
{

// Two grid points closer than this share of the model's extent (DofMap::extent) are at one place for the geometry of
// the relations: a RigidInterpolation leg on its reference point takes no lever arm. Far above the round-off of
// coordinates (1e-16 of the extent, a point given in metres to 1e-6 and written in mm: 1e-12 mm off, the
// engine hub of a test model), far below any meshed length (a 40 m aircraft: 4e-6 mm).
inline constexpr double GEOMETRIC_COINCIDENCE_RATIO = 1.0e-10;

// A DOF of the g-set named as the user reads it: the point (grid or scalar, by its row in the snapshot) and the
// component (1 to 6 for a grid point, 0 for a scalar point, as the input format numbers them); for an entity of an
// element formulation, its row in the EntityLayout and the DOF's index within it from 0.
struct DofLabel
{
    input::PointKind kind = input::PointKind::Grid;
    std::int32_t row = 0;
    std::uint8_t component = 1;

    friend constexpr bool operator==(const DofLabel&, const DofLabel&) = default;
};

// The g-set of a model: six DOFs per grid point, in its displacement
// system CD (translations 1, 2, 3, rotations 4, 5, 6), one per scalar point, then the DOFs of the entities of the
// element formulation when it has some (elements::EntityLayout: the edges, faces and interiors of the mixed hybrid
// formulation). The numbering is node-major and matches the node blocks of linalg: the grid point of row r is node r
// and its component c is g-DOF 6·r + c − 1; the scalar point of row s is node gridCount + s and g-DOF
// 6·gridCount + s (the order of input::PointIndex and of the test support's DenseSystem); the DOF i of the entity of
// row e is g-DOF 6·gridCount + scalarCount + dofOffset(e) + i, in node gridCount + scalarCount + firstNode(e) + i / 6,
// component i mod 6 (an entity is split into nodes of at most six DOFs, the largest block of linalg). A
// DofSelection built from nodeDofCounts() in the natural node order gives the same DOFs. Without entities the
// numbering is that of grid and scalar points alone.
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
    // `formulation`: the model of a formulation with entities (null: none), borrowed, and the kind of its layout this
    // analysis numbers.
    DofMap(const input::NodeTable& nodes, const input::PointIndex& points,
           const elements::FormulationModel* formulation = nullptr,
           elements::EntityLayout::Kind layoutKind = elements::EntityLayout::Kind::Static);

    [[nodiscard]] std::int32_t gridCount() const noexcept { return m_gridCount; }
    [[nodiscard]] std::int32_t scalarCount() const noexcept { return m_scalarCount; }
    // The entities of the formulation (empty without).
    [[nodiscard]] const elements::EntityLayout& entities() const noexcept { return *m_entities; }
    [[nodiscard]] bool hasEntities() const noexcept { return !m_entities->empty(); }
    // The model of the formulation with entities (null without), and the kind of its layout numbered here.
    [[nodiscard]] const elements::FormulationModel* formulation() const noexcept { return m_formulation; }
    [[nodiscard]] elements::EntityLayout::Kind layoutKind() const noexcept { return m_layoutKind; }
    // Adds to the entity DOFs of a g-set vector the values of the affine displacement u(x) = b + A·x in the basic
    // system (FormulationModel::affineField); nothing without entities. The builders of fields from the grid
    // positions (rigid-body modes, inertial loads) call it after filling the grid rows (spec section 5.7).
    void addAffineEntityField(const base::Mat3& gradient, const base::Vec3& translation,
                              std::span<double> valuesG) const;
    // The nodes of the node-block pattern: the grid points, the scalar points, then the nodes of the entities.
    [[nodiscard]] std::int32_t nodeCount() const noexcept
    {
        return m_gridCount + m_scalarCount + m_entities->nodeCount();
    }
    // The first node of the entities (every node from there on is a node of an entity).
    [[nodiscard]] std::int32_t firstEntityNode() const noexcept { return m_gridCount + m_scalarCount; }
    [[nodiscard]] base::DofIndex gSetSize() const noexcept
    {
        return 6 * m_gridCount + m_scalarCount + m_entities->totalDofCount();
    }
    // The first g-DOF of the entities.
    [[nodiscard]] base::DofIndex firstEntityDof() const noexcept { return 6 * m_gridCount + m_scalarCount; }

    // The g-DOF of component 1 to 6 of the grid point of a row.
    [[nodiscard]] base::DofIndex gridDof(std::int32_t row, int component) const noexcept
    {
        return 6 * row + component - 1;
    }
    [[nodiscard]] base::DofIndex scalarDof(std::int32_t row) const noexcept { return 6 * m_gridCount + row; }
    // The g-DOF of index `index` (from 0) of the entity of a row.
    [[nodiscard]] base::DofIndex entityDof(std::int32_t row, int index) const noexcept
    {
        return firstEntityDof() + m_entities->dofOffset(row) + index;
    }
    [[nodiscard]] base::DofIndex dofOf(const elements::ElementDof& dof) const noexcept
    {
        return dofOf(DofLabel{dof.kind, dof.row, dof.component});
    }
    [[nodiscard]] base::DofIndex dofOf(const DofLabel& label) const noexcept
    {
        switch (label.kind)
        {
        case input::PointKind::Scalar:
            return scalarDof(label.row);
        case input::PointKind::Entity:
            return entityDof(label.row, label.component);
        case input::PointKind::Grid:
            break;
        }
        return gridDof(label.row, label.component);
    }
    // The g-DOF of a point ID and a component as a card writes them (1 to 6 for a grid point; 0 for a scalar point,
    // where a blank component also reads 0). Nothing for an unknown ID or a component the point does not have.
    [[nodiscard]] std::optional<base::DofIndex> findDof(base::EntityId id, int component) const noexcept;

    [[nodiscard]] DofLabel labelOf(base::DofIndex dof) const noexcept;
    // The ID of the point of a DOF (grid point ID or scalar point ID); 0 for a DOF of an entity, which has no ID
    // (describe() names it).
    [[nodiscard]] base::EntityId pointIdOf(base::DofIndex dof) const;
    // A DOF as a message names it: "grid point 12 component 3", "scalar point 7", "DOF 2 of the edge between grid
    // points 4, 9".
    [[nodiscard]] std::string describe(base::DofIndex dof) const;
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
    [[nodiscard]] linalg::NodeDof nodeDofOf(const elements::ElementDof& dof) const noexcept;

    // The DOF count of each node (6 for a grid point, 1 for a scalar point, 1 to 6 for a node of an entity), for
    // linalg::DofSelection.
    [[nodiscard]] base::BudgetedBuffer<std::uint8_t> nodeDofCounts(base::MemoryBudget& budget) const;
    // The ID of a node as the condensed path names its points (docs/mixed-hybrid-formulation-spec.md section 5.11): a
    // grid point's or a scalar point's ID; for a node of an entity, which has no ID, entityNodeId(node −
    // firstEntityNode()), a negative number that no point of the input format has. It follows the numbering of the
    // entities: a condensation cache keys it with the entity it stands for.
    [[nodiscard]] static constexpr base::EntityId entityNodeId(std::int32_t entityNode) noexcept
    {
        return -1 - entityNode;
    }
    [[nodiscard]] base::EntityId nodeIdOf(std::int32_t node) const noexcept;
    // The node of such an ID, or nothing for an ID the model does not have.
    [[nodiscard]] std::optional<std::int32_t> findNode(base::EntityId id) const noexcept;
    // Whether a node is a node of an entity.
    [[nodiscard]] bool isEntityNode(std::int32_t node) const noexcept { return node >= firstEntityNode(); }
    // The g-DOFs of a node: its first DOF and its DOF count (a grid point's six, a scalar point's one, the DOFs of a
    // node of an entity).
    struct NodeDofs
    {
        base::DofIndex first = 0;
        int count = 0;
    };
    [[nodiscard]] NodeDofs dofsOfNode(std::int32_t node) const noexcept;
    // The position of a node in the basic system, for the geometric ordering: a grid point's, an entity's
    // (EntityLayout::position). Precondition: the node is not a scalar point (which has none).
    [[nodiscard]] base::Vec3 nodePosition(std::int32_t node) const;

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
    const elements::FormulationModel* m_formulation;
    elements::EntityLayout::Kind m_layoutKind;
    const elements::EntityLayout* m_entities;
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
