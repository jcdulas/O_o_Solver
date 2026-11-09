#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"

#include <compare>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace oos::reduction
{

// A relation that makes DOFs dependent: a rigid element or a fastener's internal constraint (by its element ID), an
// MPC set (by its set ID), or the glue equations of one glued grid point (glueRelationKey).
struct RelationKey
{
    bool isMpc = false;
    base::EntityId id = 0;

    [[nodiscard]] auto operator<=>(const RelationKey&) const = default;
};

// The relation of the glue equations of one glued grid point (glued contact: assembly::GlueConstraints, one
// relation per glued point, a point being glued once): keyed by the point's ID made negative, −(1 + nodeId). Element,
// set and grid IDs are positive in the input format, so the key never collides with a rigid element, a fastener or an
// MPC set, and a glue equation can be eliminated inside a unit or at the interface point by point
// (solutions::GlueUnitCheck).
[[nodiscard]] constexpr RelationKey glueRelationKey(base::EntityId gluedNodeId) noexcept
{
    return RelationKey{false, -1 - gluedNodeId};
}
[[nodiscard]] constexpr bool isGlueRelation(const RelationKey& key) noexcept
{
    return !key.isMpc && key.id < 0;
}

// The DOFs of a point that a relation makes dependent, and the points they depend on (the terms of their rows of Gm,
// chains of relations resolved).
struct DependentPoint
{
    std::int32_t node = 0;
    RelationKey relation;
    std::int32_t dofCount = 0;
    std::vector<std::int32_t> independentNodes; // sorted; bounded by the relation
};

// The structure the cost estimates are computed on, before any matrix is assembled (every large allocation is
// accounted against the budget before it is made, from the symbolic analysis): the node graph of the model
// (the couplings of the element matrices and of the eliminated dependent DOFs, as the stiffness pattern has them),
// and per node its ID, its DOFs and its position (for the nested dissection).
//
// Two counts of DOFs, because a relation is eliminated where it belongs: a relation absorbed into a
// unit inside the unit, before the condensation; a relation of the connection system at the interface. So:
// - unconstrainedDofCounts: the DOFs not constrained in every subcase;
// - freeDofCounts: those that no relation makes dependent either: the DOFs of the interface problem;
// - unitDofCount(): what a unit factorises, the unconstrained DOFs less those made dependent by relations eliminated
//   inside it (every relation but those of the connection system): the dependent DOFs of a relation of the connection
//   system (the legs of an RigidBodyStar whose reference is a connection-only point) are boundary DOFs of their unit.
//
// Thread-safety: immutable. Ownership: borrows the graph and the arrays, which must outlive it; owns the index of IDs
// and the dependent points (their number is bounded by the dependent DOFs of the model: a std::vector, built once).
class PlanGraph
{
public:
    // nodeIds, unconstrainedDofCounts, freeDofCounts and positions: one entry per node of `graph`; dependents sorted by
    // node then relation. Throws std::invalid_argument on a size mismatch, a DOF count outside [0, 6] or a free count
    // above the unconstrained one, an ID given twice, unsorted dependents.
    PlanGraph(const linalg::NodeGraph& graph, std::span<const base::EntityId> nodeIds,
              std::span<const std::int32_t> unconstrainedDofCounts, std::span<const std::int32_t> freeDofCounts,
              std::span<const base::Vec3> positions, std::vector<DependentPoint> dependents);

    [[nodiscard]] const linalg::NodeGraph& graph() const noexcept { return *m_graph; }
    [[nodiscard]] std::span<const base::EntityId> nodeIds() const noexcept { return m_nodeIds; }
    [[nodiscard]] std::span<const std::int32_t> unconstrainedDofCounts() const noexcept
    {
        return m_unconstrainedDofCounts;
    }
    [[nodiscard]] std::span<const std::int32_t> freeDofCounts() const noexcept { return m_freeDofCounts; }
    [[nodiscard]] std::span<const base::Vec3> positions() const noexcept { return m_positions; }
    // The dependent DOFs of a node, by relation.
    [[nodiscard]] std::span<const DependentPoint> dependentsOf(std::int32_t node) const noexcept;
    // Every dependent point, sorted by node then relation (a plan graph with some relations freed is made from them:
    // the RigidInterpolation absorbed into a unit).
    [[nodiscard]] std::span<const DependentPoint> dependents() const noexcept { return m_dependents; }
    // The DOFs a unit factorises at a node: the unconstrained ones less those made dependent by a relation that is not
    // one of interfaceRelations (sorted: the relations of the connection system).
    [[nodiscard]] std::int32_t unitDofCount(std::int32_t node,
                                            std::span<const RelationKey> interfaceRelations) const noexcept;
    // The node of a grid or scalar point ID, or nothing.
    [[nodiscard]] std::optional<std::int32_t> nodeOf(base::EntityId nodeId) const noexcept;

private:
    const linalg::NodeGraph* m_graph;
    std::span<const base::EntityId> m_nodeIds;
    std::span<const std::int32_t> m_unconstrainedDofCounts;
    std::span<const std::int32_t> m_freeDofCounts;
    std::span<const base::Vec3> m_positions;
    std::vector<DependentPoint> m_dependents;
    std::vector<std::pair<base::EntityId, std::int32_t>> m_index; // (ID, node), sorted
};

} // namespace oos::reduction
