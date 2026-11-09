#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/plan/PlanGraph.hpp"
#include "oos/reduction/units/UnitBoundaries.hpp"

#include <span>
#include <vector>

namespace oos::reduction
{

// The rule for a boundary point that a relation makes dependent (for every relation: RigidBodyStar, RigidInterpolation,
// RigidBar, MPC, fastener constraints and the glue equations). A point is on a unit's boundary because something
// outside the unit acts on it (a connection, a retained output, a constraint of some subcases only, a FreeBodySupport);
// if some of its DOFs are dependent, u_m = Gm·u_n, what acts on them acts on the DOFs they depend on. So that the
// coupling is never lost and the condensation stays exact:
// - a relation eliminated at the interface (one of `interfaceRelations`: a relation of the connection system, or a
//   glue equation whose points are on several units) leaves the dependent DOFs to the unit as boundary DOFs
//   (PlanGraph::unitDofCount counts them): the point stays a boundary point with them;
// - a relation eliminated inside the unit (absorbed with a cluster, an absorbed spider, a glue inside a merged unit)
//   removes the dependent DOFs from the unit: the points they depend on (DependentPoint::independentNodes, the terms of
//   their rows of Gm, chains resolved) are promoted to the boundary with the reasons of the dependent point;
// - the point itself stays a boundary point when it keeps DOFs for the unit, and leaves it when none is left
//   (every DOF dependent inside the unit, or constrained in every subcase), unless another unit holds it too
//   (BoundaryReason::SharedPoint: an element list of a regrouping sharing the points of its edge with its
//   neighbour): a point two units hold is on both boundaries whatever its DOFs, never interior to two units
//   (PlannedUnits refuses that). Held in every subcase, it brings no DOF to condense; each unit's terms there enter its
//   reaction, the SPC force the sum of the units'; a relation of one unit making it dependent is eliminated at the
//   interface (CondensationPlan), so its DOFs stay boundary DOFs of every unit that holds it.
// Only points of the unit are promoted (a relation eliminated inside the unit has all its points in it); a promoted
// point with no DOF for the unit (constrained in every subcase) is not added.
//
// The converse rule, for the relations eliminated at the interface (interfaceRelationPoints): every point of such a
// relation, dependent or independent, is a boundary point of the units that hold it, whether or not O_o_Reader's
// boundary names it (glued contact: the components of O_o_Reader and the glue's own search of the solver may pair
// different points), so that the equation never reaches an interior DOF. Such a
// point stays on the boundary even when every DOF of it is constrained in every subcase (the leg of an
// RigidInterpolation kept on its legs on a clamped edge): the unit then brings none of its DOFs, and the interface
// problem holds its six DOFs (their enforced values enter the relation as the full path's u_s enter u_m = Gm·u_n).
//
// Thread-safety: stateless. Ownership: none.
class BoundaryPromotion
{
public:
    // The boundary of a unit after the rule, sorted by point, each point once with the union of its reasons.
    // unitNodes: the unit's points, sorted; boundary: its boundary points before the rule (UnitBoundaries);
    // interfaceRelations: sorted; relationPoints: interfaceRelationPoints() (kept whatever their unit DOFs). Points the
    // graph does not hold are left out.
    [[nodiscard]] static std::vector<BoundaryNode> apply(const PlanGraph& graph,
                                                         std::span<const base::EntityId> unitNodes,
                                                         std::span<const BoundaryNode> boundary,
                                                         std::span<const RelationKey> interfaceRelations,
                                                         std::span<const base::EntityId> relationPoints = {});

    // Every point of the relations eliminated at the interface (their dependent points and the points their rows of
    // Gm depend on), sorted: the points a unit must have on its boundary (reason Interface) when it holds them.
    // The size is bounded by the points of the connection system's relations.
    [[nodiscard]] static std::vector<base::EntityId> interfaceRelationPoints(
        const PlanGraph& graph, std::span<const RelationKey> interfaceRelations);
};

} // namespace oos::reduction
