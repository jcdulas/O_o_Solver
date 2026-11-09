#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/absorption/AbsorptionEligibility.hpp"
#include "oos/reduction/absorption/AbsorptionSafeguard.hpp"
#include "oos/reduction/absorption/SharedConnectionNodes.hpp"
#include "oos/reduction/absorption/SpiderCatalog.hpp"
#include "oos/reduction/units/ClusterAbsorption.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace oos::reduction
{

// A point whose dependent DOFs an absorbed spider takes out of the plan graph's relations (the planner removes them):
// - the REFGRID of an absorbed RigidInterpolation: in the unit's basis its REFC DOFs are the reference DOFs of the
// change of basis,
//   free boundary DOFs of the unit (isFreeAtReference);
// - the GA (GB) of an absorbed fastener side when the card gives it as a grid point: its rows follow the pierced
//   element, which the change of basis now carries; the grid point of the pierced element standing for the side's
//   point holds the six DOFs in the estimates, so GA's own are not counted again (isFreeAtReference false) and the
//   grid points of its pierced element are not promoted back to the boundary as points of a relation of the
//   interface (BoundaryPromotion::interfaceRelationPoints).
struct FreedReference
{
    base::EntityId nodeId = 0;
    base::EntityId elementId = 0; // the RigidInterpolation, Fastener or Weld
    bool isFreeAtReference = true;
};

// The absorption of the spiders of the connection system into their units (the rigid spiders and the fastener sides
// alike): after the cluster absorption, each spider of the
// catalog whose card is still a connection of the connection system (a cluster absorbed whole took its spiders with
// it) and whose legs are all on one unit is decided by AbsorptionEligibility (O_o_Reader's topological rule, the
// exclusions of the analysis, the numerical safeguard). An absorbed spider:
// - RigidInterpolation or RigidBodyStar: becomes part of its unit (its element ID among the unit's elements, its
// revision in the unit's
//   stamp); its reference point joins the unit's points as a boundary point, its legs leave the boundary (unless
//   something else keeps them there); an RigidInterpolation's REFC DOFs become free DOFs of the unit (the reference
//   DOFs of the change of basis, FreedReference); an RigidBodyStar is eliminated inside the unit by its own equations;
// - fastener side: the grid points of its pierced element leave the boundary, the fastener stays a connection (its
//   stiffness between P_A and P_B); P has no model point when it is generated, so the plan puts the smallest grid
//   point of the pierced element on the boundary to stand for it (SpiderDecision::boundaryNodeId).
// As a last check, two absorbed spiders that share a leg (or whose reference is the other's leg) are both kept on their
// legs (O_o_Reader's rule already excludes them; the check keeps the change of basis well defined whatever the tables).
// This is exact: the relation is the same equation, eliminated inside the unit instead of at the interface.
//
// Thread-safety: immutable once built. Ownership: owns its lists (one decision per spider of the catalog: a
// std::vector, as the catalog); borrows the cluster absorption, which must outlive it.
class SpiderAbsorption
{
public:
    // measurements: one per spider of the catalog (solutions::CondensationPlanner measures them on the model).
    // sharing may be null (the decisions then name the blocking connections by ID only). absorbsSpiders false decides
    // every spider NotRequested once eligible. maxCondition: the threshold of the numerical safeguard
    // (SPIDER_ABSORPTION_MAX_CONDITION but in the calibration studies). Throws std::invalid_argument when measurements
    // and catalog differ in size.
    SpiderAbsorption(const ClusterAbsorption& absorption, const SpiderCatalog& catalog,
                     std::span<const SpiderMeasurement> measurements, const AbsorptionExclusions& exclusions,
                     const SharedConnectionNodes* sharing, bool absorbsSpiders = true,
                     double maxCondition = SPIDER_ABSORPTION_MAX_CONDITION);

    // One decision per spider of the catalog, in its order.
    [[nodiscard]] std::span<const SpiderDecision> decisions() const noexcept { return m_decisions; }
    [[nodiscard]] std::int64_t absorbedCount() const noexcept { return m_absorbedCount; }

    // Per unit, by index into absorption.units():
    // the decisions of its absorbed spiders (indices into decisions(), increasing);
    [[nodiscard]] std::span<const std::size_t> absorbedIn(std::size_t unitIndex) const noexcept
    {
        return m_absorbed[unitIndex];
    }
    // the points the absorbed spiders put on its boundary (references, stand-ins), sorted;
    [[nodiscard]] std::span<const base::EntityId> referenceNodes(std::size_t unitIndex) const noexcept
    {
        return m_referenceNodes[unitIndex];
    }
    // the RigidBodyStar and RigidInterpolation that became its elements, sorted;
    [[nodiscard]] std::span<const base::EntityId> absorbedElementIds(std::size_t unitIndex) const noexcept
    {
        return m_absorbedElements[unitIndex];
    }
    // its stamp with the absorbed spiders (the unit's own when none: an absorbed spider is part of its unit, so
    // changing its weights changes the unit's stamp).
    [[nodiscard]] std::int64_t revisionOf(std::size_t unitIndex) const noexcept { return m_revisions[unitIndex]; }

    // The points of a connection of the connection system (index into ConnectionSystem::connections()) that the
    // absorption takes off the interface: every point of an absorbed RigidBodyStar or RigidInterpolation, the grid
    // points of the element an absorbed fastener side pierces. Sorted; empty for most connections.
    [[nodiscard]] std::span<const base::EntityId> removedUses(std::size_t connection) const noexcept;
    // Whether a connection left the connection system (an absorbed RigidBodyStar or RigidInterpolation).
    [[nodiscard]] bool isAbsorbedConnection(std::size_t connection) const noexcept;
    // The RigidInterpolation references freed, sorted by point.
    [[nodiscard]] std::span<const FreedReference> freedReferences() const noexcept { return m_freedReferences; }
    // Whether the catalog had the spiders table (false: no RigidBodyStar or RigidInterpolation decided; the plan says
    // so).
    [[nodiscard]] bool hasRigidSpiders() const noexcept { return m_hasRigidSpiders; }

    [[nodiscard]] const ClusterAbsorption& clusters() const noexcept { return *m_clusters; }

private:
    const ClusterAbsorption* m_clusters;
    std::vector<SpiderDecision> m_decisions;
    std::int64_t m_absorbedCount = 0;
    std::vector<std::vector<std::size_t>> m_absorbed;                               // per unit index
    std::vector<std::vector<base::EntityId>> m_referenceNodes;                      // per unit index
    std::vector<std::vector<base::EntityId>> m_absorbedElements;                    // per unit index
    std::vector<std::int64_t> m_revisions;                                          // per unit index
    std::vector<std::pair<std::size_t, std::vector<base::EntityId>>> m_removedUses; // by connection, sorted
    std::vector<std::size_t> m_absorbedConnections;                                 // sorted
    std::vector<FreedReference> m_freedReferences;
    bool m_hasRigidSpiders = false;
};

} // namespace oos::reduction
