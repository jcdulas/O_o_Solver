#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/reduction/absorption/AbsorptionEligibility.hpp"
#include "oos/reduction/absorption/AbsorptionSafeguard.hpp"
#include "oos/reduction/plan/InterfaceEstimate.hpp"
#include "oos/reduction/plan/PlanGraph.hpp"
#include "oos/reduction/plan/UnitCostEstimate.hpp"
#include "oos/reduction/units/ClusterAbsorption.hpp"
#include "oos/reduction/units/UnitBoundaries.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::reduction
{

class SpiderAbsorption;

enum class UnitDecision : std::uint8_t
{
    Condensed,
    Full
};

// How a condensed unit is reduced: static condensation for statics and trim (exact), Craig-Bampton for
// modes and buckling (the boundary plus fixed-interface modes).
enum class ReductionKind : std::uint8_t
{
    Static,
    CraigBampton
};

// Which units the caller condenses (solver.plan(units, condense="none" | "all" | [units],
// full=[units])).
enum class CondenseMode : std::uint8_t
{
    All,   // every unit except those of fullUnits
    None,  // no unit: the full path
    Listed // the units of condensedUnits, except those of fullUnits
};

struct PlanRequest
{
    CondenseMode mode = CondenseMode::All;
    std::vector<std::int64_t> condensedUnits; // with Listed
    std::vector<std::int64_t> fullUnits;
    // The points the caller retains (measurement points, design responses): boundary points of their unit. The
    // boundaries passed to the plan must have been made with them (BoundaryInputs::retainedNodes).
    std::vector<base::EntityId> retainedNodes;
    ReductionKind reduction = ReductionKind::Static;
    // Craig-Bampton: the fixed-interface modes kept per unit; 0 for the frequency cut-off (CB_CUTOFF_FACTOR).
    std::int32_t modeCount = 0;
    // Absorb the eligible spiders and fastener sides into their units; false keeps
    // every spider on its legs (the "without absorption" side of the exactness check).
    bool absorbsSpiders = true;
    // The threshold of the numerical safeguard of spider absorption: SPIDER_ABSORPTION_MAX_CONDITION. Only the
    // calibration of that constant sets another, to measure what
    // absorbing an ill-conditioned spider would give.
    double maxSpiderCondition = SPIDER_ABSORPTION_MAX_CONDITION;
    // Keep full the units the plan recommends keeping full (UnitCostEstimate::isKeptFullRecommended: K* would cost
    // more than the interior), whatever the mode: the plan of condensation="auto" ("the units ...
    // the plan recommends keeping full excepted"). False: the plan recommends, it does not override.
    bool keepsFullWhereRecommended = false;
};

// The plan of one unit.
struct UnitPlan
{
    std::int64_t unitId = 0;
    UnitOrigin origin = UnitOrigin::Component;
    std::vector<std::int64_t> memberUnits;   // the unit and the 1D/0D-only units absorbed into it
    std::int64_t revision = 0;               // its stamp, absorbed clusters included
    std::vector<base::EntityId> elementIds; // its elements and the connections it absorbed, sorted
    std::vector<base::EntityId> nodeIds;    // its points, absorbed connection-only points included, sorted
    // Its boundary points before absorption: the points of its own units that a connection uses (O_o_Reader's
    // componentBoundaries, with the grid points of the elements the fasteners pierce), and those O_o_Reader's group
    // boundary adds.
    std::int64_t boundaryNodeCountBefore = 0;
    std::vector<BoundaryNode> boundary;         // after absorption, with the analysis' additions; free points only
    std::vector<base::EntityId> retainedNodes; // the caller's retained points in the unit
    std::vector<base::EntityId> absorbedNodes; // connection-only points made interior
    std::int64_t absorbedClusterCount = 0;
    // The spiders absorbed into it (indices into CondensationPlan::spiderDecisions(), increasing): their legs left the
    // boundary, their reference points (or the pierced element's grid point standing for a fastener side's point) are
    // on it.
    std::vector<std::size_t> absorbedSpiders;
    UnitDecision decision = UnitDecision::Full;
    ReductionKind reduction = ReductionKind::Static;
    std::int32_t modeCount = 0;
    UnitCostEstimate cost; // the cost of condensing it, computed for every unit (a full unit's tells what it would be)
};

// The connection system as the plan reports it.
struct ConnectionSummary
{
    std::int64_t connectionCount = 0;               // connections left in the connection system
    std::int64_t clusterCount = 0;                  // clusters left in it
    std::int64_t absorbedClusterCount = 0;          // clusters absorbed into a unit
    std::int64_t absorbedSpiderCount = 0;           // spiders and fastener sides absorbed into a unit
    std::vector<base::EntityId> connectionNodeIds; // connection-only points left in it (DOFs of the interface problem)
    std::vector<std::int64_t> movedUnits;           // the 1D/0D-only units it holds
    // Its relations (rigid elements, fasteners by element ID, MPC sets by set ID, and the glue equations that join
    // points of several units, glueRelationKey), and the relations of a unit that make a point another unit holds
    // dependent, eliminated at the interface; sorted.
    std::vector<RelationKey> relations;
};

// The condensation plan of an analysis (a choice per unit; two paths for every analysis): per unit, condensed or kept
// full, its retained points, its reduction kind, its boundary and the cost estimate
// of its condensation; the connection system; the estimate of the interface problem with this choice and of the full
// path, all before any condensation.
//
// Spider absorption: with a SpiderAbsorption, the absorbed RigidBodyStar and RigidInterpolation leave the connection
// system and become elements of their unit, their reference points points of it, and the stamps of the units mix the
// absorbed spiders; the boundaries must have been made with the same absorption (UnitBoundaries), and the plan graph
// with the absorbed RigidInterpolation references freed (solutions::CondensationPlanner).
//
// A boundary point with no DOF for its unit is not dropped blindly: constrained in every subcase
// it has nothing to condense; made dependent by a relation eliminated inside the unit (an RigidBodyStar absorbed with
// its cluster whose leg a connection uses), the relation's independent points of the unit take its place, with its
// reasons, since the connection acts on them once the relation is eliminated. A point two units hold (element lists
// splitting a component) stays on both boundaries (BoundaryPromotion), and a relation eliminated inside one unit that
// makes such a point dependent is eliminated at the interface instead, with the connection system's relations:
// the other unit's terms on its DOFs could not be eliminated inside it.
//
// Rules: the request decides unit by unit; a unit may be named by the ID of a 1D/0D-only unit absorbed into it; a unit
// moved to the connection system is never condensed (naming it is an error); a model of one unit and no connection is
// the full path whatever the request. The plan recommends, it does not override: a condensed unit whose
// K* would cost more than its interior keeps its decision and carries the recommendation (UnitCostEstimate).
//
// Thread-safety: immutable once built. Ownership: owns its lists; borrows nothing after construction.
class CondensationPlan
{
public:
    // Throws std::invalid_argument for an unknown unit, a unit both condensed and kept full, a unit of the connection
    // system named, a retained point that is no point of the model; MemoryBudgetError when an estimate does not fit.
    // glueInterfaceRelations: the glue relations whose points lie on several units (solutions::GlueUnitCheck),
    // eliminated at the interface with the connection system's: their dependent DOFs stay boundary DOFs of
    // their unit and every point of them is on the boundary of the unit that holds it (BoundaryPromotion).
    CondensationPlan(const ClusterAbsorption& absorption, const UnitBoundaries& boundaries, const PlanGraph& graph,
                     const PlanRequest& request, base::MemoryBudget& budget, base::ThreadPool* pool = nullptr,
                     const SpiderAbsorption* spiders = nullptr,
                     std::span<const RelationKey> glueInterfaceRelations = {});

    [[nodiscard]] std::span<const UnitPlan> units() const noexcept { return m_units; }
    [[nodiscard]] const UnitPlan* find(std::int64_t unitId) const noexcept;
    [[nodiscard]] const ConnectionSummary& connectionSystem() const noexcept { return m_connections; }
    [[nodiscard]] const InterfaceEstimate& interfaceEstimate() const noexcept { return m_interface; }
    // The factorisation of the whole model (every unit kept full), for comparison.
    [[nodiscard]] const InterfaceEstimate& fullPathEstimate() const noexcept { return m_fullPath; }
    // No unit condensed.
    [[nodiscard]] bool isFullPath() const noexcept;
    // Why the plan differs from the request (empty when it does not): the one-unit rule.
    [[nodiscard]] const std::string& note() const noexcept { return m_note; }
    [[nodiscard]] ReductionKind reduction() const noexcept { return m_reduction; }
    // The decision on every spider and fastener side (empty without spider absorption), in the catalog's order.
    [[nodiscard]] std::span<const SpiderDecision> spiderDecisions() const noexcept { return m_spiderDecisions; }
    // Whether the spiders were decided at all, and whether the snapshot had O_o_Reader's spiders table.
    [[nodiscard]] bool hasSpiderAbsorption() const noexcept { return m_hasSpiderAbsorption; }
    [[nodiscard]] bool hasRigidSpiders() const noexcept { return m_hasRigidSpiders; }

private:
    std::vector<UnitPlan> m_units;
    ConnectionSummary m_connections;
    InterfaceEstimate m_interface;
    InterfaceEstimate m_fullPath;
    std::string m_note;
    ReductionKind m_reduction = ReductionKind::Static;
    std::vector<SpiderDecision> m_spiderDecisions;
    bool m_hasSpiderAbsorption = false;
    bool m_hasRigidSpiders = false;
};

} // namespace oos::reduction
