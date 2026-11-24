#include "oos/reduction/plan/CondensationPlan.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/reduction/absorption/SpiderAbsorption.hpp"
#include "oos/reduction/plan/BoundaryPromotion.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

namespace
{

// The unit of the plan a requested ID names (its own, or that of the unit it was absorbed into).
std::int64_t resolveUnit(const ClusterAbsorption& absorption, std::int64_t requested, std::string_view list)
{
    if (const std::optional<std::int64_t> unit = absorption.unitOfMember(requested))
        return *unit;
    if (absorption.system().isMoved(requested))
        throw std::invalid_argument(base::concatenate(
            {"plan: unit ", std::to_string(requested), " of '", list,
             "' is made only of 1D and 0D elements linking several units: it is part of the connection system and "
             "never condensed; remove it from the list, or regroup it with a unit"}));
    throw std::invalid_argument(base::concatenate({"plan: unit ", std::to_string(requested), " of '", list,
                                                   "' is not one of the units given; take the IDs from "
                                                   "solver.units() or solver.regroup()"}));
}

std::vector<std::int64_t> resolveAll(const ClusterAbsorption& absorption, const std::vector<std::int64_t>& requested,
                                     std::string_view list)
{
    std::vector<std::int64_t> units;
    for (const std::int64_t id : requested)
        units.push_back(resolveUnit(absorption, id, list));
    std::sort(units.begin(), units.end());
    units.erase(std::unique(units.begin(), units.end()), units.end());
    return units;
}

// The relations eliminated inside a unit whose dependent points another unit holds too, sorted: they must be
// eliminated at the interface. An element list of Solver.regroup that splits a component shares the points of its
// edge with its neighbour (UnitBoundaries, SharedPoint); a rigid element, a RigidInterpolation or an MPC set of one
// side that makes such a point dependent would leave the other side with element terms on DOFs it cannot eliminate
// (their independent points are not its points: UnitSystem refuses it). Eliminated at the interface, the relation
// leaves the dependent DOFs to every unit that holds them as boundary DOFs (PlanGraph::unitDofCount) and puts its
// points on the boundaries of their units (BoundaryPromotion::interfaceRelationPoints): the elimination is a
// congruence, Tᵀ·(Σ K)·T = Σ Tᵀ·K·T, so where it is made does not change the result. `known`: the relations already
// eliminated there, sorted.
std::vector<RelationKey> sharedDependentRelations(const ClusterAbsorption& absorption, const PlanGraph& graph,
                                                  const SpiderAbsorption* spiders, std::span<const RelationKey> known,
                                                  base::MemoryBudget& budget)
{
    // How many units hold each node of the graph (saturated at 2: shared or not).
    const auto nodeCount = static_cast<std::size_t>(graph.graph().nodeCount());
    base::BudgetedBuffer<std::uint8_t> holders(budget, nodeCount, "plan: shared points");
    std::fill(holders.data(), holders.data() + nodeCount, std::uint8_t{0});
    const auto count = [&](base::EntityId nodeId)
    {
        if (const std::optional<std::int32_t> node = graph.nodeOf(nodeId))
        {
            std::uint8_t& held = holders[static_cast<std::size_t>(*node)];
            held = static_cast<std::uint8_t>(std::min(2, held + 1));
        }
    };
    for (std::size_t index = 0; index < absorption.units().size(); ++index)
    {
        const std::vector<base::EntityId>& nodes = absorption.units()[index].nodeIds;
        for (const base::EntityId nodeId : nodes)
            count(nodeId);
        // An absorbed spider's reference that was a connection-only point is a point of its unit (UnitBoundaries).
        if (spiders != nullptr)
            for (const base::EntityId nodeId : spiders->referenceNodes(index))
                if (!std::binary_search(nodes.begin(), nodes.end(), nodeId))
                    count(nodeId);
    }
    std::vector<RelationKey> shared; // bounded by the dependent points
    for (const DependentPoint& entry : graph.dependents())
        if (holders[static_cast<std::size_t>(entry.node)] > 1 &&
            !std::binary_search(known.begin(), known.end(), entry.relation))
            shared.push_back(entry.relation);
    std::sort(shared.begin(), shared.end());
    shared.erase(std::unique(shared.begin(), shared.end()), shared.end());
    return shared;
}

// The entity nodes of each unit (its elements' and those of the spiders absorbed into it) and those to put on its
// boundary (docs/mixed-hybrid-formulation-spec.md section 5.11): an entity node another unit or an element of no unit
// (a connection: a fastener that reads a pierced shell's unknowns) holds too, and a node whose constraints some
// subcases only have. Empty without entities.
struct UnitEntityNodes
{
    std::vector<std::vector<base::EntityId>> nodes;    // per unit of the absorption, sorted IDs
    std::vector<std::vector<base::EntityId>> boundary; // per unit, sorted IDs
};

UnitEntityNodes unitEntityNodesOf(const ClusterAbsorption& absorption, const PlanGraph& graph,
                                  const SpiderAbsorption* spiders)
{
    UnitEntityNodes result;
    const ElementEntityNodes* table = graph.entityNodes();
    const std::size_t unitCount = absorption.units().size();
    result.nodes.resize(unitCount);
    result.boundary.resize(unitCount);
    if (table == nullptr)
        return result;
    std::vector<std::vector<std::int32_t>> nodesOf(unitCount); // the units' entity nodes, bounded by the model's
    std::vector<std::uint8_t> isUnitElement(table->elementIds.size(), 0);
    for (std::size_t index = 0; index < unitCount; ++index)
    {
        std::vector<base::EntityId> elements = absorption.units()[index].elementIds;
        if (spiders != nullptr)
        {
            const std::span<const base::EntityId> absorbed = spiders->absorbedElementIds(index);
            elements.insert(elements.end(), absorbed.begin(), absorbed.end());
        }
        for (const base::EntityId elementId : elements)
        {
            const auto found = std::lower_bound(table->elementIds.begin(), table->elementIds.end(), elementId);
            if (found == table->elementIds.end() || *found != elementId)
                continue;
            isUnitElement[static_cast<std::size_t>(found - table->elementIds.begin())] = 1;
            const std::span<const std::int32_t> nodes = table->of(elementId);
            nodesOf[index].insert(nodesOf[index].end(), nodes.begin(), nodes.end());
        }
        std::sort(nodesOf[index].begin(), nodesOf[index].end());
        nodesOf[index].erase(std::unique(nodesOf[index].begin(), nodesOf[index].end()), nodesOf[index].end());
    }
    // How many holders each entity node has: the units, and the elements of no unit (saturated at 2).
    std::map<std::int32_t, std::uint8_t> holders; // bounded by the entity nodes
    for (const std::vector<std::int32_t>& nodes : nodesOf)
        for (const std::int32_t node : nodes)
        {
            std::uint8_t& count = holders[node];
            count = static_cast<std::uint8_t>(std::min(2, count + 1));
        }
    for (std::size_t k = 0; k < table->elementIds.size(); ++k)
        if (isUnitElement[k] == 0)
            for (std::size_t j = table->offsets[k]; j < table->offsets[k + 1]; ++j)
                holders[table->nodes[j]] = 2;
    for (std::size_t index = 0; index < unitCount; ++index)
        for (const std::int32_t node : nodesOf[index])
        {
            const base::EntityId nodeId = graph.nodeIds()[static_cast<std::size_t>(node)];
            result.nodes[index].push_back(nodeId);
            if (holders[node] > 1 || std::binary_search(table->varyingNodes.begin(), table->varyingNodes.end(), node))
                result.boundary[index].push_back(nodeId);
        }
    for (std::size_t index = 0; index < unitCount; ++index)
    {
        std::sort(result.nodes[index].begin(), result.nodes[index].end());
        std::sort(result.boundary[index].begin(), result.boundary[index].end());
    }
    return result;
}

// The relations of the element formulation eliminated inside a unit whose points are not all the unit's (a tie or an
// observer relation across units that share grid points: element lists splitting a component), sorted: eliminated at
// the interface, as sharedDependentRelations does for the dependent points two units hold.
std::vector<RelationKey> formulationRelationsAcrossUnits(const ClusterAbsorption& absorption, const PlanGraph& graph,
                                                         const UnitEntityNodes& entityNodes,
                                                         std::span<const RelationKey> known)
{
    std::vector<RelationKey> across; // bounded by the dependent points
    if (graph.entityNodes() == nullptr)
        return across;
    // The unit of each node a single unit holds; −1 for a node of no unit, −2 for a node several units hold.
    constexpr std::int32_t NO_UNIT = -1;
    constexpr std::int32_t SEVERAL_UNITS = -2;
    std::vector<std::int32_t> unitOf(static_cast<std::size_t>(graph.graph().nodeCount()), NO_UNIT);
    const auto hold = [&](base::EntityId nodeId, std::size_t index)
    {
        if (const std::optional<std::int32_t> node = graph.nodeOf(nodeId))
        {
            std::int32_t& unit = unitOf[static_cast<std::size_t>(*node)];
            unit = unit == NO_UNIT || unit == static_cast<std::int32_t>(index) ? static_cast<std::int32_t>(index)
                                                                               : SEVERAL_UNITS;
        }
    };
    for (std::size_t index = 0; index < absorption.units().size(); ++index)
    {
        for (const base::EntityId nodeId : absorption.units()[index].nodeIds)
            hold(nodeId, index);
        for (const base::EntityId nodeId : entityNodes.nodes[index])
            hold(nodeId, index);
    }
    for (const DependentPoint& entry : graph.dependents())
    {
        if (!entry.relation.isFormulation || std::binary_search(known.begin(), known.end(), entry.relation))
            continue;
        const std::int32_t unit = unitOf[static_cast<std::size_t>(entry.node)];
        const bool isAcross =
            unit < 0 || std::any_of(
                            entry.independentNodes.begin(), entry.independentNodes.end(),
                            [&](std::int32_t node) { return unitOf[static_cast<std::size_t>(node)] != unit; });
        if (isAcross)
            across.push_back(entry.relation);
    }
    std::sort(across.begin(), across.end());
    across.erase(std::unique(across.begin(), across.end()), across.end());
    return across;
}

} // namespace

CondensationPlan::CondensationPlan(const ClusterAbsorption& absorption, const UnitBoundaries& boundaries,
                                   const PlanGraph& graph, const PlanRequest& request, base::MemoryBudget& budget,
                                   base::ThreadPool* pool, const SpiderAbsorption* spiders,
                                   std::span<const RelationKey> glueInterfaceRelations)
    : m_reduction(request.reduction),
      m_hasSpiderAbsorption(spiders != nullptr),
      m_hasRigidSpiders(spiders != nullptr && spiders->hasRigidSpiders())
{
    if (spiders != nullptr)
        m_spiderDecisions.assign(spiders->decisions().begin(), spiders->decisions().end());
    const ConnectionSystem& system = absorption.system();
    const UnitSet& set = system.units();
    const std::vector<std::int64_t> condensed = resolveAll(absorption, request.condensedUnits, "condense");
    const std::vector<std::int64_t> full = resolveAll(absorption, request.fullUnits, "full");
    for (const std::int64_t unit : condensed)
        if (std::binary_search(full.begin(), full.end(), unit))
            throw std::invalid_argument(
                base::concatenate({"plan: unit ", std::to_string(unit),
                                   " is both in 'condense' and in 'full'; name each unit in one list at most"}));
    for (const base::EntityId nodeId : request.retainedNodes)
        if (!graph.nodeOf(nodeId))
            throw std::invalid_argument(
                base::concatenate({"plan: retained point ", std::to_string(nodeId),
                                   " is not a grid or scalar point of the model; retain points of the model only"}));

    // The relations of the connection system: eliminated at the interface, their dependent DOFs on a unit are boundary
    // DOFs of the unit (PlanGraph::unitDofCount). An absorbed RigidBodyStar or RigidInterpolation is the unit's.
    for (const std::size_t c : absorption.remainingClusters())
        for (const std::size_t connection : system.clusters()[c].connections)
        {
            if (spiders != nullptr && spiders->isAbsorbedConnection(connection))
                continue;
            const Connection& entry = system.connections()[connection];
            if (entry.kind == ConnectionKind::Mpc)
                m_connections.relations.push_back(RelationKey{true, entry.id});
            else if (entry.kind == ConnectionKind::Rigid || entry.kind == ConnectionKind::RigidInterpolation ||
                     entry.kind == ConnectionKind::Fastener)
                m_connections.relations.push_back(RelationKey{false, entry.id});
        }
    // The glue equations that join units are eliminated at the interface too.
    m_connections.relations.insert(m_connections.relations.end(), glueInterfaceRelations.begin(),
                                   glueInterfaceRelations.end());
    std::sort(m_connections.relations.begin(), m_connections.relations.end());
    m_connections.relations.erase(std::unique(m_connections.relations.begin(), m_connections.relations.end()),
                                  m_connections.relations.end());
    // The relations of a unit that make a point another unit holds dependent are eliminated at the interface too, and
    // so are the formulation's relations whose points are not all one unit's.
    const std::vector<RelationKey> shared =
        sharedDependentRelations(absorption, graph, spiders, m_connections.relations, budget);
    if (!shared.empty())
    {
        m_connections.relations.insert(m_connections.relations.end(), shared.begin(), shared.end());
        std::sort(m_connections.relations.begin(), m_connections.relations.end());
    }
    const UnitEntityNodes entityNodes = unitEntityNodesOf(absorption, graph, spiders);
    const std::vector<RelationKey> across =
        formulationRelationsAcrossUnits(absorption, graph, entityNodes, m_connections.relations);
    if (!across.empty())
    {
        m_connections.relations.insert(m_connections.relations.end(), across.begin(), across.end());
        std::sort(m_connections.relations.begin(), m_connections.relations.end());
    }
    const std::span<const RelationKey> interfaceRelations = m_connections.relations;
    const std::vector<base::EntityId> relationPoints =
        BoundaryPromotion::interfaceRelationPoints(graph, interfaceRelations);
    std::vector<base::EntityId> connectionUse; // every point a connection uses, sorted
    for (const Connection& connection : system.connections())
        connectionUse.insert(connectionUse.end(), connection.nodes.begin(), connection.nodes.end());
    std::sort(connectionUse.begin(), connectionUse.end());
    connectionUse.erase(std::unique(connectionUse.begin(), connectionUse.end()), connectionUse.end());

    // The one-unit rule: a model of one component and no connection takes the full path whatever the plan.
    const bool isSingleUnit = absorption.units().size() == 1 && absorption.remainingClusters().empty();
    if (isSingleUnit && request.mode != CondenseMode::None)
        m_note = "the model is one unit with no connection: it is solved on the full path whatever the plan";

    for (std::size_t index = 0; index < absorption.units().size(); ++index)
    {
        const AbsorbedUnit& absorbed = absorption.units()[index];
        UnitPlan unit;
        unit.unitId = absorbed.id;
        unit.origin = set.find(absorbed.id)->origin;
        unit.memberUnits = absorbed.memberUnits;
        unit.revision = spiders != nullptr ? spiders->revisionOf(index) : absorbed.revision;
        unit.elementIds = absorbed.elementIds;
        unit.nodeIds = absorbed.nodeIds;
        // The entity nodes of its elements (the IDs of entity nodes are negative: they come first).
        unit.nodeIds.insert(unit.nodeIds.begin(), entityNodes.nodes[index].begin(), entityNodes.nodes[index].end());
        if (spiders != nullptr)
        {
            // The absorbed RigidBodyStar and RigidInterpolation are elements of the unit, their reference points points
            // of it.
            const std::span<const base::EntityId> elements = spiders->absorbedElementIds(index);
            unit.elementIds.insert(unit.elementIds.end(), elements.begin(), elements.end());
            std::sort(unit.elementIds.begin(), unit.elementIds.end());
            unit.elementIds.erase(std::unique(unit.elementIds.begin(), unit.elementIds.end()), unit.elementIds.end());
            const std::span<const base::EntityId> references = spiders->referenceNodes(index);
            unit.nodeIds.insert(unit.nodeIds.end(), references.begin(), references.end());
            std::sort(unit.nodeIds.begin(), unit.nodeIds.end());
            unit.nodeIds.erase(std::unique(unit.nodeIds.begin(), unit.nodeIds.end()), unit.nodeIds.end());
            const std::span<const std::size_t> absorbedSpiders = spiders->absorbedIn(index);
            unit.absorbedSpiders.assign(absorbedSpiders.begin(), absorbedSpiders.end());
        }
        // Before absorption: the points of its own units that any connection uses, and the points O_o_Model puts on
        // their boundaries (a point an element list shares with its neighbour).
        std::vector<base::EntityId> before;
        for (const std::int64_t member : absorbed.memberUnits)
        {
            if (system.isMoved(member))
                continue;
            const std::vector<base::EntityId>& memberBoundary = set.find(member)->boundaryNodeIds;
            before.insert(before.end(), memberBoundary.begin(), memberBoundary.end());
            for (const base::EntityId nodeId : system.unitNodes(member))
                if (std::binary_search(connectionUse.begin(), connectionUse.end(), nodeId))
                    before.push_back(nodeId);
        }
        std::sort(before.begin(), before.end());
        unit.boundaryNodeCountBefore =
            static_cast<std::int64_t>(std::unique(before.begin(), before.end()) - before.begin());
        // The rule of a boundary point some of whose DOFs a relation makes dependent (BoundaryPromotion): it keeps the
        // DOFs left to the unit, and the points a relation eliminated inside the unit makes them depend on come onto
        // the boundary with its reasons (a retained leg is retained through them).
        // Every point of a relation eliminated at the interface that the unit holds is on its boundary, named by
        // O_o_Model or not (BoundaryPromotion::interfaceRelationPoints).
        std::vector<BoundaryNode> candidates(boundaries.at(index).begin(), boundaries.at(index).end());
        for (const base::EntityId nodeId : entityNodes.boundary[index])
            candidates.push_back(BoundaryNode{nodeId, static_cast<std::uint8_t>(BoundaryReason::SharedPoint)});
        for (const base::EntityId nodeId : unit.nodeIds)
            if (std::binary_search(relationPoints.begin(), relationPoints.end(), nodeId))
                candidates.push_back(BoundaryNode{nodeId, static_cast<std::uint8_t>(BoundaryReason::Interface)});
        unit.boundary = BoundaryPromotion::apply(graph, unit.nodeIds, candidates, interfaceRelations, relationPoints);
        for (const BoundaryNode& node : unit.boundary)
            if (node.has(BoundaryReason::Retained))
                unit.retainedNodes.push_back(node.nodeId);
        const std::vector<BoundaryNode>& boundary = unit.boundary;
        unit.absorbedNodes = absorbed.absorbedNodeIds;
        unit.absorbedClusterCount = static_cast<std::int64_t>(absorbed.absorbedClusters.size());

        bool isCondensed = false;
        switch (request.mode)
        {
        case CondenseMode::All:
            isCondensed = true;
            break;
        case CondenseMode::None:
            isCondensed = false;
            break;
        case CondenseMode::Listed:
            isCondensed = std::binary_search(condensed.begin(), condensed.end(), absorbed.id);
            break;
        }
        if (std::binary_search(full.begin(), full.end(), absorbed.id) || isSingleUnit)
            isCondensed = false;
        unit.decision = isCondensed ? UnitDecision::Condensed : UnitDecision::Full;
        unit.reduction = request.reduction;
        unit.modeCount = request.reduction == ReductionKind::CraigBampton ? request.modeCount : 0;

        std::vector<base::EntityId> boundaryIds;
        boundaryIds.reserve(boundary.size());
        for (const BoundaryNode& node : boundary)
            boundaryIds.push_back(node.nodeId);
        unit.cost = UnitCostEstimate::estimate(graph, unit.nodeIds, boundaryIds, interfaceRelations, budget, pool);
        if (request.keepsFullWhereRecommended && unit.cost.isKeptFullRecommended)
            unit.decision = UnitDecision::Full;
        // A unit whose every DOF is held or dependent has nothing to condense (UnitSystem refuses it): kept full.
        if (unit.decision == UnitDecision::Condensed && unit.cost.interiorDofCount + unit.cost.boundaryDofCount == 0)
        {
            unit.decision = UnitDecision::Full;
            m_note = base::concatenate({m_note, m_note.empty() ? "" : "; ", "unit ", std::to_string(unit.unitId),
                                        " is kept full: every DOF of it is held or dependent, nothing to condense"});
        }
        m_units.push_back(std::move(unit));
    }

    for (const std::size_t c : absorption.remainingClusters())
        for (const std::size_t connection : system.clusters()[c].connections)
            m_connections.connectionCount += spiders != nullptr && spiders->isAbsorbedConnection(connection) ? 0 : 1;
    m_connections.absorbedSpiderCount = spiders != nullptr ? spiders->absorbedCount() : 0;
    m_connections.clusterCount = static_cast<std::int64_t>(absorption.remainingClusters().size());
    m_connections.absorbedClusterCount =
        static_cast<std::int64_t>(system.clusters().size()) - m_connections.clusterCount;
    // The connection-only points left: an absorbed spider's reference point is now a point of its unit.
    std::vector<base::EntityId> unitReferences;
    if (spiders != nullptr)
        for (std::size_t index = 0; index < absorption.units().size(); ++index)
        {
            const std::span<const base::EntityId> references = spiders->referenceNodes(index);
            unitReferences.insert(unitReferences.end(), references.begin(), references.end());
        }
    std::sort(unitReferences.begin(), unitReferences.end());
    for (const base::EntityId nodeId : absorption.connectionNodes())
        if (!std::binary_search(unitReferences.begin(), unitReferences.end(), nodeId))
            m_connections.connectionNodeIds.push_back(nodeId);
    for (const std::size_t c : absorption.remainingClusters())
    {
        const std::vector<std::int64_t>& moved = system.clusters()[c].movedUnits;
        m_connections.movedUnits.insert(m_connections.movedUnits.end(), moved.begin(), moved.end());
    }
    std::sort(m_connections.movedUnits.begin(), m_connections.movedUnits.end());

    // The interface problem with these decisions, and the full path (every unit full).
    std::vector<std::vector<base::EntityId>> boundaryIds(m_units.size());
    std::vector<InterfacePart> parts(m_units.size());
    for (std::size_t index = 0; index < m_units.size(); ++index)
    {
        for (const BoundaryNode& node : m_units[index].boundary)
            boundaryIds[index].push_back(node.nodeId);
        parts[index] = InterfacePart{m_units[index].decision == UnitDecision::Condensed, m_units[index].nodeIds,
                                     boundaryIds[index]};
    }
    const std::span<const base::EntityId> connectionNodes = m_connections.connectionNodeIds;
    m_interface = InterfaceEstimate::estimate(graph, parts, connectionNodes, interfaceRelations, budget, pool, true);
    for (InterfacePart& part : parts)
        part.isCondensed = false;
    m_fullPath = InterfaceEstimate::estimate(graph, parts, connectionNodes, interfaceRelations, budget, pool);
}

const UnitPlan* CondensationPlan::find(std::int64_t unitId) const noexcept
{
    const auto found =
        std::find_if(m_units.begin(), m_units.end(), [unitId](const UnitPlan& unit) { return unit.unitId == unitId; });
    return found != m_units.end() ? &*found : nullptr;
}

bool CondensationPlan::isFullPath() const noexcept
{
    return std::none_of(m_units.begin(), m_units.end(),
                        [](const UnitPlan& unit) { return unit.decision == UnitDecision::Condensed; });
}

} // namespace oos::reduction
