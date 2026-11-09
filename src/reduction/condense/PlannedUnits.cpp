#include "oos/reduction/condense/PlannedUnits.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/absorption/FastenerPiercing.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <variant>

namespace oos::reduction
{

namespace
{

constexpr std::int32_t NO_UNIT = -1;

std::int32_t nodeOf(const assembly::DofMap& dofMap, base::EntityId nodeId)
{
    const std::optional<input::PointRef> point = dofMap.points().find(nodeId);
    if (!point)
        throw std::invalid_argument(base::concatenate({"PlannedUnits: point ", std::to_string(nodeId),
                                                       " of the plan is not a point of the model; the plan and the "
                                                       "model must come from one snapshot"}));
    return point->kind == input::PointKind::Grid ? point->row : dofMap.gridCount() + point->row;
}

bool isFastenerSource(assembly::ConstraintSource source) noexcept
{
    return source == assembly::ConstraintSource::Fastener || source == assembly::ConstraintSource::Weld;
}

bool isGlueSource(assembly::ConstraintSource source) noexcept
{
    return source == assembly::ConstraintSource::ContactTable || source == assembly::ConstraintSource::ContactPair;
}

std::string sideText(const SpiderKey& key)
{
    return base::concatenate(
        {"fastener ", std::to_string(key.elementId), key.side == FastenerSide::A ? " side A" : " side B"});
}

// The points and the boundary of a unit as its description has them: the plan's, the stand-ins of its generated
// fastener points dropped from the boundary when nothing else keeps them there, the given GA (GB) of its absorbed
// fastener sides added to both, its points the interface reaches (`promoted`) added to the boundary. Both sorted.
struct UnitPoints
{
    std::vector<base::EntityId> nodes;
    std::vector<base::EntityId> boundary;
};

UnitPoints pointsOf(const CondensationPlan& plan, const UnitPlan& unit, std::span<const base::EntityId> promoted)
{
    const std::span<const SpiderDecision> decisions = plan.spiderDecisions();
    std::vector<base::EntityId> standIns;   // bounded by the unit's spiders
    std::vector<base::EntityId> references; // the references of its absorbed RigidBodyStar and RigidInterpolation
    UnitPoints points{unit.nodeIds, {}};
    for (const std::size_t k : unit.absorbedSpiders)
    {
        const SpiderDecision& decision = decisions[k];
        if (decision.kind != SpiderKind::FastenerSide)
        {
            references.push_back(decision.referenceNodeId);
            continue;
        }
        standIns.push_back(decision.boundaryNodeId);
        if (decision.referenceNodeId != 0)
        {
            points.nodes.push_back(decision.referenceNodeId);
            points.boundary.push_back(decision.referenceNodeId);
        }
    }
    std::sort(standIns.begin(), standIns.end());
    std::sort(references.begin(), references.end());
    constexpr auto SPIDER_ONLY = static_cast<std::uint8_t>(BoundaryReason::AbsorbedSpider);
    for (const BoundaryNode& node : unit.boundary)
    {
        // A stand-in with no reason of its own besides standing for a fastener point is interior (the
        // generated point stands on the boundary in its place).
        const bool isStandInOnly = node.reasons == SPIDER_ONLY &&
                                   std::binary_search(standIns.begin(), standIns.end(), node.nodeId) &&
                                   !std::binary_search(references.begin(), references.end(), node.nodeId);
        if (!isStandInOnly)
            points.boundary.push_back(node.nodeId);
    }
    points.boundary.insert(points.boundary.end(), promoted.begin(), promoted.end());
    for (std::vector<base::EntityId>* list : {&points.nodes, &points.boundary})
    {
        std::sort(list->begin(), list->end());
        list->erase(std::unique(list->begin(), list->end()), list->end());
    }
    return points;
}

} // namespace

PlannedUnits::PlannedUnits(const CondensationPlan& plan, const assembly::DofMap& dofMap, base::MemoryBudget& budget,
                           std::span<const base::EntityId> reachedPoints)
    : m_plan(&plan), m_dofMap(&dofMap)
{
    const std::span<const UnitPlan> units = plan.units();
    const std::span<const SpiderDecision> decisions = plan.spiderDecisions();
    // The points the interface reaches inside each condensed unit, beyond its plan's boundary (not the legs of its
    // absorbed fastener sides, which their fastener reaches through the change of basis).
    m_promoted.resize(units.size());
    for (std::size_t index = 0; index < units.size(); ++index)
    {
        const UnitPlan& unit = units[index];
        if (unit.decision != UnitDecision::Condensed)
            continue;
        std::vector<base::EntityId> legs; // bounded by the unit's spiders
        for (const std::size_t k : unit.absorbedSpiders)
            if (decisions[k].kind == SpiderKind::FastenerSide)
                legs.insert(legs.end(), decisions[k].legNodeIds.begin(), decisions[k].legNodeIds.end());
        std::sort(legs.begin(), legs.end());
        const UnitPoints planned = pointsOf(plan, unit, {});
        for (const base::EntityId nodeId : reachedPoints)
            if (std::binary_search(planned.nodes.begin(), planned.nodes.end(), nodeId) &&
                !std::binary_search(planned.boundary.begin(), planned.boundary.end(), nodeId) &&
                !std::binary_search(legs.begin(), legs.end(), nodeId))
                m_promoted[index].push_back(nodeId);
    }
    m_relations = plan.connectionSystem().relations;
    std::sort(m_relations.begin(), m_relations.end());
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    m_isInteriorNode = base::BudgetedBuffer<std::uint8_t>(budget, nodeCount, "plan interior points");
    m_unitOfNode = base::BudgetedBuffer<std::int32_t>(budget, nodeCount, "plan interior points");
    std::fill(m_unitOfNode.data(), m_unitOfNode.data() + nodeCount, NO_UNIT);
    std::size_t elementCount = 0;
    for (std::size_t index = 0; index < units.size(); ++index)
    {
        const UnitPlan& unit = units[index];
        for (const std::size_t k : unit.absorbedSpiders)
        {
            const SpiderDecision& decision = decisions[k];
            if (decision.kind == SpiderKind::FastenerSide && decision.referenceNodeId != 0)
                m_givenSides.emplace_back(decision.key.elementId, decision.referenceNodeId);
        }
        if (unit.decision != UnitDecision::Condensed)
            continue;
        m_condensed.push_back(index);
        elementCount += unit.elementIds.size();
        for (const std::size_t k : unit.absorbedSpiders)
        {
            const SpiderDecision& decision = decisions[k];
            if (decision.kind == SpiderKind::RigidInterpolation)
                m_condensedRigidInterpolations.push_back(decision.key.elementId);
            else if (decision.kind == SpiderKind::FastenerSide && decision.referenceNodeId != 0)
                m_condensedGivenSides.emplace_back(decision.key.elementId, decision.referenceNodeId);
        }
    }
    std::sort(m_givenSides.begin(), m_givenSides.end());
    std::sort(m_condensedGivenSides.begin(), m_condensedGivenSides.end());
    std::sort(m_condensedRigidInterpolations.begin(), m_condensedRigidInterpolations.end());
    m_condensedElements = base::BudgetedBuffer<base::EntityId>(budget, elementCount, "plan condensed elements");
    std::size_t nextElement = 0;
    for (const std::size_t index : m_condensed)
    {
        const UnitPlan& unit = units[index];
        for (const base::EntityId elementId : unit.elementIds)
            m_condensedElements[nextElement++] = elementId;
        const UnitPoints points = pointsOf(plan, unit, m_promoted[index]);
        for (const base::EntityId nodeId : points.nodes)
        {
            if (std::binary_search(points.boundary.begin(), points.boundary.end(), nodeId))
                continue;
            const auto node = static_cast<std::size_t>(nodeOf(dofMap, nodeId));
            if (m_unitOfNode[node] != NO_UNIT)
                throw std::invalid_argument(
                    base::concatenate({"PlannedUnits: point ", std::to_string(nodeId), " is interior to units ",
                                       std::to_string(units[static_cast<std::size_t>(m_unitOfNode[node])].unitId),
                                       " and ", std::to_string(unit.unitId),
                                       "; a point two units hold is on both boundaries (an internal "
                                       "error of the plan: report it)"}));
            m_isInteriorNode[node] = 1;
            m_unitOfNode[node] = static_cast<std::int32_t>(index);
        }
    }
    std::sort(m_condensedElements.data(), m_condensedElements.data() + elementCount);
}

RelationKey PlannedUnits::keyOf(const assembly::ConstraintRowOrigin& origin) const
{
    if (isGlueSource(origin.source))
        return glueRelationKey(m_dofMap->pointIdOf(origin.dependent));
    return RelationKey{origin.source == assembly::ConstraintSource::Mpc, origin.id};
}

bool PlannedUnits::isGivenSideRow(const assembly::ConstraintRowOrigin& origin, bool condensedOnly) const
{
    if (!isFastenerSource(origin.source))
        return false;
    const std::vector<std::pair<base::EntityId, base::EntityId>>& sides =
        condensedOnly ? m_condensedGivenSides : m_givenSides;
    return std::binary_search(sides.begin(), sides.end(),
                              std::make_pair(origin.id, m_dofMap->pointIdOf(origin.dependent)));
}

assembly::ConstraintRowFilter PlannedUnits::unitRowFilter() const
{
    return [this](const assembly::ConstraintRowOrigin& origin)
    {
        const RelationKey key = keyOf(origin);
        return !std::binary_search(m_relations.begin(), m_relations.end(), key) || isGivenSideRow(origin, false);
    };
}

assembly::ConstraintRowFilter PlannedUnits::interfaceRowFilter() const
{
    return [this](const assembly::ConstraintRowOrigin& origin)
    {
        const bool isUnitRelation =
            !std::binary_search(m_relations.begin(), m_relations.end(), keyOf(origin)) || isGivenSideRow(origin, false);
        if (!isUnitRelation)
            return true;
        // A row a condensed unit eliminates inside it: its dependent DOF is an interior point of the unit, or the
        // reference DOF of a spider absorbed by change of basis (an unknown of the interface).
        const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(origin.dependent);
        if (m_isInteriorNode[static_cast<std::size_t>(nodeDof.node)] != 0)
            return false;
        const bool isReferenceRow = (origin.source == assembly::ConstraintSource::RigidInterpolation &&
                                     std::binary_search(m_condensedRigidInterpolations.begin(),
                                                        m_condensedRigidInterpolations.end(), origin.id)) ||
                                    isGivenSideRow(origin, true);
        return !isReferenceRow;
    };
}

std::span<const base::EntityId> PlannedUnits::promotedPoints(std::size_t unit) const noexcept
{
    return m_promoted[unit];
}

bool PlannedUnits::isCondensedElement(base::EntityId elementId) const noexcept
{
    return std::binary_search(m_condensedElements.data(), m_condensedElements.data() + m_condensedElements.size(),
                              elementId);
}

std::optional<std::size_t> PlannedUnits::unitOfInteriorNode(std::int32_t node) const noexcept
{
    const std::int32_t unit = m_unitOfNode[static_cast<std::size_t>(node)];
    if (unit == NO_UNIT)
        return std::nullopt;
    return static_cast<std::size_t>(unit);
}

UnitDescription PlannedUnits::describe(std::size_t unitIndex, const SpiderWeights& unitRelations,
                                       const input::FastenerPiercingTable* piercings) const
{
    const UnitPlan& unit = m_plan->units()[unitIndex];
    const std::span<const SpiderDecision> decisions = m_plan->spiderDecisions();
    UnitPoints points = pointsOf(*m_plan, unit, m_promoted[unitIndex]);
    UnitDescription description;
    description.unitId = unit.unitId;
    description.nodeIds = std::move(points.nodes);
    description.elementIds = unit.elementIds;
    description.boundaryNodeIds = std::move(points.boundary);
    for (const std::size_t k : unit.absorbedSpiders)
    {
        const SpiderDecision& decision = decisions[k];
        UnitSpider spider;
        spider.key = decision.key;
        switch (decision.kind)
        {
        case SpiderKind::RigidBodyStar:
            continue; // eliminated by its own rows of the units' Gm
        case SpiderKind::RigidInterpolation:
        {
            const std::optional<SpiderRelation> relation =
                unitRelations.rigidInterpolationRelation(decision.key.elementId);
            if (!relation)
                throw std::invalid_argument(
                    base::concatenate({"PlannedUnits: the RigidInterpolation ", std::to_string(decision.key.elementId),
                                       " absorbed into unit ", std::to_string(unit.unitId),
                                       " has no rows in the Gm of the units; give the Gm made with unitRowFilter()"}));
            spider.relation = *relation;
            break;
        }
        case SpiderKind::FastenerSide:
        {
            if (decision.referenceNodeId != 0)
            {
                const assembly::ConstraintSource source = decision.card == "Weld"
                                                              ? assembly::ConstraintSource::Weld
                                                              : assembly::ConstraintSource::Fastener;
                const std::optional<SpiderRelation> relation =
                    unitRelations.fastenerPointRelation(decision.key.elementId, source, decision.referenceNodeId);
                if (!relation)
                    throw std::invalid_argument(base::concatenate(
                        {"PlannedUnits: ", sideText(decision.key), " absorbed into unit ", std::to_string(unit.unitId),
                         " has no rows of its point ", std::to_string(decision.referenceNodeId),
                         " in the Gm of the units; give the Gm made with unitRowFilter() and the fastener "
                         "constraints"}));
                spider.relation = *relation;
                break;
            }
            if (piercings == nullptr)
                throw std::invalid_argument(
                    base::concatenate({"PlannedUnits: ", sideText(decision.key), " absorbed into unit ",
                                       std::to_string(unit.unitId), " needs the fastener piercing table; give it"}));
            std::variant<SpiderRelation, std::string> relation =
                FastenerPiercing::relation(*piercings, *m_dofMap, decision.key.elementId, decision.key.side);
            if (const std::string* problem = std::get_if<std::string>(&relation))
                throw std::invalid_argument(
                    base::concatenate({"PlannedUnits: ", sideText(decision.key), " absorbed into unit ",
                                       std::to_string(unit.unitId), ": ", *problem,
                                       "; the plan absorbs only sides whose pierced element interpolates: a bug of "
                                       "O_o_Solver, report it"}));
            spider.relation = std::move(std::get<SpiderRelation>(relation));
            spider.relation.referenceDofs.clear();
            // The piercing point of the side (the piercing table's, the same on every row of the side).
            const std::optional<input::FastenerPiercingRows> rows = piercings->findFastener(decision.key.elementId);
            const input::PiercingRows& side = decision.key.side == FastenerSide::A ? rows->sideA : rows->sideB;
            const std::span<const double> piercing = piercings->piercingPoints()[side.first];
            spider.generatedPointPosition = base::Vec3{piercing[0], piercing[1], piercing[2]};
            break;
        }
        }
        description.spiders.push_back(std::move(spider));
    }
    return description;
}

} // namespace oos::reduction
