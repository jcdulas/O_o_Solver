#include "oos/reduction/absorption/SpiderAbsorption.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/units/StampHasher.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

namespace
{

constexpr std::size_t NO_CONNECTION = std::numeric_limits<std::size_t>::max();

template <typename T>
void sortAndMakeUnique(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

const char* connectionKindOf(SpiderKind kind)
{
    switch (kind)
    {
    case SpiderKind::RigidInterpolation:
        return "rigidInterpolation";
    case SpiderKind::RigidBodyStar:
        return "rigid";
    case SpiderKind::FastenerSide:
        return "fastener";
    }
    return "";
}

// The connections that touch the concerned legs, the spider itself left out (connectionSharing lists it too).
std::vector<BlockingConnection> blockingOf(const Spider& spider, const std::vector<base::EntityId>& concerned,
                                           const SharedConnectionNodes* sharing)
{
    std::vector<BlockingConnection> blocking;
    if (sharing == nullptr || concerned.empty())
    {
        for (const base::EntityId id : spider.blockingConnections)
            blocking.push_back(BlockingConnection{0, {}, {}, id});
        return blocking;
    }
    for (const base::EntityId leg : concerned)
        for (const SharedConnection& connection : sharing->connectionsAt(leg))
        {
            const bool isItself = connection.kind != "mpc" && connection.id == spider.key.elementId;
            if (!isItself)
                blocking.push_back(BlockingConnection{leg, connection.kind, connection.card, connection.id});
        }
    return blocking;
}

} // namespace

SpiderAbsorption::SpiderAbsorption(const ClusterAbsorption& absorption, const SpiderCatalog& catalog,
                                   std::span<const SpiderMeasurement> measurements,
                                   const AbsorptionExclusions& exclusions, const SharedConnectionNodes* sharing,
                                   bool absorbsSpiders, double maxCondition)
    : m_clusters(&absorption), m_hasRigidSpiders(catalog.hasRigidSpiders())
{
    const std::span<const Spider> spiders = catalog.spiders();
    if (measurements.size() != spiders.size())
        throw std::invalid_argument(
            base::concatenate({"SpiderAbsorption: expected one measurement per spider (",
                               std::to_string(spiders.size()), "), found ", std::to_string(measurements.size())}));
    const ConnectionSystem& system = absorption.system();
    const std::span<const AbsorbedUnit> units = absorption.units();
    m_absorbed.resize(units.size());
    m_referenceNodes.resize(units.size());
    m_absorbedElements.resize(units.size());
    m_revisions.resize(units.size());

    // The connections by element ID (an MPC equation is no spider), and the unit each absorbed cluster went into.
    std::vector<std::pair<base::EntityId, std::size_t>> connectionOf;
    for (std::size_t c = 0; c < system.connections().size(); ++c)
        if (system.connections()[c].kind != ConnectionKind::Mpc)
            connectionOf.emplace_back(system.connections()[c].id, c);
    std::sort(connectionOf.begin(), connectionOf.end());
    std::vector<std::int64_t> unitOfCluster(system.clusters().size(), 0);
    for (const AbsorbedUnit& unit : units)
        for (const std::size_t c : unit.absorbedClusters)
            unitOfCluster[c] = unit.id;
    const std::span<const std::size_t> remaining = absorption.remainingClusters();
    const auto unitIndexOf = [&](std::int64_t id) -> std::optional<std::size_t>
    {
        const auto found = std::lower_bound(units.begin(), units.end(), id,
                                            [](const AbsorbedUnit& unit, std::int64_t key) { return unit.id < key; });
        if (found == units.end() || found->id != id)
            return std::nullopt;
        return static_cast<std::size_t>(found - units.begin());
    };
    // The unit (index) holding every leg of a spider, or nothing when they are on several units or one on none.
    const auto unitOfLegs = [&](const Spider& spider) -> std::optional<std::size_t>
    {
        std::optional<std::size_t> unit;
        for (const base::EntityId leg : spider.legNodeIds)
        {
            const std::vector<std::int64_t> holders = system.unitsOfNode(leg);
            if (holders.size() != 1)
                return std::nullopt;
            const std::optional<std::int64_t> member = absorption.unitOfMember(holders.front());
            const std::optional<std::size_t> index = member ? unitIndexOf(*member) : std::nullopt;
            if (!index || (unit && *unit != *index))
                return std::nullopt;
            unit = index;
        }
        return unit;
    };

    std::vector<std::size_t> connectionIndex(spiders.size(), NO_CONNECTION);
    std::vector<std::size_t> unitIndex(spiders.size(), 0);
    m_decisions.reserve(spiders.size());
    for (std::size_t k = 0; k < spiders.size(); ++k)
    {
        const Spider& spider = spiders[k];
        const SpiderMeasurement& measurement = measurements[k];
        SpiderDecision decision;
        decision.key = spider.key;
        decision.kind = spider.kind;
        decision.card = spider.card;
        decision.referenceNodeId = spider.referenceNodeId;
        decision.legNodeIds = spider.legNodeIds;
        decision.revision = spider.revision;
        decision.conditionNumber = measurement.conditionNumber;
        const auto decide = [&]() -> AbsorptionDecision
        {
            const auto found = std::lower_bound(connectionOf.begin(), connectionOf.end(),
                                                std::make_pair(spider.key.elementId, std::size_t{0}));
            if (found == connectionOf.end() || found->first != spider.key.elementId)
            {
                if (const std::optional<std::int64_t> owner = system.units().unitOfElement(spider.key.elementId))
                {
                    decision.unitId = absorption.unitOfMember(*owner).value_or(*owner);
                    return AbsorptionDecision::PartOfUnit;
                }
                return AbsorptionDecision::NotConnection;
            }
            connectionIndex[k] = found->second;
            const std::size_t cluster = system.clusterOf(found->second);
            if (!std::binary_search(remaining.begin(), remaining.end(), cluster))
            {
                decision.unitId = unitOfCluster[cluster];
                return AbsorptionDecision::AbsorbedWithCluster;
            }
            if (!spider.isSpider)
            {
                decision.detail = spider.notSpiderReason;
                return AbsorptionDecision::NotSpider;
            }
            const std::optional<std::size_t> unit = unitOfLegs(spider);
            if (!unit)
                return AbsorptionDecision::LegsOnSeveralUnits;
            unitIndex[k] = *unit;
            decision.unitId = units[*unit].id;
            const AbsorptionDecision topological =
                AbsorptionEligibility::topological(spider, decision.concernedNodeIds);
            if (topological != AbsorptionDecision::Absorbed)
            {
                if (topological == AbsorptionDecision::LegTouched)
                {
                    // A fastener side's rows name the blocking connections, not the legs: the legs come from the
                    // connections at each point.
                    if (decision.concernedNodeIds.empty() && sharing != nullptr)
                        for (const base::EntityId leg : spider.legNodeIds)
                            if (!blockingOf(spider, {leg}, sharing).empty())
                                decision.concernedNodeIds.push_back(leg);
                    decision.blockingConnections = blockingOf(spider, decision.concernedNodeIds, sharing);
                }
                return topological;
            }
            const AbsorptionDecision glued = AbsorptionEligibility::glued(spider, exclusions, decision.concernedNodeIds,
                                                                          decision.blockingConnections);
            if (glued != AbsorptionDecision::Absorbed)
                return glued;
            const AbsorptionDecision analysis =
                AbsorptionEligibility::analysis(spider, exclusions, measurement, decision.concernedNodeIds);
            if (analysis != AbsorptionDecision::Absorbed)
                return analysis;
            if (!absorbsSpiders)
                return AbsorptionDecision::NotRequested;
            return AbsorptionEligibility::numerical(measurement, maxCondition);
        };
        decision.decision = decide();
        if (decision.decision == AbsorptionDecision::Absorbed)
            decision.boundaryNodeId =
                spider.kind == SpiderKind::FastenerSide ? spider.legNodeIds.front() : spider.referenceNodeId;
        m_decisions.push_back(std::move(decision));
    }

    // The last check: absorbed spiders sharing a leg, or one's reference on the other's leg, are kept on their legs.
    std::map<base::EntityId, std::vector<std::size_t>> owners; // leg -> absorbed spiders; bounded by the legs
    for (std::size_t k = 0; k < m_decisions.size(); ++k)
        if (m_decisions[k].decision == AbsorptionDecision::Absorbed)
            for (const base::EntityId leg : m_decisions[k].legNodeIds)
                owners[leg].push_back(k);
    std::vector<std::pair<std::size_t, std::size_t>> conflicts; // (kept on its legs, because of)
    for (const auto& [leg, list] : owners)
        if (list.size() > 1)
            for (const std::size_t k : list)
                for (const std::size_t other : list)
                    if (other != k)
                        conflicts.emplace_back(k, other);
    for (std::size_t k = 0; k < m_decisions.size(); ++k)
    {
        if (m_decisions[k].decision != AbsorptionDecision::Absorbed || m_decisions[k].kind == SpiderKind::FastenerSide)
            continue;
        if (const auto found = owners.find(m_decisions[k].referenceNodeId); found != owners.end())
            for (const std::size_t other : found->second)
                if (other != k)
                    conflicts.emplace_back(other, k);
    }
    for (const auto& [k, other] : conflicts)
    {
        SpiderDecision& decision = m_decisions[k];
        const SpiderDecision& cause = m_decisions[other];
        std::vector<base::EntityId> shared;
        std::set_intersection(decision.legNodeIds.begin(), decision.legNodeIds.end(), cause.legNodeIds.begin(),
                              cause.legNodeIds.end(), std::back_inserter(shared));
        if (std::binary_search(decision.legNodeIds.begin(), decision.legNodeIds.end(), cause.referenceNodeId))
            shared.push_back(cause.referenceNodeId);
        decision.decision = AbsorptionDecision::LegTouched;
        decision.boundaryNodeId = 0;
        for (const base::EntityId leg : shared)
        {
            decision.concernedNodeIds.push_back(leg);
            decision.blockingConnections.push_back(
                BlockingConnection{leg, connectionKindOf(cause.kind), cause.card, cause.key.elementId});
        }
        sortAndMakeUnique(decision.concernedNodeIds);
    }

    // What the absorbed spiders change, unit by unit and connection by connection.
    std::map<std::size_t, std::vector<base::EntityId>> removed;
    for (std::size_t k = 0; k < m_decisions.size(); ++k)
    {
        const SpiderDecision& decision = m_decisions[k];
        if (decision.decision != AbsorptionDecision::Absorbed)
            continue;
        ++m_absorbedCount;
        const std::size_t u = unitIndex[k];
        const std::size_t c = connectionIndex[k];
        m_absorbed[u].push_back(k);
        m_referenceNodes[u].push_back(decision.boundaryNodeId);
        std::vector<base::EntityId>& uses = removed[c];
        if (decision.kind == SpiderKind::FastenerSide)
        {
            uses.insert(uses.end(), decision.legNodeIds.begin(), decision.legNodeIds.end());
            if (decision.referenceNodeId != 0)
                m_freedReferences.push_back(FreedReference{decision.referenceNodeId, decision.key.elementId, false});
            continue;
        }
        m_absorbedElements[u].push_back(decision.key.elementId);
        m_absorbedConnections.push_back(c);
        const std::vector<base::EntityId>& nodes = system.connections()[c].nodes;
        uses.insert(uses.end(), nodes.begin(), nodes.end());
        if (decision.kind == SpiderKind::RigidInterpolation)
            m_freedReferences.push_back(FreedReference{decision.referenceNodeId, decision.key.elementId, true});
    }
    for (auto& [c, uses] : removed)
    {
        sortAndMakeUnique(uses);
        m_removedUses.emplace_back(c, std::move(uses));
    }
    sortAndMakeUnique(m_absorbedConnections);
    std::sort(m_freedReferences.begin(), m_freedReferences.end(), [](const FreedReference& a, const FreedReference& b)
              { return a.nodeId != b.nodeId ? a.nodeId < b.nodeId : a.elementId < b.elementId; });
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        sortAndMakeUnique(m_referenceNodes[u]);
        sortAndMakeUnique(m_absorbedElements[u]);
        if (m_absorbed[u].empty())
        {
            m_revisions[u] = units[u].revision;
            continue;
        }
        // The unit's stamp, then each absorbed spider in key order: its ID, side and revision.
        StampHasher hasher;
        hasher.mix(units[u].revision);
        for (const std::size_t k : m_absorbed[u])
        {
            hasher.mix(static_cast<std::int64_t>(m_decisions[k].key.elementId));
            hasher.mix(static_cast<std::int64_t>(m_decisions[k].key.side));
            hasher.mix(m_decisions[k].revision);
        }
        m_revisions[u] = hasher.stamp();
    }
}

std::span<const base::EntityId> SpiderAbsorption::removedUses(std::size_t connection) const noexcept
{
    using Entry = std::pair<std::size_t, std::vector<base::EntityId>>;
    const auto found = std::lower_bound(m_removedUses.begin(), m_removedUses.end(), connection,
                                        [](const Entry& entry, std::size_t c) { return entry.first < c; });
    if (found == m_removedUses.end() || found->first != connection)
        return {};
    return found->second;
}

bool SpiderAbsorption::isAbsorbedConnection(std::size_t connection) const noexcept
{
    return std::binary_search(m_absorbedConnections.begin(), m_absorbedConnections.end(), connection);
}

} // namespace oos::reduction
