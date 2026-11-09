#include "oos/reduction/units/ConnectionSystem.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/units/StampHasher.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

// Union-find with path halving; the smaller index becomes the root, so the roots do not depend on the order of the
// unions.
class DisjointSets
{
public:
    explicit DisjointSets(std::size_t count) : m_parent(count) { std::iota(m_parent.begin(), m_parent.end(), 0); }
    [[nodiscard]] std::size_t find(std::size_t item) noexcept
    {
        while (m_parent[item] != item)
        {
            m_parent[item] = m_parent[m_parent[item]];
            item = m_parent[item];
        }
        return item;
    }
    void join(std::size_t first, std::size_t second) noexcept
    {
        const std::size_t left = find(first);
        const std::size_t right = find(second);
        if (left < right)
            m_parent[right] = left;
        else if (right < left)
            m_parent[left] = right;
    }

private:
    std::vector<std::size_t> m_parent;
};

ConnectionKind kindOf(std::string_view text)
{
    constexpr std::array<std::pair<std::string_view, ConnectionKind>, 7> KINDS{
        {{"rigid", ConnectionKind::Rigid},
         {"rigidInterpolation", ConnectionKind::RigidInterpolation},
         {"mpc", ConnectionKind::Mpc},
         {"fastener", ConnectionKind::Fastener},
         {"connector", ConnectionKind::Connector},
         {"mass", ConnectionKind::Mass},
         {"contact", ConnectionKind::Contact}}};
    for (const auto& [name, kind] : KINDS)
        if (text == name)
            return kind;
    throw base::ContractError(base::concatenate(
        {"connectionList (O_o_Reader's connectionClusters(detail=\"connections\")): expected a "
         "kind among \"rigid\", \"rigidInterpolation\", \"mpc\", \"fastener\", \"connector\", \"mass\", "
         "\"contact\", found \"",
         text, "\"; install a O_o_Reader whose data contract this version reads"}));
}

// The units that hold each point: (point, unit index), sorted.
using NodeOwners = std::vector<std::pair<base::EntityId, std::size_t>>;

std::span<const std::pair<base::EntityId, std::size_t>> ownersOf(const NodeOwners& owners, base::EntityId nodeId)
{
    const auto first = std::lower_bound(owners.begin(), owners.end(), std::make_pair(nodeId, std::size_t{0}));
    auto last = first;
    while (last != owners.end() && last->first == nodeId)
        ++last;
    return {first, last};
}

} // namespace

ConnectionSystem::ConnectionSystem(const UnitSet& units, const base::ContractReader& connectionNodes,
                                   const base::ContractReader& connectionClusters,
                                   std::span<const MpcSetAlias> mpcAliases)
    : m_units(&units)
{
    const std::span<const CondensationUnit> list = units.units();

    // The points of each unit: a group's own list, a component's (or merged components') from componentNodes.
    m_unitNodes.resize(list.size());
    {
        const std::span<const std::int32_t> nodeIds = connectionNodes.values<std::int32_t>("nodeId");
        const std::span<const std::int64_t> components =
            connectionNodes.values<std::int64_t>("componentId", base::NullPolicy::Allow);
        const base::ValidityView isInComponent = connectionNodes.validity("componentId");
        for (std::size_t row = 0; row < nodeIds.size(); ++row)
        {
            if (!isInComponent.isValid(static_cast<std::int64_t>(row)))
                continue;
            const std::optional<std::int64_t> unit = units.unitOfComponent(components[row]);
            if (!unit)
                continue; // a component an element list took elements from: its groups list their points
            const std::size_t index = *units.indexOf(*unit);
            if (list[index].nodeIds.empty())
                m_unitNodes[index].push_back(nodeIds[row]);
        }
        for (std::size_t index = 0; index < list.size(); ++index)
        {
            if (!list[index].nodeIds.empty())
                m_unitNodes[index] = list[index].nodeIds;
            std::vector<base::EntityId>& nodes = m_unitNodes[index];
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        }
    }
    NodeOwners owners;
    for (std::size_t index = 0; index < list.size(); ++index)
        for (const base::EntityId nodeId : m_unitNodes[index])
            owners.emplace_back(nodeId, index);
    std::sort(owners.begin(), owners.end());

    // The connections, less those a group holds as its elements.
    {
        const std::span<const std::int32_t> ids = connectionClusters.values<std::int32_t>("id");
        const base::DictionaryView kinds = connectionClusters.dictionaryTexts("kind");
        const base::DictionaryView cards = connectionClusters.dictionaryTexts("card");
        const base::ListView<std::int32_t> nodes = connectionClusters.lists<std::int32_t>("nodeIds");
        const std::span<const std::int64_t> revisions = connectionClusters.values<std::int64_t>("revision");
        for (std::size_t row = 0; row < ids.size(); ++row)
        {
            const auto at = static_cast<std::int64_t>(row);
            Connection connection;
            connection.kind = kindOf(kinds[at]);
            connection.card = std::string(cards[at]);
            connection.id = ids[row];
            if (connection.kind != ConnectionKind::Mpc && units.unitOfElement(connection.id))
                continue;
            const std::span<const std::int32_t> points = nodes[at];
            connection.nodes.assign(points.begin(), points.end());
            std::sort(connection.nodes.begin(), connection.nodes.end());
            connection.nodes.erase(std::unique(connection.nodes.begin(), connection.nodes.end()),
                                   connection.nodes.end());
            connection.revision = revisions[row];
            if (connection.kind == ConnectionKind::Mpc)
            {
                // The selected sets that hold this card's equations: the same equation under their IDs.
                const auto first =
                    std::lower_bound(mpcAliases.begin(), mpcAliases.end(),
                                     MpcSetAlias{connection.id, std::numeric_limits<base::EntityId>::min()});
                for (auto alias = first; alias != mpcAliases.end() && alias->cardSetId == connection.id; ++alias)
                {
                    Connection aliased = connection;
                    aliased.id = alias->setId;
                    m_connections.push_back(std::move(aliased));
                }
            }
            m_connections.push_back(std::move(connection));
        }
    }

    // The elements the caller declared connections (UnitSet::declaredConnections): O_o_Reader's connection list holds
    // them, as its component queries take the list (their `connections` option). A list without one was taken without
    // the declaration, and the element would be in no unit and no
    // connection: refused.
    for (const DeclaredConnection& declared : units.declaredConnections())
    {
        if (std::none_of(m_connections.begin(), m_connections.end(), [&](const Connection& connection)
                         { return connection.kind != ConnectionKind::Mpc && connection.id == declared.elementId; }))
            throw base::ContractError(base::concatenate(
                {"connectionClusters: element ", std::to_string(declared.elementId), " (", declared.card,
                 ") is declared a connection but the connection list does not hold it; take the unit tables with the "
                 "same connections= option"}));
    }

    // The 1D/0D-only units linked to two or more units move to the connection system, until none moves. Items of the
    // union-find: the connections, then the units.
    const std::size_t connectionCount = m_connections.size();
    std::vector<bool> isMoved(list.size(), false);
    const auto clusterRoots = [&]
    {
        DisjointSets sets(connectionCount + list.size());
        std::map<base::EntityId, std::size_t> firstUser; // connection-only point -> first connection using it
        for (std::size_t c = 0; c < connectionCount; ++c)
        {
            for (const base::EntityId nodeId : m_connections[c].nodes)
            {
                const auto nodeOwners = ownersOf(owners, nodeId);
                if (nodeOwners.empty())
                {
                    const auto [found, isNew] = firstUser.emplace(nodeId, c);
                    if (!isNew)
                        sets.join(c, found->second);
                }
                for (const auto& [node, unit] : nodeOwners)
                    if (isMoved[unit])
                        sets.join(c, connectionCount + unit);
            }
        }
        std::vector<std::size_t> roots(connectionCount + list.size());
        for (std::size_t item = 0; item < roots.size(); ++item)
            roots[item] = sets.find(item);
        return roots;
    };
    std::vector<std::size_t> roots = clusterRoots();
    for (;;)
    {
        // The units (not moved) each cluster touches.
        std::map<std::size_t, std::vector<std::size_t>> touched;
        for (std::size_t c = 0; c < connectionCount; ++c)
            for (const base::EntityId nodeId : m_connections[c].nodes)
                for (const auto& [node, unit] : ownersOf(owners, nodeId))
                    if (!isMoved[unit])
                        touched[roots[c]].push_back(unit);
        std::vector<std::vector<std::size_t>> linked(list.size());
        for (auto& [root, touchedUnits] : touched)
        {
            std::sort(touchedUnits.begin(), touchedUnits.end());
            touchedUnits.erase(std::unique(touchedUnits.begin(), touchedUnits.end()), touchedUnits.end());
            for (const std::size_t unit : touchedUnits)
                linked[unit].insert(linked[unit].end(), touchedUnits.begin(), touchedUnits.end());
        }
        bool hasMoved = false;
        for (std::size_t unit = 0; unit < list.size(); ++unit)
        {
            if (isMoved[unit] || !list[unit].isLineOrPointOnly)
                continue;
            std::vector<std::size_t>& others = linked[unit];
            std::sort(others.begin(), others.end());
            others.erase(std::unique(others.begin(), others.end()), others.end());
            const auto self = std::find(others.begin(), others.end(), unit);
            const std::size_t otherCount = others.size() - (self != others.end() ? 1 : 0);
            if (otherCount >= 2)
            {
                isMoved[unit] = true;
                hasMoved = true;
            }
        }
        if (!hasMoved)
            break;
        roots = clusterRoots();
    }
    for (std::size_t unit = 0; unit < list.size(); ++unit)
    {
        if (isMoved[unit])
            m_movedUnits.push_back(list[unit].id);
        else
            for (const base::EntityId nodeId : m_unitNodes[unit])
                m_nodeUnits.emplace_back(nodeId, list[unit].id);
    }
    std::sort(m_nodeUnits.begin(), m_nodeUnits.end());

    // The clusters, in the order of their first connection.
    std::map<std::size_t, std::size_t> clusterOfRoot;
    m_clusterOf.resize(connectionCount);
    for (std::size_t c = 0; c < connectionCount; ++c)
    {
        const auto [found, isNew] = clusterOfRoot.emplace(roots[c], m_clusters.size());
        if (isNew)
            m_clusters.emplace_back();
        m_clusterOf[c] = found->second;
        m_clusters[found->second].connections.push_back(c);
    }
    for (std::size_t unit = 0; unit < list.size(); ++unit)
    {
        if (!isMoved[unit])
            continue;
        const auto found = clusterOfRoot.find(roots[connectionCount + unit]);
        if (found != clusterOfRoot.end())
            m_clusters[found->second].movedUnits.push_back(list[unit].id);
    }
    for (ConnectionCluster& cluster : m_clusters)
    {
        StampHasher hasher;
        for (const std::size_t c : cluster.connections)
        {
            const Connection& connection = m_connections[c];
            hasher.mix(connection.revision);
            for (const base::EntityId nodeId : connection.nodes)
            {
                bool isInUnit = false;
                for (const auto& [node, unit] : ownersOf(owners, nodeId))
                {
                    if (isMoved[unit])
                        continue;
                    isInUnit = true;
                    cluster.touchedUnits.push_back(list[unit].id);
                }
                if (!isInUnit)
                    cluster.connectionOnlyNodes.push_back(nodeId);
            }
        }
        for (const std::int64_t moved : cluster.movedUnits)
        {
            hasher.mix(units.find(moved)->revision);
            const std::span<const base::EntityId> nodes = unitNodes(moved);
            cluster.connectionOnlyNodes.insert(cluster.connectionOnlyNodes.end(), nodes.begin(), nodes.end());
        }
        std::sort(cluster.connectionOnlyNodes.begin(), cluster.connectionOnlyNodes.end());
        cluster.connectionOnlyNodes.erase(
            std::unique(cluster.connectionOnlyNodes.begin(), cluster.connectionOnlyNodes.end()),
            cluster.connectionOnlyNodes.end());
        std::sort(cluster.touchedUnits.begin(), cluster.touchedUnits.end());
        cluster.touchedUnits.erase(std::unique(cluster.touchedUnits.begin(), cluster.touchedUnits.end()),
                                   cluster.touchedUnits.end());
        cluster.stamp = hasher.stamp();
    }
}

bool ConnectionSystem::isMoved(std::int64_t unitId) const noexcept
{
    return std::binary_search(m_movedUnits.begin(), m_movedUnits.end(), unitId);
}

std::span<const base::EntityId> ConnectionSystem::unitNodes(std::int64_t unitId) const noexcept
{
    const std::optional<std::size_t> index = m_units->indexOf(unitId);
    if (!index)
        return {};
    return m_unitNodes[*index];
}

std::vector<std::int64_t> ConnectionSystem::unitsOfNode(base::EntityId nodeId) const
{
    std::vector<std::int64_t> result;
    auto found = std::lower_bound(m_nodeUnits.begin(), m_nodeUnits.end(),
                                  std::make_pair(nodeId, std::numeric_limits<std::int64_t>::min()));
    for (; found != m_nodeUnits.end() && found->first == nodeId; ++found)
        result.push_back(found->second);
    return result;
}

} // namespace oos::reduction
