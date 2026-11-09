#include "oos/reduction/units/ClusterAbsorption.hpp"

#include "oos/reduction/units/StampHasher.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

namespace oos::reduction
{

namespace
{

template <typename T>
void sortAndMakeUnique(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Whether a cluster may be absorbed: it holds no equation of an MPC set that varies between subcases.
bool isAbsorbable(const ConnectionSystem& system, const ConnectionCluster& cluster, const AbsorptionOptions& options)
{
    for (const std::size_t c : cluster.connections)
    {
        const Connection& connection = system.connections()[c];
        if (connection.kind == ConnectionKind::Mpc &&
            std::binary_search(options.varyingMpcSets.begin(), options.varyingMpcSets.end(), connection.id))
            return false;
    }
    return true;
}

} // namespace

ClusterAbsorption::ClusterAbsorption(const ConnectionSystem& system, const AbsorptionOptions& options)
    : m_system(&system)
{
    const UnitSet& set = system.units();
    const std::span<const CondensationUnit> list = set.units();
    const std::span<const ConnectionCluster> clusters = system.clusters();

    // Step 1: each unit index points to the unit index it is absorbed into (itself when none), resolved below.
    std::vector<std::size_t> target(list.size());
    std::iota(target.begin(), target.end(), 0);
    const auto indexOf = [&](std::int64_t id) { return *set.indexOf(id); };
    if (options.absorbsClusters)
    {
        std::vector<std::vector<std::size_t>> linked(list.size());
        for (const ConnectionCluster& cluster : clusters)
            for (const std::int64_t unit : cluster.touchedUnits)
                for (const std::int64_t other : cluster.touchedUnits)
                    if (other != unit)
                        linked[indexOf(unit)].push_back(indexOf(other));
        // Union-find over the units: a 1D/0D-only unit joins the one unit it is linked to. The root of a set is the
        // structural unit it holds (at most one: a 1D/0D-only unit linked to two units was moved), else its smallest.
        std::vector<std::size_t> parent(list.size());
        std::iota(parent.begin(), parent.end(), 0);
        const auto find = [&](std::size_t item)
        {
            while (parent[item] != item)
                item = parent[item] = parent[parent[item]];
            return item;
        };
        const auto join = [&](std::size_t a, std::size_t b)
        {
            a = find(a);
            b = find(b);
            if (a == b)
                return;
            const bool aFirst =
                list[a].isLineOrPointOnly != list[b].isLineOrPointOnly ? !list[a].isLineOrPointOnly : a < b;
            if (aFirst)
                parent[b] = a;
            else
                parent[a] = b;
        };
        for (std::size_t unit = 0; unit < list.size(); ++unit)
        {
            if (!list[unit].isLineOrPointOnly || system.isMoved(list[unit].id))
                continue;
            sortAndMakeUnique(linked[unit]);
            if (linked[unit].size() == 1)
                join(unit, linked[unit].front());
        }
        for (std::size_t unit = 0; unit < list.size(); ++unit)
            target[unit] = find(unit);
    }

    // Step 2: the clusters touching one unit (after step 1).
    std::map<std::size_t, AbsorbedUnit> result; // by root index
    for (std::size_t unit = 0; unit < list.size(); ++unit)
    {
        if (system.isMoved(list[unit].id))
            continue;
        AbsorbedUnit& absorbed = result[target[unit]];
        absorbed.id = list[target[unit]].id;
        absorbed.memberUnits.push_back(list[unit].id);
    }
    for (std::size_t c = 0; c < clusters.size(); ++c)
    {
        const ConnectionCluster& cluster = clusters[c];
        std::vector<std::size_t> touched;
        for (const std::int64_t unit : cluster.touchedUnits)
            touched.push_back(target[indexOf(unit)]);
        sortAndMakeUnique(touched);
        if (!options.absorbsClusters || touched.size() != 1 || !isAbsorbable(system, cluster, options))
        {
            m_remainingClusters.push_back(c);
            m_connectionNodes.insert(m_connectionNodes.end(), cluster.connectionOnlyNodes.begin(),
                                     cluster.connectionOnlyNodes.end());
            continue;
        }
        AbsorbedUnit& absorbed = result[touched.front()];
        absorbed.absorbedClusters.push_back(c);
        absorbed.absorbedNodeIds.insert(absorbed.absorbedNodeIds.end(), cluster.connectionOnlyNodes.begin(),
                                        cluster.connectionOnlyNodes.end());
        absorbed.memberUnits.insert(absorbed.memberUnits.end(), cluster.movedUnits.begin(), cluster.movedUnits.end());
        for (const std::size_t connection : cluster.connections)
            if (system.connections()[connection].kind != ConnectionKind::Mpc)
                absorbed.elementIds.push_back(system.connections()[connection].id);
    }
    sortAndMakeUnique(m_connectionNodes);

    for (auto& [root, absorbed] : result)
    {
        // Its own ID first, then the absorbed units sorted.
        std::sort(absorbed.memberUnits.begin(), absorbed.memberUnits.end());
        std::iter_swap(absorbed.memberUnits.begin(),
                       std::find(absorbed.memberUnits.begin(), absorbed.memberUnits.end(), absorbed.id));
        std::sort(absorbed.memberUnits.begin() + 1, absorbed.memberUnits.end());
        StampHasher hasher;
        for (const std::int64_t member : absorbed.memberUnits)
        {
            const CondensationUnit& unit = *set.find(member);
            hasher.mix(unit.revision);
            const std::span<const base::EntityId> nodes = system.unitNodes(member);
            absorbed.nodeIds.insert(absorbed.nodeIds.end(), nodes.begin(), nodes.end());
            absorbed.elementIds.insert(absorbed.elementIds.end(), unit.elementIds.begin(), unit.elementIds.end());
            m_memberOf.emplace_back(member, absorbed.id);
        }
        for (const std::size_t c : absorbed.absorbedClusters)
            hasher.mix(clusters[c].stamp);
        absorbed.revision = absorbed.absorbedClusters.empty() && absorbed.memberUnits.size() == 1
                                ? set.find(absorbed.id)->revision
                                : hasher.stamp();
        absorbed.nodeIds.insert(absorbed.nodeIds.end(), absorbed.absorbedNodeIds.begin(),
                                absorbed.absorbedNodeIds.end());
        sortAndMakeUnique(absorbed.nodeIds);
        sortAndMakeUnique(absorbed.absorbedNodeIds);
        sortAndMakeUnique(absorbed.elementIds);
        m_units.push_back(std::move(absorbed));
    }
    std::sort(m_units.begin(), m_units.end(), [](const AbsorbedUnit& a, const AbsorbedUnit& b) { return a.id < b.id; });
    std::sort(m_memberOf.begin(), m_memberOf.end());
}

const AbsorbedUnit* ClusterAbsorption::find(std::int64_t unitId) const noexcept
{
    const auto found = std::lower_bound(m_units.begin(), m_units.end(), unitId,
                                        [](const AbsorbedUnit& unit, std::int64_t id) { return unit.id < id; });
    return found != m_units.end() && found->id == unitId ? &*found : nullptr;
}

std::optional<std::int64_t> ClusterAbsorption::unitOfMember(std::int64_t unitId) const noexcept
{
    const auto found = std::lower_bound(m_memberOf.begin(), m_memberOf.end(),
                                        std::make_pair(unitId, std::numeric_limits<std::int64_t>::min()));
    if (found == m_memberOf.end() || found->first != unitId)
        return std::nullopt;
    return found->second;
}

} // namespace oos::reduction
