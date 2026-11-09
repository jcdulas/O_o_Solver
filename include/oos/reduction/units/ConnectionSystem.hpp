#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/units/UnitSet.hpp"

#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::reduction
{

// What a connection is (the kind of O_o_Reader's connectionClusters(detail="connections")). The
// relations among them (Rigid, RigidInterpolation, Mpc, and the piercing constraints of a Fastener) are eliminated at
// the interface when they stay in the connection system.
enum class ConnectionKind : std::uint8_t
{
    Rigid,              // "rigid": RigidBodyMulti, RigidBodyStar, RigidBar, RigidRod and the other rigid elements
    RigidInterpolation, // "rigidInterpolation": a RigidInterpolation
    Mpc,                // "mpc": one MPC equation, by the set ID of its card
    Fastener,           // "fastener": a Fastener or Weld (the constraints of its piercing points are relations)
    Connector,          // "connector": springs, dampers, bushes, gaps, general elements, a 1D element or a mass between
                        // components
    Mass,               // "mass": a mass on a point that only connections use
    Contact // "contact": a pair of contact bodies that meet (contact="connection" only; the glue is GlueUnitCheck's)
};

// A connection: an element that belongs to no unit, or an MPC equation (whatever links components is a connection).
struct Connection
{
    ConnectionKind kind = ConnectionKind::Fastener;
    std::string card;
    base::EntityId id = 0;             // the element ID, the set ID of an MPC
    std::vector<base::EntityId> nodes; // the points it acts on (nodeIds), sorted and distinct
    std::int64_t revision = 0;         // O_o_Reader's stamp of the connection
};

// An MPC set a subcase selects that holds the equations of an MPC card set (an MultiPointConstrUnion of it), when
// multiPointConstrTable does not name the card of each row (its source_sid column): O_o_Reader's connection list names
// an MPC equation by the set ID of its card, the constraints of the analysis by the set the subcase selects, so the
// connections of the card stand for the selected set too (ConnectionSystem).
struct MpcSetAlias
{
    base::EntityId cardSetId = 0;
    base::EntityId setId = 0;

    friend auto operator<=>(const MpcSetAlias&, const MpcSetAlias&) = default;
};

// A cluster: connections joined through connection-only points, with the 1D/0D-only units moved into the connection
// system that they reach (clusters touching one unit are absorbed into it).
struct ConnectionCluster
{
    std::vector<std::size_t> connections;            // into ConnectionSystem::connections(), increasing
    std::vector<std::int64_t> movedUnits;            // 1D/0D-only units moved into the connection system, sorted
    std::vector<base::EntityId> connectionOnlyNodes; // its points that are in no unit, sorted
    std::vector<std::int64_t> touchedUnits;          // the units its connections use a point of, sorted
    std::int64_t stamp = 0;                          // StampHasher over its connections and moved units
};

// The connection system of a set of units (any mixture of connections, any chain of them, and components made only of
// 1D and 0D elements), from O_o_Reader's connection tables (connection_nodes, connectionClusters):
// - the points of each unit: a component's from connection_nodes, a group's from its own list;
// - the connections: every row of connectionClusters(detail="connections") (the direct links between
//   two components included), less those a group holds (a merged group or an element list that lists them: they are
//   elements of that unit); an MPC equation of a card set that a selected set holds (MpcSetAlias) is also a connection
//   of that set, as the constraints of the analysis name it;
// - a 1D/0D-only unit linked through clusters to two or more other units is moved to the connection system (never
//   condensed: a handful of bars between thousands of units); moving it joins the clusters on its two sides, so the
//   rule is applied until no unit moves (a chain of such units moves as a whole);
// - the clusters: connections joined through points that no unit holds (connection-only points and the points of
//   moved units), computed again by O_o_Solver because a regrouping or a move changes them; each has its stamp.
// What becomes of each cluster (absorbed into the one unit it touches, or part of the interface problem) is
// ClusterAbsorption's.
//
// Thread-safety: immutable once built. Ownership: owns its lists (their size is the number of connections and
// connection-only points: std::vector).
class ConnectionSystem
{
public:
    // Throws base::ContractError for a connection kind outside ConnectionKind. A point of connection_nodes whose
    // component is no unit's (a component an element list took elements from) is skipped: the groups list their own
    // points.
    // mpcAliases: sorted (CondensationPlanner gives them when multiPointConstrTable has no source_sid).
    ConnectionSystem(const UnitSet& units, const base::ContractReader& connectionNodes,
                     const base::ContractReader& connectionClusters, std::span<const MpcSetAlias> mpcAliases = {});

    [[nodiscard]] std::span<const Connection> connections() const noexcept { return m_connections; }
    [[nodiscard]] std::span<const ConnectionCluster> clusters() const noexcept { return m_clusters; }
    // The 1D/0D-only units moved to the connection system, sorted.
    [[nodiscard]] std::span<const std::int64_t> movedUnits() const noexcept { return m_movedUnits; }
    [[nodiscard]] bool isMoved(std::int64_t unitId) const noexcept;
    // The points of a unit of the set (moved ones included), sorted; empty for an unknown ID.
    [[nodiscard]] std::span<const base::EntityId> unitNodes(std::int64_t unitId) const noexcept;
    // The units (not moved) that hold a point, sorted: none for a connection-only point, two or more for a point an
    // element list shares with another unit.
    [[nodiscard]] std::vector<std::int64_t> unitsOfNode(base::EntityId nodeId) const;
    // The cluster of each connection (index into clusters()).
    [[nodiscard]] std::size_t clusterOf(std::size_t connection) const noexcept { return m_clusterOf[connection]; }
    // The units of the set, as given.
    [[nodiscard]] const UnitSet& units() const noexcept { return *m_units; }

private:
    const UnitSet* m_units;
    std::vector<Connection> m_connections;
    std::vector<ConnectionCluster> m_clusters;
    std::vector<std::size_t> m_clusterOf;
    std::vector<std::int64_t> m_movedUnits;
    std::vector<std::vector<base::EntityId>> m_unitNodes;             // parallel to units().units()
    std::vector<std::pair<base::EntityId, std::int64_t>> m_nodeUnits; // (point, unit), sorted, moved units left out
};

} // namespace oos::reduction
