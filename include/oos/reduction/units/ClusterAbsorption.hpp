#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/units/ConnectionSystem.hpp"
#include "oos/reduction/units/UnitSet.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::reduction
{

// What the absorption may do.
struct AbsorptionOptions
{
    // False keeps every cluster and every 1D/0D-only unit where the connection system put it (the boundaries are then
    // O_o_Reader's interface points: a check, and the "before absorption" column of the plan report).
    bool absorbsClusters = true;
    // The SIDs of the MPC cards that some subcases select and others do not (an MultiPointConstrUnion resolved to its
    // cards: the connections name an MPC equation by its card's SID): a cluster holding an equation of such a card
    // stays in the connection system, so that one condensation serves every subcase. Sorted.
    std::vector<base::EntityId> varyingMpcSets;
};

// A unit after absorption: the unit itself, the 1D/0D-only units hanging from it, and the clusters touching it alone.
struct AbsorbedUnit
{
    std::int64_t id = 0;
    std::vector<std::int64_t> memberUnits;        // its own ID first, then the 1D/0D-only units absorbed, sorted
    std::vector<std::size_t> absorbedClusters;    // into ConnectionSystem::clusters(), increasing
    std::vector<base::EntityId> nodeIds;         // every point it holds, sorted
    std::vector<base::EntityId> absorbedNodeIds; // the connection-only points of its absorbed clusters, sorted
    std::vector<base::EntityId> elementIds;      // its elements and the connections it absorbed (not the MPCs)
    std::int64_t revision = 0;                    // the unit's stamp with what it absorbed (StampHasher)
};

// The absorption of the clusters that touch one unit (such clusters, and components made only of 1D and 0D elements,
// are absorbed into it):
// 1. a 1D/0D-only unit the connection system did not move and that is linked to exactly one other unit is absorbed
//    into it (a 1D chain hanging from a unit); two such units linked only to each other become one unit;
// 2. a cluster whose connections use points of exactly one unit (after step 1) is absorbed into it: a PointMassInertia on the
//    reference point of a spider of that unit, a spring or bush to the ground, an RigidBodyStar tying points of the
//    unit, a chain hanging from it. Its elements and relations become part of the unit, its connection-only points
//    interior points of the unit, its stamp part of the unit's stamp. This is exact: the relations are the same
//    equations, eliminated inside the unit instead of at the interface;
// 3. the other clusters (two units or more, none, or holding an MPC equation of a set that varies between subcases)
//    stay in the connection system: their connections are assembled in the interface problem and their
//    connection-only points are DOFs of it.
// The unit keeps the ID of the unit it grew from; an absorbed 1D/0D-only unit's ID names the unit it went into
// (unitOfMember).
//
// Thread-safety: immutable once built. Ownership: owns its lists; borrows the connection system (and through it the
// units), which must outlive it.
class ClusterAbsorption
{
public:
    ClusterAbsorption(const ConnectionSystem& system, const AbsorptionOptions& options = {});

    // The units after absorption, sorted by ID (the moved and the absorbed 1D/0D-only units are not among them).
    [[nodiscard]] std::span<const AbsorbedUnit> units() const noexcept { return m_units; }
    [[nodiscard]] const AbsorbedUnit* find(std::int64_t unitId) const noexcept;
    // The unit of the result that a unit of the set became part of; nothing for a unit moved to the connection system.
    [[nodiscard]] std::optional<std::int64_t> unitOfMember(std::int64_t unitId) const noexcept;
    // The clusters left in the connection system, increasing.
    [[nodiscard]] std::span<const std::size_t> remainingClusters() const noexcept { return m_remainingClusters; }
    // The connection-only points left in the connection system (DOFs of the interface problem), sorted.
    [[nodiscard]] std::span<const base::EntityId> connectionNodes() const noexcept { return m_connectionNodes; }
    [[nodiscard]] const ConnectionSystem& system() const noexcept { return *m_system; }

private:
    const ConnectionSystem* m_system;
    std::vector<AbsorbedUnit> m_units;
    std::vector<std::pair<std::int64_t, std::int64_t>> m_memberOf; // (unit of the set, unit of the result), sorted
    std::vector<std::size_t> m_remainingClusters;
    std::vector<base::EntityId> m_connectionNodes;
};

} // namespace oos::reduction
