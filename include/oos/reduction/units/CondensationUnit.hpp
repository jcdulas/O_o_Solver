#pragma once

#include "oos/base/ids/EntityId.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oos::reduction
{

// Where a unit comes from (by default the unit of condensation is the component).
enum class UnitOrigin : std::uint8_t
{
    Component,        // a component of O_o_Reader (connectedComponents), the default unit
    MergedComponents, // several components merged by the caller (solver.regroup(merge=...))
    ElementList       // an explicit list of elements given by the caller, connections included, or what remains of a
                      // unit those lists took elements from
};

// One unit of condensation as O_o_Reader describes it: a component, or a group the caller made of components or of
// elements. It is the unit before the analysis-dependent rules of the solver (the connection system,
// the absorption of clusters, the boundary DOFs of an analysis): those are applied by ConnectionSystem,
// ClusterAbsorption and UnitBoundaries on top of it.
//
// Invariants (UnitSet checks them): componentIds, elementIds, nodeIds and boundaryNodeIds sorted and distinct;
// boundaryNodeIds ⊂ nodeIds when nodeIds is known; a Component unit has exactly one component, whose ID is the unit's.
struct CondensationUnit
{
    std::int64_t id = 0;
    UnitOrigin origin = UnitOrigin::Component;
    // The components of O_o_Reader it is made of (empty for an element list).
    std::vector<std::int64_t> componentIds;
    // The revision stamp: O_o_Reader's component revision, or its group revision for a regrouped unit; equal
    // for equal content (the cache key starts from it).
    std::int64_t revision = 0;
    // A component made only of 1D and 0D elements (no shell, no solid): the connection system decides whether it is a
    // unit, part of the connection system, or absorbed into the one unit it hangs from. Always false
    // for an element list: the caller chose it as a unit.
    bool isLineOrPointOnly = false;
    // Its elements (and, for a group, the connections the group holds), by element ID.
    std::vector<base::EntityId> elementIds;
    // Its grid and scalar points: given by O_o_Reader for a group; empty for a component until the connection
    // system reads them (componentNodes).
    std::vector<base::EntityId> nodeIds;
    // The boundary O_o_Reader computes (componentBoundaries, or the group boundary): the points of the unit
    // that a connection or an element outside it uses, before any absorption.
    std::vector<base::EntityId> boundaryNodeIds;
};

// An element the caller declares a connection whatever its type (the plan's `connections` option: a Bar2, Beam2,
// CELAS or Bushing between two parts): O_o_Reader's component queries take the list (their `connections` option), so
// it belongs to no component and O_o_Reader's connection list holds it (a connector, its points on the boundaries of
// the components they are in). Its card, points and revision are those of that list.
struct DeclaredConnection
{
    base::EntityId elementId = 0;
    std::string card;
    std::vector<base::EntityId> nodes; // sorted and distinct
    std::int64_t revision = 0;
};

} // namespace oos::reduction
