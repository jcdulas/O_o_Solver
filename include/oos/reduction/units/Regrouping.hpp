#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/reduction/units/CondensationUnit.hpp"
#include "oos/reduction/units/UnitSet.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// A group whose boundary and stamp O_o_Reader is asked for: merged components, or a list of
// elements. groupId is the ID the unit takes.
struct GroupDefinition
{
    std::int64_t groupId = 0;
    UnitOrigin origin = UnitOrigin::MergedComponents; // MergedComponents or ElementList
    std::vector<std::int64_t> componentIds;           // MergedComponents: sorted
    std::vector<base::EntityId> elementIds;          // ElementList: sorted
};

// O_o_Reader's tables of the groups (input/contract/GroupContracts.hpp), read through their contracts.
struct GroupTables
{
    const base::ContractReader& elements;
    const base::ContractReader& nodes;
    const base::ContractReader& boundaries;
    const base::ContractReader& summary;
};

// The caller's regrouping of units: merge several components into one condensation unit (for example components 12,
// 13 and 14), or give an explicit list of elements (connections included) as a unit.
//
// Rules:
// - merges first: each list of unit IDs becomes one unit, of the smallest ID of the list; only units made of whole
//   components merge (a component or merged components);
// - then element lists: each becomes a new unit, numbered after the largest unit ID in the order given; a unit an
//   element list takes elements from keeps its ID and becomes the element list of what remains (and disappears when
//   nothing remains): its boundary changes, so O_o_Reader is asked for it too;
// - a unit no rule touches keeps O_o_Reader's component data: a group never changes the boundary of a unit it takes
//   nothing from (the point a group shares with it is in both boundaries);
// - an element belongs to one unit: O_o_Reader adds to a group every element whose grid points are all in it (a frame
//   on the plane where two lists meet is in both), so an element stays with the unit it was given to, and one no list
//   names that several groups take goes to the first of them by ID.
// - elements declared connections (the plan's `connections` option) are in no unit given: O_o_Reader makes the
//   components without them. They stay the units' declared connections
//   (UnitSet::declaredConnections), an element list may not name one, and no group of O_o_Reader keeps one (a group
//   takes every element whose grid points are all in it).
// groups() lists what O_o_Reader is asked for; apply() builds the new units from its answer (boundaries and stamps
// computed by O_o_Reader, never by O_o_Solver, so that both packages agree).
//
// Thread-safety: immutable once built. Ownership: owns its definitions; borrows nothing.
class Regrouping
{
public:
    // Throws std::invalid_argument for an unknown unit ID, a unit named twice, a merge of fewer than two units or of an
    // element-list unit, an empty element list, an element named twice, or an element both listed and declared a
    // connection.
    Regrouping(const UnitSet& units, std::span<const std::vector<std::int64_t>> merges,
               std::span<const std::vector<base::EntityId>> elementLists);

    [[nodiscard]] const std::vector<GroupDefinition>& groups() const noexcept { return m_groups; }

    // The units after regrouping: those no rule touches, as they were, and one per group with O_o_Reader's elements,
    // points, boundary and revision. Throws base::ContractError when a group is missing from a table or the tables
    // disagree with the definitions (an element of a definition missing from group_elements).
    [[nodiscard]] UnitSet apply(const GroupTables& tables) const;

private:
    std::vector<CondensationUnit> m_untouched;
    std::vector<GroupDefinition> m_groups;
    std::vector<bool> m_isLineOrPointOnly;      // per group: every merged component is 1D/0D only
    std::vector<DeclaredConnection> m_declared; // the units', by increasing ID
};

} // namespace oos::reduction
