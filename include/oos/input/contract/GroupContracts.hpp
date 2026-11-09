#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <array>
#include <string_view>

namespace oos::input
{

// The tables of user-defined groups of elements (the boundaries and stamps of a group given as a list of components to
// merge or as a list of element IDs; O_o_Reader data contract 34). They are not tables of the model snapshot: they
// answer a question about groups the caller defines (a regrouping), so the caller asks O_o_Reader for them with the
// group definitions, each group named by the decimal text of its ID, and hands them to reduction::Regrouping. The
// summary is O_o_Reader's groupRevisions(groups), the boundaries its componentBoundaries(groups=); the elements and
// the points of each group, which O_o_Reader does not list, are gathered by the caller from its
// connectedComponents(detail="elements"), enclosed_elements(groups) and node_elements queries.
//
// The rules these tables follow:
// - a merged group holds the elements of its components and every connection whose grid points are all in the group;
//   an element-list group holds the elements listed (connections included);
// - the boundary of a group is the grid points of the group that an element or connection outside the group uses;
//   never a point that is only constrained or only requested for output;
// - the revision stamp of a group is a 64-bit hash of the cards of its elements and of what they depend on, equal for
//   equal content and stable across sessions.
enum class GroupTable
{
    Elements,   // groupElements: group utf8, elementId int32 (one row per element of each group)
    Nodes,      // groupNodes: group utf8, nodeId int32 (the grid and scalar points of each group, sorted)
    Boundaries, // componentBoundaries(groups=): group utf8, nodeId int32, isInterface bool
    Summary     // groupRevisions(groups): group utf8, revision int64
};

inline constexpr std::array<GroupTable, 4> ALL_GROUP_TABLES{GroupTable::Elements, GroupTable::Nodes,
                                                            GroupTable::Boundaries, GroupTable::Summary};

// "groupElements", "groupNodes", "componentBoundaries", "groupRevisions".
[[nodiscard]] std::string_view groupTableName(GroupTable table) noexcept;

// The text that describes these tables in ContractError messages.
inline constexpr std::string_view GROUP_TABLES_PREREQUISITE = "the boundaries and stamps of user-defined groups";

[[nodiscard]] base::TableContract groupTableContract(GroupTable table);

// A reader of `table` checked against the contract of `kind` (ContractError when it breaks it).
[[nodiscard]] base::ContractReader readGroupTable(GroupTable kind, base::ImportedTable table);

} // namespace oos::input
