#include "oos/reduction/units/Regrouping.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/GroupContracts.hpp"

#include <algorithm>
#include <charconv>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace oos::reduction
{

namespace
{

// A unit while the rules are applied.
struct WorkingUnit
{
    CondensationUnit unit;
    bool isTouched = false;
};

[[noreturn]] void refuse(std::string_view what)
{
    throw std::invalid_argument(base::concatenate({"regroup: ", what})); // remedy given by each caller, in `what`
}

std::string missingGroup(input::GroupTable table, std::int64_t groupId)
{
    return base::concatenate({"the table ", input::groupTableName(table), " (", input::GROUP_TABLES_PREREQUISITE,
                              ") has no row for group ", std::to_string(groupId),
                              "; ask O_o_Reader for every group of Regrouping::groups()"});
}

// The rows of a group table by group: group ID -> rows. O_o_Reader keys its group tables by the name the caller gave
// each group (`group` utf8); O_o_Solver names a group by the decimal text of its ID (snapshot.group_tables).
std::map<std::int64_t, std::vector<std::int64_t>> rowsByGroup(const base::ContractReader& table)
{
    std::map<std::int64_t, std::vector<std::int64_t>> rows;
    const base::Utf8View groups = table.texts("group");
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const std::string_view name = groups[row];
        std::int64_t groupId = 0;
        const auto [end, error] = std::from_chars(name.data(), name.data() + name.size(), groupId);
        if (error != std::errc() || end != name.data() + name.size())
            throw base::ContractError(
                base::concatenate({"table '", table.tableName(), "', row ", std::to_string(row),
                                   ": expected a group named by the decimal text of its ID, found '", name,
                                   "'; ask O_o_Reader for the groups of Regrouping::groups() under those names"}));
        rows[groupId].push_back(row);
    }
    return rows;
}

std::vector<base::EntityId> sortedValues(const base::ContractReader& table, const std::string& column,
                                         const std::vector<std::int64_t>& rows)
{
    const std::span<const std::int32_t> values = table.values<std::int32_t>(column);
    std::vector<base::EntityId> result;
    result.reserve(rows.size());
    for (const std::int64_t row : rows)
        result.push_back(values[static_cast<std::size_t>(row)]);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace

Regrouping::Regrouping(const UnitSet& units, std::span<const std::vector<std::int64_t>> merges,
                       std::span<const std::vector<base::EntityId>> elementLists)
    : m_declared(units.declaredConnections().begin(), units.declaredConnections().end())
{
    const auto isDeclared = [this](base::EntityId element)
    {
        return std::binary_search(m_declared.begin(), m_declared.end(), DeclaredConnection{element, {}, {}, 0},
                                  [](const DeclaredConnection& a, const DeclaredConnection& b)
                                  { return a.elementId < b.elementId; });
    };
    std::map<std::int64_t, WorkingUnit> working;
    for (const CondensationUnit& unit : units.units())
        working.emplace(unit.id, WorkingUnit{unit, false});
    std::int64_t largestId = units.units().empty() ? 0 : units.units().back().id;

    // Merges: each list becomes one unit of its smallest ID.
    std::vector<std::int64_t> named;
    for (const std::vector<std::int64_t>& merge : merges)
        named.insert(named.end(), merge.begin(), merge.end());
    std::sort(named.begin(), named.end());
    if (const auto twice = std::adjacent_find(named.begin(), named.end()); twice != named.end())
        refuse(base::concatenate({"unit ", std::to_string(*twice),
                                  " is named twice in the merges; a unit belongs to one group: put the lists "
                                  "together"}));
    for (const std::vector<std::int64_t>& merge : merges)
    {
        if (merge.size() < 2)
            refuse(base::concatenate({"a merge needs two units or more, found ", std::to_string(merge.size()),
                                      "; give each merge as a list of unit IDs, e.g. merge=[[12, 13, 14]]"}));
        for (const std::int64_t id : merge)
        {
            const auto found = working.find(id);
            if (found == working.end())
                refuse(base::concatenate(
                    {"unit ", std::to_string(id), " is not one of the units given; take the IDs from solver.units()"}));
            if (found->second.unit.origin == UnitOrigin::ElementList)
                refuse(base::concatenate({"unit ", std::to_string(id),
                                          " is an element list; merge units made of whole components, or give "
                                          "one element list with the elements of both"}));
        }
        const std::int64_t target = *std::min_element(merge.begin(), merge.end());
        CondensationUnit merged;
        merged.id = target;
        merged.origin = UnitOrigin::MergedComponents;
        merged.isLineOrPointOnly = true;
        for (const std::int64_t id : merge)
        {
            const CondensationUnit& part = working.at(id).unit;
            merged.componentIds.insert(merged.componentIds.end(), part.componentIds.begin(), part.componentIds.end());
            merged.elementIds.insert(merged.elementIds.end(), part.elementIds.begin(), part.elementIds.end());
            merged.isLineOrPointOnly = merged.isLineOrPointOnly && part.isLineOrPointOnly;
        }
        std::sort(merged.componentIds.begin(), merged.componentIds.end());
        std::sort(merged.elementIds.begin(), merged.elementIds.end());
        for (const std::int64_t id : merge)
            working.erase(id);
        working.emplace(target, WorkingUnit{std::move(merged), true});
    }

    // Element lists: new units after the largest ID; the units they take elements from keep what remains.
    std::map<base::EntityId, std::int64_t> ownerOfElement;
    for (const auto& [id, entry] : working)
        for (const base::EntityId element : entry.unit.elementIds)
            ownerOfElement.emplace(element, id);
    // An element named twice is refused before any unit gives elements away: the loop below erases each listed
    // element from its owner once, and a second erase would find it gone.
    std::vector<base::EntityId> listed;
    for (const std::vector<base::EntityId>& list : elementLists)
    {
        if (list.empty())
            refuse("an element list is empty; give each unit as a non-empty list of element IDs");
        listed.insert(listed.end(), list.begin(), list.end());
    }
    std::sort(listed.begin(), listed.end());
    if (const auto twice = std::adjacent_find(listed.begin(), listed.end()); twice != listed.end())
        refuse(base::concatenate({"element ", std::to_string(*twice),
                                  " is in two element lists (or twice in one); an element belongs to one unit"}));
    for (const base::EntityId element : listed)
        if (isDeclared(element))
            refuse(base::concatenate({"element ", std::to_string(element),
                                      " is in an element list and declared a connection; a connection belongs to no "
                                      "unit: leave it out of the lists, or do not declare it"}));
    std::vector<CondensationUnit> newUnits;
    for (const std::vector<base::EntityId>& list : elementLists)
    {
        CondensationUnit unit;
        unit.id = ++largestId;
        unit.origin = UnitOrigin::ElementList;
        unit.elementIds = list;
        std::sort(unit.elementIds.begin(), unit.elementIds.end());
        for (const base::EntityId element : unit.elementIds)
        {
            const auto owner = ownerOfElement.find(element);
            if (owner == ownerOfElement.end())
                continue; // a connection (or an element no unit holds): O_o_Reader checks that it exists
            WorkingUnit& from = working.at(owner->second);
            std::vector<base::EntityId>& remaining = from.unit.elementIds;
            remaining.erase(std::lower_bound(remaining.begin(), remaining.end(), element));
            from.unit.origin = UnitOrigin::ElementList;
            from.unit.componentIds.clear();
            from.unit.isLineOrPointOnly = false;
            from.isTouched = true;
        }
        newUnits.push_back(std::move(unit));
    }

    for (auto& [id, entry] : working)
    {
        if (!entry.isTouched)
        {
            m_untouched.push_back(std::move(entry.unit));
            continue;
        }
        if (entry.unit.elementIds.empty())
            continue; // every element went to an element list
        GroupDefinition group{entry.unit.id, entry.unit.origin, {}, {}};
        if (group.origin == UnitOrigin::MergedComponents)
            group.componentIds = entry.unit.componentIds;
        else
            group.elementIds = entry.unit.elementIds;
        m_groups.push_back(std::move(group));
        m_isLineOrPointOnly.push_back(entry.unit.isLineOrPointOnly);
    }
    for (CondensationUnit& unit : newUnits)
    {
        m_groups.push_back(GroupDefinition{unit.id, UnitOrigin::ElementList, {}, std::move(unit.elementIds)});
        m_isLineOrPointOnly.push_back(false);
    }
}

UnitSet Regrouping::apply(const GroupTables& tables) const
{
    const auto elementRows = rowsByGroup(tables.elements);
    const auto nodeRows = rowsByGroup(tables.nodes);
    const auto boundaryRows = rowsByGroup(tables.boundaries);
    const auto summaryRows = rowsByGroup(tables.summary);
    const std::span<const std::int64_t> revisions = tables.summary.values<std::int64_t>("revision");

    std::vector<CondensationUnit> units = m_untouched;
    // O_o_Reader puts in a group every element whose grid points are all in it (group_revisions): a frame
    // lying on the plane where two element lists meet is then in both, the list it was given to and the one beside
    // it. An element belongs to the unit it was given to (an untouched unit, or an element list); one that no list
    // names and several groups absorb goes to the first of them by ID.
    std::map<base::EntityId, std::int64_t> givenTo;
    for (const CondensationUnit& unit : m_untouched)
        for (const base::EntityId element : unit.elementIds)
            givenTo.emplace(element, unit.id);
    for (const GroupDefinition& group : m_groups)
        for (const base::EntityId element : group.elementIds)
            givenTo.emplace(element, group.groupId);
    std::map<base::EntityId, std::int64_t> absorbedBy;
    for (std::size_t g = 0; g < m_groups.size(); ++g)
    {
        const GroupDefinition& group = m_groups[g];
        const auto summary = summaryRows.find(group.groupId);
        if (summary == summaryRows.end() || summary->second.size() != 1)
            throw base::ContractError(missingGroup(input::GroupTable::Summary, group.groupId));
        const auto elements = elementRows.find(group.groupId);
        if (elements == elementRows.end())
            throw base::ContractError(missingGroup(input::GroupTable::Elements, group.groupId));
        const auto nodes = nodeRows.find(group.groupId);
        if (nodes == nodeRows.end())
            throw base::ContractError(missingGroup(input::GroupTable::Nodes, group.groupId));

        CondensationUnit unit;
        unit.id = group.groupId;
        unit.origin = group.origin;
        unit.componentIds = group.componentIds;
        unit.revision = revisions[static_cast<std::size_t>(summary->second.front())];
        unit.isLineOrPointOnly = m_isLineOrPointOnly[g];
        unit.elementIds = sortedValues(tables.elements, "elementId", elements->second);
        std::erase_if(
            unit.elementIds,
            [&](base::EntityId element)
            {
                // A declared connection is in no unit (O_o_Reader's group takes it when its points are
                // all in the group).
                if (std::binary_search(m_declared.begin(), m_declared.end(), DeclaredConnection{element, {}, {}, 0},
                                       [](const DeclaredConnection& a, const DeclaredConnection& b)
                                       { return a.elementId < b.elementId; }))
                    return true;
                if (const auto given = givenTo.find(element); given != givenTo.end())
                    return given->second != unit.id;
                return !absorbedBy.emplace(element, unit.id).second;
            });
        unit.nodeIds = sortedValues(tables.nodes, "nodeId", nodes->second);
        if (const auto boundary = boundaryRows.find(group.groupId); boundary != boundaryRows.end())
            unit.boundaryNodeIds = sortedValues(tables.boundaries, "nodeId", boundary->second);
        if (group.origin == UnitOrigin::ElementList && !std::includes(unit.elementIds.begin(), unit.elementIds.end(),
                                                                      group.elementIds.begin(), group.elementIds.end()))
            throw base::ContractError(base::concatenate(
                {"group_elements (", input::GROUP_TABLES_PREREQUISITE, ") lacks elements of group ",
                 std::to_string(group.groupId),
                 " as it was defined (an element ID that is not in the model?); check the element lists given"}));
        units.push_back(std::move(unit));
    }
    return UnitSet(std::move(units), m_declared);
}

} // namespace oos::reduction
