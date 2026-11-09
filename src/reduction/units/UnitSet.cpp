#include "oos/reduction/units/UnitSet.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

template <typename T>
bool isSortedAndDistinct(const std::vector<T>& values)
{
    return std::adjacent_find(values.begin(), values.end(), [](T a, T b) { return a >= b; }) == values.end();
}

template <typename T>
void sortAndMakeUnique(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

[[noreturn]] void refuseUnit(std::int64_t unitId, std::string_view what)
{
    throw std::invalid_argument(
        base::concatenate({"UnitSet: unit ", std::to_string(unitId), " ", what,
                           "; build the units with UnitSet::fromComponents or reduction::Regrouping"}));
}

template <typename Key>
void checkOwnedOnce(std::vector<std::pair<Key, std::int64_t>>& owners, std::string_view what)
{
    std::sort(owners.begin(), owners.end());
    for (std::size_t k = 1; k < owners.size(); ++k)
    {
        if (owners[k].first == owners[k - 1].first)
            throw std::invalid_argument(base::concatenate(
                {"UnitSet: ", what, " ", std::to_string(owners[k].first), " is in units ",
                 std::to_string(owners[k - 1].second), " and ", std::to_string(owners[k].second), "; a ", what,
                 " belongs to one unit at most: regroup so that the groups do not overlap"}));
    }
}

} // namespace

UnitSet::UnitSet(std::vector<CondensationUnit> units, std::vector<DeclaredConnection> declaredConnections)
    : m_units(std::move(units)), m_declaredConnections(std::move(declaredConnections))
{
    std::sort(m_units.begin(), m_units.end(),
              [](const CondensationUnit& a, const CondensationUnit& b) { return a.id < b.id; });
    for (std::size_t k = 0; k < m_units.size(); ++k)
    {
        const CondensationUnit& unit = m_units[k];
        if (k > 0 && m_units[k - 1].id == unit.id)
            refuseUnit(unit.id, "is given twice");
        if (!isSortedAndDistinct(unit.componentIds) || !isSortedAndDistinct(unit.elementIds) ||
            !isSortedAndDistinct(unit.nodeIds) || !isSortedAndDistinct(unit.boundaryNodeIds))
            refuseUnit(unit.id, "has a list of components, elements or points that is not sorted and distinct");
        if (unit.origin == UnitOrigin::Component &&
            (unit.componentIds.size() != 1 || unit.componentIds.front() != unit.id))
            refuseUnit(unit.id, "is a component unit whose component list is not its own ID");
        if (!unit.nodeIds.empty() && !std::includes(unit.nodeIds.begin(), unit.nodeIds.end(),
                                                    unit.boundaryNodeIds.begin(), unit.boundaryNodeIds.end()))
            refuseUnit(unit.id, "has a boundary point that is not one of its points");
        for (const std::int64_t component : unit.componentIds)
            m_componentUnits.emplace_back(component, unit.id);
        for (const base::EntityId element : unit.elementIds)
            m_elementUnits.emplace_back(element, unit.id);
    }
    checkOwnedOnce(m_componentUnits, "component");
    checkOwnedOnce(m_elementUnits, "element");
    std::sort(m_declaredConnections.begin(), m_declaredConnections.end(),
              [](const DeclaredConnection& a, const DeclaredConnection& b) { return a.elementId < b.elementId; });
    for (std::size_t k = 0; k < m_declaredConnections.size(); ++k)
    {
        const DeclaredConnection& connection = m_declaredConnections[k];
        if (k > 0 && m_declaredConnections[k - 1].elementId == connection.elementId)
            throw std::invalid_argument(base::concatenate({"UnitSet: element ", std::to_string(connection.elementId),
                                                           " is declared a connection twice; declare it once"}));
        if (const std::optional<std::int64_t> owner = unitOfElement(connection.elementId))
            throw std::invalid_argument(base::concatenate(
                {"UnitSet: element ", std::to_string(connection.elementId), " is declared a connection but is in unit ",
                 std::to_string(*owner),
                 "; take the components with the connection list (O_o_Reader's connections= option)"}));
        if (!isSortedAndDistinct(connection.nodes))
            throw std::invalid_argument(
                base::concatenate({"UnitSet: the points of declared connection ", std::to_string(connection.elementId),
                                   " are not sorted and distinct; give them so"}));
    }
}

UnitSet UnitSet::fromComponents(const input::UnitTable& table, std::vector<DeclaredConnection> declaredConnections)
{
    const base::ContractReader& components = table.components();
    const std::span<const std::int64_t> ids = table.componentIds();
    const std::span<const std::int64_t> revisions = table.revisions();
    // O_o_Reader (contract 34) flags the components made only of 1D and 0D elements (is_line_only); a table without
    // the flag tells them by their shell and solid counts.
    const bool hasLineFlag = components.hasColumn("isLineOnly");
    const bool hasCounts = components.hasColumn("shellCount") && components.hasColumn("solidCount");
    std::map<std::int64_t, CondensationUnit> units;
    for (std::size_t row = 0; row < ids.size(); ++row)
    {
        CondensationUnit unit;
        unit.id = ids[row];
        unit.origin = UnitOrigin::Component;
        unit.componentIds = {ids[row]};
        unit.revision = revisions[row];
        if (hasLineFlag)
        {
            unit.isLineOrPointOnly = components.booleans("isLineOnly")[static_cast<std::int64_t>(row)];
        }
        else if (hasCounts)
        {
            const std::int64_t shells = components.values<std::int64_t>("shellCount")[row];
            const std::int64_t solids = components.values<std::int64_t>("solidCount")[row];
            unit.isLineOrPointOnly = shells == 0 && solids == 0;
        }
        units.emplace(unit.id, std::move(unit));
    }

    // connectedComponents(detail="elements"): the component of every element, null for a connection.
    const base::ContractReader& elements = table.componentElements();
    const std::span<const std::int32_t> elementIds = elements.values<std::int32_t>("elementId");
    const std::span<const std::int64_t> elementComponents =
        elements.values<std::int64_t>("componentId", base::NullPolicy::Allow);
    const base::ValidityView isInComponent = elements.validity("componentId");
    for (std::size_t row = 0; row < elementIds.size(); ++row)
    {
        if (!isInComponent.isValid(static_cast<std::int64_t>(row)))
            continue;
        const auto found = units.find(elementComponents[row]);
        if (found == units.end())
            throw base::ContractError(base::concatenate(
                {"componentElements: element ", std::to_string(elementIds[row]), " names component ",
                 std::to_string(elementComponents[row]),
                 ", which connectedComponents does not list; take the snapshot again from one model state"}));
        found->second.elementIds.push_back(elementIds[row]);
    }

    if (table.hasBoundaries())
    {
        // componentBoundaries: one row per boundary point and load group (a single null load group with
        // retain=None); the unit's boundary is the union over its load groups.
        const base::ContractReader& boundaries = table.boundaries();
        const std::span<const std::int64_t> boundaryComponents = boundaries.values<std::int64_t>("componentId");
        const std::span<const std::int32_t> boundaryNodes = boundaries.values<std::int32_t>("nodeId");
        for (std::size_t row = 0; row < boundaryNodes.size(); ++row)
        {
            const auto found = units.find(boundaryComponents[row]);
            if (found == units.end())
                throw base::ContractError(base::concatenate(
                    {"componentBoundaries: node ", std::to_string(boundaryNodes[row]), " is on component ",
                     std::to_string(boundaryComponents[row]),
                     ", which connectedComponents does not list; take the snapshot again from one model state"}));
            found->second.boundaryNodeIds.push_back(boundaryNodes[row]);
        }
    }

    std::vector<CondensationUnit> list;
    list.reserve(units.size());
    for (auto& [id, unit] : units)
    {
        sortAndMakeUnique(unit.elementIds);
        sortAndMakeUnique(unit.boundaryNodeIds);
        list.push_back(std::move(unit));
    }
    return UnitSet(std::move(list), std::move(declaredConnections));
}

const CondensationUnit* UnitSet::find(std::int64_t unitId) const noexcept
{
    const std::optional<std::size_t> index = indexOf(unitId);
    return index ? &m_units[*index] : nullptr;
}

std::optional<std::size_t> UnitSet::indexOf(std::int64_t unitId) const noexcept
{
    const auto found = std::lower_bound(m_units.begin(), m_units.end(), unitId,
                                        [](const CondensationUnit& unit, std::int64_t id) { return unit.id < id; });
    if (found == m_units.end() || found->id != unitId)
        return std::nullopt;
    return static_cast<std::size_t>(found - m_units.begin());
}

std::optional<std::int64_t> UnitSet::unitOfComponent(std::int64_t componentId) const noexcept
{
    const auto found = std::lower_bound(m_componentUnits.begin(), m_componentUnits.end(),
                                        std::make_pair(componentId, std::numeric_limits<std::int64_t>::min()));
    if (found == m_componentUnits.end() || found->first != componentId)
        return std::nullopt;
    return found->second;
}

std::optional<std::int64_t> UnitSet::unitOfElement(base::EntityId elementId) const noexcept
{
    const auto found = std::lower_bound(m_elementUnits.begin(), m_elementUnits.end(),
                                        std::make_pair(elementId, std::numeric_limits<std::int64_t>::min()));
    if (found == m_elementUnits.end() || found->first != elementId)
        return std::nullopt;
    return found->second;
}

} // namespace oos::reduction
