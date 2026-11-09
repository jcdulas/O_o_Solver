#include "oos/solutions/common/GlueUnitCheck.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace oos::solutions
{

namespace
{

// What the crossing points of one contact table touch, for its report line.
struct CrossingTally
{
    std::int64_t pointCount = 0;
    std::set<std::int64_t> units; // bounded by the units the table joins
};

} // namespace

bool isGlueSource(assembly::ConstraintSource source) noexcept
{
    return source == assembly::ConstraintSource::ContactTable || source == assembly::ConstraintSource::ContactPair;
}

base::BudgetedBuffer<std::int64_t> unitOfGridRows(const reduction::UnitSet& units, const input::ModelSnapshot& snapshot,
                                                  const assembly::DofMap& dofMap, base::MemoryBudget& budget)
{
    const auto gridCount = static_cast<std::size_t>(dofMap.gridCount());
    base::BudgetedBuffer<std::int64_t> unitOf(budget, gridCount, "glue: unit of each grid point");
    for (std::size_t row = 0; row < gridCount; ++row)
        unitOf[row] = NO_UNIT;
    if (snapshot.hasTable(input::TableRole::ConnectionNodes))
    {
        const base::ContractReader& nodes = snapshot.connectionNodes();
        const std::span<const std::int32_t> nodeIds = nodes.values<std::int32_t>("nodeId");
        const std::span<const std::int64_t> components =
            nodes.values<std::int64_t>("componentId", base::NullPolicy::Allow);
        const base::ValidityView hasComponent = nodes.validity("componentId");
        for (std::size_t row = 0; row < nodeIds.size(); ++row)
        {
            if (!hasComponent.isValid(static_cast<std::int64_t>(row)))
                continue;
            const std::optional<std::int32_t> gridRow = dofMap.points().findGrid(nodeIds[row]);
            const std::optional<std::int64_t> unit = units.unitOfComponent(components[row]);
            if (gridRow && unit)
                unitOf[static_cast<std::size_t>(*gridRow)] = *unit;
        }
    }
    // A unit that lists its points (a group) says which points it holds.
    for (const reduction::CondensationUnit& unit : units.units())
        for (const base::EntityId nodeId : unit.nodeIds)
            if (const std::optional<std::int32_t> gridRow = dofMap.points().findGrid(nodeId))
                unitOf[static_cast<std::size_t>(*gridRow)] = unit.id;
    return unitOf;
}

GlueUnitCheck checkGlueAgainstUnits(const assembly::GlueConstraints& glue, const assembly::DofMap& dofMap,
                                    std::span<const std::int64_t> unitOfGridRow)
{
    GlueUnitCheck check;
    const auto unitOfNode = [&](std::int32_t node) -> std::int64_t
    {
        // Grid points first in the node numbering of the DOF map; a scalar point is in no unit here.
        return node < dofMap.gridCount() ? unitOfGridRow[static_cast<std::size_t>(node)] : NO_UNIT;
    };
    // The rows of a glued point are consecutive (GlueConstraints writes them point by point): one decision per point.
    std::map<std::pair<std::uint8_t, base::EntityId>, CrossingTally> crossing; // bounded by the contact tables
    std::size_t first = 0;
    while (first < glue.rowCount())
    {
        const std::int32_t point = dofMap.nodeDofOf(glue.dependentDof(first)).node;
        std::size_t last = first + 1;
        while (last < glue.rowCount() && dofMap.nodeDofOf(glue.dependentDof(last)).node == point)
            ++last;
        const std::int64_t unit = unitOfNode(point);
        bool isInside = unit != NO_UNIT;
        CrossingTally* tally = nullptr;
        for (std::size_t i = first; i < last; ++i)
            for (const assembly::ConstraintTerm& term : glue.row(i))
                isInside = isInside && unitOfNode(dofMap.nodeDofOf(term.dof).node) == unit;
        if (isInside)
        {
            ++check.insidePointCount;
            first = last;
            continue;
        }
        ++check.crossingPointCount;
        check.interfaceRelations.push_back(reduction::glueRelationKey(dofMap.gridId(point)));
        tally = &crossing[{static_cast<std::uint8_t>(glue.sourceOf(first)), glue.sourceIdOf(first)}];
        ++tally->pointCount;
        if (unit != NO_UNIT)
            tally->units.insert(unit);
        for (std::size_t i = first; i < last; ++i)
            for (const assembly::ConstraintTerm& term : glue.row(i))
                if (const std::int64_t faceUnit = unitOfNode(dofMap.nodeDofOf(term.dof).node); faceUnit != NO_UNIT)
                    tally->units.insert(faceUnit);
        first = last;
    }
    std::sort(check.interfaceRelations.begin(), check.interfaceRelations.end());
    check.interfaceRelations.erase(std::unique(check.interfaceRelations.begin(), check.interfaceRelations.end()),
                                   check.interfaceRelations.end());
    for (const auto& [key, tally] : crossing)
    {
        std::string units;
        for (const std::int64_t unit : tally.units)
            units.append(units.empty() ? "" : ", ").append(std::to_string(unit));
        check.reportLines.push_back(base::concatenate(
            {assembly::cardNameOf(static_cast<assembly::ConstraintSource>(key.first)), " ", std::to_string(key.second),
             ": ", std::to_string(tally.pointCount), " glued grid points join different units (",
             units.empty() ? std::string("none") : units,
             "): their glue equations are eliminated at the interface, the points and the corners of their faces on "
             "the boundaries of their units; none is dropped"}));
    }
    return check;
}

} // namespace oos::solutions
