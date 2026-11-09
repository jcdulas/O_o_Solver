#include "oos/input/snapshot/SnapshotDiff.hpp"

#include <algorithm>
#include <span>

namespace oos::input
{

namespace
{

std::vector<std::int64_t> sortedCopy(std::span<const std::int64_t> values)
{
    std::vector<std::int64_t> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

bool contains(const std::vector<std::int64_t>& sorted, std::int64_t value)
{
    return std::binary_search(sorted.begin(), sorted.end(), value);
}

} // namespace

SnapshotDiff SnapshotDiff::between(const ModelSnapshot& before, const ModelSnapshot& after)
{
    SnapshotDiff diff;
    diff.isStructureChanged = before.versions().structureVersion != after.versions().structureVersion;
    diff.isContentChanged =
        diff.isStructureChanged || before.versions().contentVersion != after.versions().contentVersion;
    diff.hasUnitStamps = before.hasTable(TableRole::Components) && after.hasTable(TableRole::Components);
    if (!diff.hasUnitStamps)
        return diff;

    const UnitTable& unitsBefore = before.units();
    const UnitTable& unitsAfter = after.units();
    const std::vector<std::int64_t> stampsBefore = sortedCopy(unitsBefore.revisions());
    const std::vector<std::int64_t> stampsAfter = sortedCopy(unitsAfter.revisions());
    const std::span<const std::int64_t> idsAfter = unitsAfter.componentIds();
    const std::span<const std::int64_t> revisionsAfter = unitsAfter.revisions();
    for (std::size_t row = 0; row < idsAfter.size(); ++row)
    {
        if (!contains(stampsBefore, revisionsAfter[row]))
            diff.changedComponents.push_back(idsAfter[row]);
    }
    std::sort(diff.changedComponents.begin(), diff.changedComponents.end());
    for (const std::int64_t stamp : stampsBefore)
    {
        if (!contains(stampsAfter, stamp) && (diff.removedRevisions.empty() || diff.removedRevisions.back() != stamp))
            diff.removedRevisions.push_back(stamp);
    }

    if (unitsBefore.hasLoadGroups() && unitsAfter.hasLoadGroups())
    {
        const base::ContractReader& groupsAfter = unitsAfter.loadGroups();
        const std::vector<std::int64_t> loadStampsBefore =
            sortedCopy(unitsBefore.loadGroups().values<std::int64_t>("loadRevision"));
        const std::span<const std::int64_t> loadStamps = groupsAfter.values<std::int64_t>("loadRevision");
        const std::span<const std::int64_t> components = groupsAfter.values<std::int64_t>("componentId");
        const std::span<const std::int64_t> groups = groupsAfter.values<std::int64_t>("loadGroup");
        const std::span<const std::int32_t> subcases = groupsAfter.values<std::int32_t>("subcase");
        for (std::size_t row = 0; row < loadStamps.size(); ++row)
        {
            if (!contains(loadStampsBefore, loadStamps[row]))
                diff.changedLoadGroups.push_back({components[row], groups[row], subcases[row]});
        }
    }
    return diff;
}

} // namespace oos::input
