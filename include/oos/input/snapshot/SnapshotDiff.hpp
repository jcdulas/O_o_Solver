#pragma once

#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstdint>
#include <vector>

namespace oos::input
{

// What changed between two snapshots of a model (taken again after an edit): the structure and content
// versions, and the components and load case groups whose revision stamps changed, so that the solver redoes only what
// depends on them.
//
// A revision stamp is a hash of a component's content, not of its number: a component of `after` whose stamp is one of
// the stamps of `before` is unchanged, even if the components were renumbered by a change of structure; one whose
// stamp is not there is changed (edited, or new). The same for the load revision of each load case group.
struct SnapshotDiff
{
    bool isStructureChanged = false; // cards added or removed: the row-index columns of `before` are stale
    bool isContentChanged = false;   // any edit
    bool hasUnitStamps = false;      // both snapshots hold the components; otherwise the lists below are empty
    std::vector<std::int64_t> changedComponents; // component IDs of `after`, increasing
    std::vector<std::int64_t> removedRevisions;  // stamps of `before` that no component of `after` has, increasing
    struct LoadGroup
    {
        std::int64_t componentId = 0;
        std::int64_t loadGroup = 0;
        std::int32_t subcase = 0;
    };
    std::vector<LoadGroup> changedLoadGroups; // rows of loadCaseGroups of `after`, in table order

    [[nodiscard]] bool isUnchanged() const noexcept { return !isStructureChanged && !isContentChanged; }

    // Compares two snapshots of the same model, `before` taken first.
    [[nodiscard]] static SnapshotDiff between(const ModelSnapshot& before, const ModelSnapshot& after);
};

} // namespace oos::input
