#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oos::reduction
{

// The zone of an external unit as an analysis resolved it (the contour of output/sections/ZoneContour on the model;
// solutions/sol101/ExternalUnitPath), in the terms of the reduction layer:
// - the elements the unit replaces (the members of the zone: its elements with a stiffness or a mass, sorted);
// - its contour DOFs: per contour grid point the components through which the rest of the model acts on it, in the
//   displacement system of the grid point: the rows K* must have;
// - its interior DOFs: the six components of each interior grid point; their loads go to the plugin and the DOFs leave
//   the analysis (nothing acts on them once the zone's elements are replaced);
// each list in increasing grid ID then component, with the g-set DOF of each label (assembly::DofMap).
//
// The lists are bounded by the zone (std::vector). Thread-safety: a value.
struct ExternalZone
{
    std::int32_t id = 0;
    std::string name;
    std::vector<base::EntityId> replacedElements;
    std::vector<UnitDofLabel> contourLabels;
    std::vector<base::DofIndex> contourDofs;
    std::vector<UnitDofLabel> interiorLabels;
    std::vector<base::DofIndex> interiorDofs;
    // An interior grid point is held in every subcase (the zone holds the constraint and its local model is held
    // there too): K* then need not leave the rigid-body motions of the contour free.
    bool holdsConstraints = false;

    // The name as messages print it: "zone <name>" (or its ID).
    [[nodiscard]] std::string displayName() const
    {
        return std::string("zone ").append(name.empty() ? std::to_string(id) : name);
    }
};

} // namespace oos::reduction
