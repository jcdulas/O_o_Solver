#pragma once

#include "oos/base/math/Vec3.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace oos::output
{

// The components of a cut load: the force along the axes of CD and the moment about them through the point.
inline constexpr std::array<std::string_view, 6> CUT_COMPONENTS{"fx", "fy", "fz", "mx", "my", "mz"};

// One cut: a set of grid points, the set of the elements on one side, a reference point and a coordinate system (the
// fields GRIDSET, ELEMSET, CID and X, Y, Z, CD of a MONPNT3). The loads of the cut are the forces that the elements of
// the set exert on the grid points of the set (the element rows of their grid point forces), summed with their moments
// about the point, along the axes of CD.
struct CutDefinition
{
    std::int32_t id = 0;
    std::string name;
    std::int32_t group = 0; // a tag of the caller carried to the labels (a DesignProblem: the response that asks)
    std::vector<std::int32_t> grids;      // the grid points of the cut
    std::vector<std::int32_t> elements;   // the elements on one side
    base::Vec3 point;                     // in the system positionFrameId
    std::int32_t positionFrameId = 0;     // the rectangular system of the point (MONPNT3 CID)
    std::int32_t displacementFrameId = 0; // the rectangular system of the axes of the loads (MONPNT3 CD)
};

} // namespace oos::output
