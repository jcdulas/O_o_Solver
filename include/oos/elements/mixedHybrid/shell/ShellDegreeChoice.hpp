#pragma once

#include "oos/elements/mixedHybrid/shell/ShellElementData.hpp"

#include <cstddef>

namespace oos::elements::mixedHybrid
{

// The shape indicator can flag distortion but cannot estimate the error in the solution on an affine cell.
// Auto therefore starts with cubic displacements, including cubic traces across shared edges. This also gives
// the quadratic edge rotation required by a rigid rotation of a conic edge.
//
// The departure of an element measures that distance on its surface alone: with A the affine map of its corners (for
// a quadrilateral the map through the mean of its corners whose two directions are the means of its opposite sides;
// for a triangle the flat triangle of its corners), it is the largest distance |X − A| at the corners, at the middles
// of the edges and at the centre, over the square root of the area of A. It is zero for a flat parallelogram and for
// a flat triangle, |c| / √area for a quadrilateral with straight edges whose bilinear term is c (a quarter of
// X0 − X1 + X2 − X3: for a trapezoid of parallel sides a and b as high as it is wide, |a − b| / (2 (a + b))), and
// about the sagitta of the element over its size for a curved one (an eighth of the angle it subtends). It does not
// change with the size of the element nor with a rigid motion.
//
struct ShellDegreeChoice
{
    std::size_t baseDegree = 3;
    std::size_t enrichedDegree = 4;
    double enrichmentDeparture = 0.05;
};

// Distances from the affine map, sampled on the same surface. The normal distance is the sagitta in length units.
struct ShellShapeDeparture
{
    double relative;
    double normal;
};

[[nodiscard]] ShellShapeDeparture shellShapeDeparture(std::size_t cornerCount, const SurfaceAt& surface);

// Enrich for in-plane distortion or for curvature whose sagitta exceeds the local thickness.
[[nodiscard]] std::size_t chooseShellDegree(const ShellShapeDeparture& departure, double thickness,
                                            const ShellDegreeChoice& choice);

} // namespace oos::elements::mixedHybrid
