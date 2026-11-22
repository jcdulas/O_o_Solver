#include "oos/elements/mixedHybrid/shell/ShellDegreeChoice.hpp"

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace oos::elements::mixedHybrid
{

namespace
{

// An affine map of the reference coordinates: X = origin + ξ along[0] + η along[1], and its area over the reference
// domain.
struct AffineMap
{
    SpaceVector origin{};
    std::array<SpaceVector, 2> along{};
    double area = 0.0;

    [[nodiscard]] SpaceVector at(const ReferencePoint& point) const
    {
        return add3(origin, add3(scl3(along[0], point[0]), scl3(along[1], point[1])));
    }
};

// The affine map of the corners of an element. A quadrilateral on [−1, 1]²: the bilinear map of its corners without
// its term in ξ η, X̄ + ξ a + η b with X̄ the mean of the corners, a = (−X0 + X1 + X2 − X3) / 4 and
// b = (−X0 − X1 + X2 + X3) / 4, of area 4 |a × b|. A triangle on (−1, −1), (1, −1), (−1, 1): the flat triangle of its
// corners, a = (X1 − X0) / 2, b = (X2 − X0) / 2, of area 2 |a × b|.
AffineMap affineMapOf(std::size_t cornerCount, const SurfaceAt& surface)
{
    std::array<SpaceVector, 4> corners{};
    for (std::size_t k = 0; k < cornerCount; ++k)
    {
        const ReferencePoint at = referenceCorner(cornerCount, k);
        corners[k] = surface(at[0], at[1]);
    }
    AffineMap map;
    if (cornerCount == 3)
    {
        map.along = {scl3(sub3(corners[1], corners[0]), 0.5), scl3(sub3(corners[2], corners[0]), 0.5)};
        // the corner 0 is at (−1, −1)
        map.origin = add3(corners[0], add3(map.along[0], map.along[1]));
        map.area = 2.0 * norm3(cross3(map.along[0], map.along[1]));
        return map;
    }
    map.origin = scl3(add3(add3(corners[0], corners[1]), add3(corners[2], corners[3])), 0.25);
    map.along = {scl3(sub3(add3(corners[1], corners[2]), add3(corners[0], corners[3])), 0.25),
                 scl3(sub3(add3(corners[2], corners[3]), add3(corners[0], corners[1])), 0.25)};
    map.area = 4.0 * norm3(cross3(map.along[0], map.along[1]));
    return map;
}

// The points where a surface is compared with the affine map of its corners: the corners, the middles of the edges and
// the centre.
std::vector<ReferencePoint> departurePoints(std::size_t cornerCount)
{
    std::vector<ReferencePoint> points;
    for (std::size_t k = 0; k < cornerCount; ++k)
    {
        points.push_back(referenceCorner(cornerCount, k));
        points.push_back(referenceEdgePointAndDirection(cornerCount, k, 0.5).first);
    }
    points.push_back(referenceCentre(cornerCount));
    return points;
}

} // namespace

ShellShapeDeparture shellShapeDeparture(std::size_t cornerCount, const SurfaceAt& surface)
{
    const AffineMap affine = affineMapOf(cornerCount, surface);
    if (!(affine.area > 0.0))
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    const SpaceVector normal = cross3(affine.along[0], affine.along[1]);
    const SpaceVector unitNormal = scl3(normal, 1.0 / norm3(normal));
    double largest = 0.0;
    double sagitta = 0.0;
    for (const ReferencePoint& point : departurePoints(cornerCount))
    {
        const SpaceVector offset = sub3(surface(point[0], point[1]), affine.at(point));
        largest = std::max(largest, norm3(offset));
        sagitta = std::max(sagitta, std::fabs(dot3(offset, unitNormal)));
    }
    return {largest / std::sqrt(affine.area), sagitta};
}

std::size_t chooseShellDegree(const ShellShapeDeparture& departure, double thickness, const ShellDegreeChoice& choice)
{
    return departure.relative > choice.enrichmentDeparture || departure.normal > thickness ? choice.enrichedDegree
                                                                                          : choice.baseDegree;
}

} // namespace oos::elements::mixedHybrid
