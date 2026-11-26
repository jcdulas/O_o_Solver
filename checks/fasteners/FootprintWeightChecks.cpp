// The checks of what the tie of a fastener side over its cross-section (elements::interpolateFootprint) refuses: a
// footprint whose weights, or the shape functions of one of its points, do not sum to 1 is not a mean of the motion
// of the shell, and a rigid translation of the grid points would not be that of the piercing point.

#include "CheckRunner.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ShapeFunctions.hpp"
#include "oos/elements/fasteners/PiercingInterpolation.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

using oos::base::Vec3;
using oos::elements::AuxiliaryPoint;
using oos::elements::PiercingMap;
using oos::elements::SideGrid;

namespace
{

constexpr double SIDE = 10.0;           // the side of the element
constexpr double PARENT_HALF_SIDE = 0.2; // the corners of the footprint at (±0.2, ±0.2) in the parent square

// One square element (grid points counter-clockwise from the origin) and the four corners of a footprint about its
// centre, each with the weight `weight` and its shape functions times `shapeScale`.
struct Patch
{
    std::array<SideGrid, 4> grids;
    std::vector<AuxiliaryPoint> corners;
    Vec3 point;
};

Patch footprintOf(double weight, double shapeScale)
{
    Patch patch;
    constexpr std::array<std::array<double, 2>, 4> CORNERS{{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
    for (std::size_t g = 0; g < 4; ++g)
    {
        patch.grids[g].row = static_cast<std::int32_t>(g);
        patch.grids[g].position = {SIDE * CORNERS[g][0], SIDE * CORNERS[g][1], 0.0};
        patch.grids[g].frame = oos::base::Mat3::identity();
    }
    patch.point = {0.5 * SIDE, 0.5 * SIDE, 0.0};
    for (const std::array<double, 2> side : {std::array<double, 2>{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}})
    {
        const oos::elements::BilinearShape shape =
            oos::elements::bilinearShapeAt(side[0] * PARENT_HALF_SIDE, side[1] * PARENT_HALF_SIDE);
        AuxiliaryPoint corner;
        corner.weight = weight;
        corner.gridIndices = {0, 1, 2, 3};
        for (const double value : shape.n)
            corner.shapeValues.push_back(shapeScale * value);
        patch.corners.push_back(corner);
    }
    return patch;
}

// Whether the footprint gives a map (true) or a problem (false).
bool givesMap(const Patch& patch)
{
    return std::holds_alternative<PiercingMap>(
        oos::elements::interpolateFootprint(patch.point, patch.grids, patch.corners));
}

} // namespace

CHECK_CASE("fasteners: a footprint whose weights do not sum to one is refused")
{
    constexpr double QUARTER = 0.25;
    const Patch whole = footprintOf(QUARTER, 1.0);
    const std::variant<PiercingMap, std::string> result =
        oos::elements::interpolateFootprint(whole.point, whole.grids, whole.corners);
    runner.expect(std::holds_alternative<PiercingMap>(result), "four corners of weight 1/4 give a map");
    if (std::holds_alternative<PiercingMap>(result))
    {
        // A rigid translation of the grid points is the translation of the piercing point: the columns of the three
        // translations of the four grid points sum to the identity.
        const PiercingMap& map = std::get<PiercingMap>(result);
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
            {
                double sum = 0.0;
                for (std::size_t g = 0; g < 4; ++g)
                    sum += map.at(r, 6 * g + c);
                runner.expectNear(sum, r == c ? 1.0 : 0.0, 1.0e-12, "a rigid translation is carried whole");
            }
    }

    // Weights that sum to 0.8: the mean translation would be 0.8 of a rigid translation.
    runner.expect(!givesMap(footprintOf(0.2, 1.0)), "four corners of weight 0.2 are refused");
    // Shape functions that sum to 0.9 at each corner: the same defect, in the read of each corner.
    runner.expect(!givesMap(footprintOf(QUARTER, 0.9)), "shape functions that sum to 0.9 are refused");
    // Within the tolerance of the table (PIERCING_WEIGHT_SUM_TOLERANCE): taken.
    runner.expect(givesMap(footprintOf(QUARTER * (1.0 + 1.0e-12), 1.0)), "a sum off 1 by round-off is taken");
}
