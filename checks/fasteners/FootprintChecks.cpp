// The checks of the tie of a fastener side over its cross-section (elements::interpolateFootprint): the four corners
// of a square about a grid point, each on one of the four elements around it. The piercing point takes the mean
// translation of the corners, the rotation of the shell about the axes of its plane (the rotations of the grid
// points) and the rotation of the sheet in its plane (from the translations).

#include "CheckRunner.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ShapeFunctions.hpp"
#include "oos/elements/fasteners/PiercingInterpolation.hpp"

#include <array>
#include <cmath>
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

constexpr double CELL = 30.0;      // the side of an element
constexpr double HALF_SIDE = 1.77; // the half side of the square of a fastener of diameter 4
constexpr std::size_t GRIDS = 9;   // 3 × 3 grid points, four elements about the middle one

// The grid points (row by row, x then y) and the four elements about the middle one, with the square about it.
struct Patch
{
    std::array<SideGrid, GRIDS> grids;
    std::vector<AuxiliaryPoint> corners;
    Vec3 point;
};

Patch squareAboutGridPoint()
{
    Patch patch;
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 3; ++i)
        {
            SideGrid& grid = patch.grids[3 * j + i];
            grid.row = static_cast<std::int32_t>(3 * j + i);
            grid.position = {CELL * static_cast<double>(i), CELL * static_cast<double>(j), 0.0};
            grid.frame = oos::base::Mat3::identity();
        }
    patch.point = {CELL, CELL, 0.0};
    // The corner (sx, sy)·h of the square is in the element on that side of the middle grid point, at the parent
    // point that is h from the corner of the element the middle grid point is.
    for (const std::array<int, 2> side : {std::array<int, 2>{-1, -1}, {1, -1}, {1, 1}, {-1, 1}})
    {
        const std::size_t i0 = side[0] < 0 ? 0 : 1;
        const std::size_t j0 = side[1] < 0 ? 0 : 1;
        const double xi = side[0] < 0 ? 1.0 - 2.0 * HALF_SIDE / CELL : -1.0 + 2.0 * HALF_SIDE / CELL;
        const double eta = side[1] < 0 ? 1.0 - 2.0 * HALF_SIDE / CELL : -1.0 + 2.0 * HALF_SIDE / CELL;
        const oos::elements::BilinearShape shape = oos::elements::bilinearShapeAt(xi, eta);
        AuxiliaryPoint corner;
        corner.weight = 0.25;
        corner.gridIndices = {3 * j0 + i0, 3 * j0 + i0 + 1, 3 * (j0 + 1) + i0 + 1, 3 * (j0 + 1) + i0};
        corner.shapeValues.assign(shape.n.begin(), shape.n.end());
        patch.corners.push_back(corner);
    }
    return patch;
}

// The translation and the rotation a field gives to a grid point.
struct GridMotion
{
    Vec3 translation;
    Vec3 rotation;
};

// The motion of the piercing point for the motions `field` gives to the grid points.
template <typename Field>
std::array<double, 6> motionOf(const Patch& patch, const PiercingMap& map, Field field)
{
    std::array<double, 6> motion{};
    for (std::size_t g = 0; g < GRIDS; ++g)
    {
        const GridMotion at = field(patch.grids[g].position);
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                motion[r] += map.at(r, 6 * g + c) * at.translation[c] + map.at(r, 6 * g + 3 + c) * at.rotation[c];
    }
    return motion;
}

} // namespace

CHECK_CASE("fasteners: the tie over the four corners of the footprint")
{
    const Patch patch = squareAboutGridPoint();
    const std::variant<PiercingMap, std::string> result =
        oos::elements::interpolateFootprint(patch.point, patch.grids, patch.corners);
    runner.expect(std::holds_alternative<PiercingMap>(result), "the four corners give a map");
    if (!std::holds_alternative<PiercingMap>(result))
        return;
    const PiercingMap& map = std::get<PiercingMap>(result);

    // A rigid motion of the grid points is the rigid motion of the piercing point.
    const Vec3 translation{0.3, -0.2, 0.5};
    const Vec3 rotation{0.01, -0.02, 0.03};
    const std::array<double, 6> rigid = motionOf(patch, map, [&](Vec3 x) {
        return GridMotion{translation + oos::base::cross(rotation, x), rotation};
    });
    const Vec3 expected = translation + oos::base::cross(rotation, patch.point);
    for (std::size_t c = 0; c < 3; ++c)
    {
        runner.expectNear(rigid[c], expected[c], 1.0e-12, "rigid motion: translation");
        runner.expectNear(rigid[3 + c] / 0.01, rotation[c] / 0.01, 1.0e-10, "rigid motion: rotation");
    }

    // A linear field of translations u = a + B·x with no rotation of the grid points: the piercing point has the
    // translation of the field and the rotation of the sheet in its plane, θ_z = ½(∂v/∂x − ∂u/∂y); the slopes of
    // the deflection, ∂w/∂y and −∂w/∂x, are not its rotations about x and y (those of the grid points are).
    using Slope = std::array<double, 2>;
    const std::array<Slope, 3> slope{Slope{0.004, -0.003}, Slope{0.002, 0.005}, Slope{-0.006, 0.007}};
    const std::array<double, 6> linear = motionOf(patch, map, [&](Vec3 x) {
        return GridMotion{Vec3{slope[0][0] * x[0] + slope[0][1] * x[1], slope[1][0] * x[0] + slope[1][1] * x[1],
                               slope[2][0] * x[0] + slope[2][1] * x[1]},
                          Vec3{}};
    });
    for (std::size_t c = 0; c < 3; ++c)
        runner.expectNear(linear[c], (slope[c][0] + slope[c][1]) * CELL, 1.0e-12, "linear field: translation");
    runner.expectNear(linear[3] / 0.001, 0.0, 1.0e-10, "linear field: rotation about x");
    runner.expectNear(linear[4] / 0.001, 0.0, 1.0e-10, "linear field: rotation about y");
    runner.expectNear(linear[5] / 0.001, 0.5 * (slope[1][0] - slope[0][1]) / 0.001, 1.0e-10,
                      "linear field: rotation about the normal");

    // Rotations of the grid points linear in x and y, without translation: the piercing point has those of the
    // field about x and y (the mean of the four corners of the square, which is the field at its centre), and none
    // about the normal, whatever the grid points have there.
    const std::array<double, 6> turned = motionOf(patch, map, [&](Vec3 x) {
        return GridMotion{Vec3{}, Vec3{0.01 + 0.002 * x[0], -0.02 + 0.003 * x[1], 0.5 + 0.1 * x[0]}};
    });
    runner.expectNear(turned[3] / 0.01, (0.01 + 0.002 * CELL) / 0.01, 1.0e-10, "rotations: about x");
    runner.expectNear(turned[4] / 0.01, (-0.02 + 0.003 * CELL) / 0.01, 1.0e-10, "rotations: about y");
    runner.expectNear(turned[5], 0.0, 1.0e-12, "rotations: about the normal");
    for (std::size_t c = 0; c < 3; ++c)
        runner.expectNear(turned[c], 0.0, 1.0e-12, "rotations: translation");

    // A dome w = c·((x − X)² + (y − Y)²) about the piercing point, the grid points turned as its slopes (the sheet
    // bends alike on every side of the fastener): no rotation, and the deflection of the bilinear elements at the
    // corners of the square.
    const std::array<double, 6> dome = motionOf(patch, map, [&](Vec3 x) {
        const double dx = x[0] - patch.point[0];
        const double dy = x[1] - patch.point[1];
        return GridMotion{Vec3{0.0, 0.0, 1.0e-4 * (dx * dx + dy * dy)}, Vec3{2.0e-4 * dy, -2.0e-4 * dx, 0.0}};
    });
    runner.expectNear(dome[3] / 1.0e-4, 0.0, 1.0e-10, "dome: rotation about x");
    runner.expectNear(dome[4] / 1.0e-4, 0.0, 1.0e-10, "dome: rotation about y");
    // The elements hold the dome at their grid points: on a side through the middle grid point, at h from it, they
    // give c·CELL·h, and at the corner of the square twice that.
    runner.expectNear(dome[2], 2.0 * 1.0e-4 * CELL * HALF_SIDE, 1.0e-12, "dome: deflection");
}
