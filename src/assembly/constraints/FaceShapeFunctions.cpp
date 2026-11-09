#include "oos/assembly/constraints/FaceShapeFunctions.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::assembly
{

namespace
{

constexpr std::size_t QUAD_CORNERS = 4;
constexpr std::size_t TRIANGLE_CORNERS = 3;

// The parametric coordinates of the corners of a quadrilateral face.
constexpr std::array<std::array<double, 2>, QUAD_CORNERS> QUAD_CORNER_COORDINATES{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};

// Subtracts half of each present mid-side function from the two corners of its edge (edge k joins corners k and
// k + 1, cyclically): the variable-number-nodes correction of the header.
void correctCorners(FaceShapeValues& shape, std::span<const std::int32_t> face, std::size_t corners)
{
    for (std::size_t edge = 0; edge < corners; ++edge)
    {
        const std::size_t middle = corners + edge;
        if (face[middle] == 0)
        {
            shape.values[middle] = 0.0;
            continue;
        }
        shape.values[edge] -= 0.5 * shape.values[middle];
        shape.values[(edge + 1) % corners] -= 0.5 * shape.values[middle];
    }
}

} // namespace

std::variant<FaceShapeValues, std::string> faceShapeFunctions(std::span<const std::int32_t> face, double xi, double eta)
{
    const std::size_t count = face.size();
    if (count != 3 && count != 4 && count != 6 && count != 8)
        return base::concatenate({"expected a face of 3, 4, 6 or 8 grid points, found ", std::to_string(count)});
    const std::size_t corners = count == 3 || count == 6 ? TRIANGLE_CORNERS : QUAD_CORNERS;
    for (std::size_t k = 0; k < corners; ++k)
        if (face[k] == 0)
            return base::concatenate({"corner ", std::to_string(k + 1), " of the face is absent (0)"});
    if (!std::isfinite(xi) || !std::isfinite(eta))
        return "the local coordinates of the projection are not finite numbers";

    FaceShapeValues shape;
    shape.count = count;
    if (corners == QUAD_CORNERS)
    {
        for (std::size_t k = 0; k < QUAD_CORNERS; ++k)
            shape.values[k] =
                0.25 * (1.0 + xi * QUAD_CORNER_COORDINATES[k][0]) * (1.0 + eta * QUAD_CORNER_COORDINATES[k][1]);
        if (count == 8)
        {
            shape.values[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta);
            shape.values[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta);
            shape.values[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta);
            shape.values[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta);
            correctCorners(shape, face, QUAD_CORNERS);
        }
        return shape;
    }
    const double l1 = 1.0 - xi - eta;
    const double l2 = xi;
    const double l3 = eta;
    shape.values[0] = l1;
    shape.values[1] = l2;
    shape.values[2] = l3;
    if (count == 6)
    {
        shape.values[3] = 4.0 * l1 * l2;
        shape.values[4] = 4.0 * l2 * l3;
        shape.values[5] = 4.0 * l3 * l1;
        correctCorners(shape, face, TRIANGLE_CORNERS);
    }
    return shape;
}

FaceCoordinates clampToFace(std::size_t pointCount, FaceCoordinates coordinates) noexcept
{
    if (pointCount == 4 || pointCount == 8)
        return {std::clamp(coordinates.xi, -1.0, 1.0), std::clamp(coordinates.eta, -1.0, 1.0)};
    double l2 = std::max(coordinates.xi, 0.0);
    double l3 = std::max(coordinates.eta, 0.0);
    const double sum = l2 + l3;
    if (sum > 1.0)
    {
        l2 /= sum;
        l3 /= sum;
    }
    return {l2, l3};
}

} // namespace oos::assembly
