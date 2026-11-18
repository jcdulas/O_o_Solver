#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace oos::elements
{

// The geometry of the higher-order shells ShellQuad8 and ShellTria6 from their grid points alone (HigherOrderShell).
// The directors their nodes share at the grid points of a model are those of ShellDirectors.hpp.

// A higher-order shell whose height off the mean plane of its corners is at most this fraction of its size is flat.
inline constexpr double HIGHER_ORDER_SHELL_FLATNESS = 1.0e-8;
// The smallest cosine allowed between the surface normal of a curved element and its element normal: cos 45°.
inline constexpr double HIGHER_ORDER_SHELL_MIN_NORMAL_COSINE = 0.70710678118654752;

// The geometry of one element: the element frame of its corners (z along the cross product of the diagonals of a
// ShellQuad8, the normal of the corners of a ShellTria6; x along the edge G1→G2 projected on the plane normal to z;
// origin at the mean of the corners), its nodes in that frame (projected on its plane when the element is flat) and
// its own directors there: the unit normal of its isoparametric surface at each node
// (MindlinShell:: nodalDirectors), the element normal n at every node of a flat element.
template <std::size_t N>
struct HigherOrderShellGeometry
{
    ShellCorners<N> nodes;                    // the grid points (rows, IDs, basic positions, CD frames)
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, n of the element frame in basic
    double edgeAngle = 0.0;                   // the edge G1→G2 from the x axis (the reference of THETA)
    bool isCurved = false;
    std::array<base::Vec3, N> position{};      // in the element frame; z = 0 on a flat element
    std::array<base::Vec3, N> ownDirector{};   // in the element frame
};

// The geometry of the element of batch row `row` (Shape: SerendipityQuadShape or QuadraticTriangleShape); refused: a
// blank edge grid point, an unknown grid point, corners on a line, a map whose Jacobian on the element plane is not
// positive, a curved element whose normal turns by more than 45° from the plane of its corners.
template <typename Shape>
[[nodiscard]] Resolution<HigherOrderShellGeometry<Shape::NODES>> resolveHigherOrderShellGeometry(
    const ElementBatchView& view, const ShellConnectivity& connectivity, std::int64_t row);

} // namespace oos::elements
