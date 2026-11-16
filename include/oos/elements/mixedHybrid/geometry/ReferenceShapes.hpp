#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace oos::elements::mixedHybrid
{

// The reference domains of the shell elements of the mixed hybrid formulation, shared by its spaces, its geometry and
// its elements: the square [-1, 1]² of a quadrilateral and the triangle (-1, -1), (1, -1), (-1, 1) of a triangle,
// corners counter-clockwise, edge k from corner k to corner k + 1.

using ReferencePoint = std::array<double, 2>;

// The corners of the reference square.
inline constexpr std::array<ReferencePoint, 4> REFERENCE_SQUARE{{{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}}};
// The corners of the reference triangle.
inline constexpr std::array<ReferencePoint, 3> REFERENCE_TRIANGLE{{{-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}}};

// Corner k (taken modulo the corner count) of the reference domain of `cornerCount` corners (3 or 4).
[[nodiscard]] constexpr ReferencePoint referenceCorner(std::size_t cornerCount, std::size_t k) noexcept
{
    return cornerCount == 3 ? REFERENCE_TRIANGLE[k % 3] : REFERENCE_SQUARE[k % 4];
}

// The centre of the reference domain: (0, 0) for the square, the centroid (-1/3, -1/3) for the triangle.
[[nodiscard]] constexpr ReferencePoint referenceCentre(std::size_t cornerCount) noexcept
{
    return cornerCount == 3 ? ReferencePoint{-1.0 / 3.0, -1.0 / 3.0} : ReferencePoint{0.0, 0.0};
}

// The barycentric coordinates of a point (u, v) of the reference triangle: L_k is 1 at corner k,
// L = (−(u + v)/2, (1 + u)/2, (1 + v)/2).
template <Scalar T>
[[nodiscard]] std::array<T, 3> barycentric(const T& u, const T& v)
{
    const T one = constantOf<T>(1.0);
    const T half = constantOf<T>(0.5);
    return {-(u + v) * half, (one + u) * half, (one + v) * half};
}

// The value at (ξ, η) of a field given at the corners: bilinear on the reference square, linear on the reference
// triangle (its barycentric coordinates).
[[nodiscard]] inline double cornerInterpolation(std::size_t cornerCount, const std::array<double, 4>& values, double xi,
                                                double eta)
{
    if (cornerCount == 3)
    {
        const std::array<double, 3> l = barycentric(xi, eta);
        return l[0] * values[0] + l[1] * values[1] + l[2] * values[2];
    }
    double v = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
        v += 0.25 * (1.0 + REFERENCE_SQUARE[k][0] * xi) * (1.0 + REFERENCE_SQUARE[k][1] * eta) * values[k];
    return v;
}

// The point of edge k of the reference domain at the parameter tau in [0, 1] (from corner k to corner k + 1).
template <Scalar T>
[[nodiscard]] std::array<T, 2> referenceEdgePoint(std::size_t cornerCount, std::size_t k, const T& tau)
{
    const ReferencePoint a = referenceCorner(cornerCount, k);
    const ReferencePoint b = referenceCorner(cornerCount, k + 1);
    return {constantOf<T>(a[0]) + scale(tau, b[0] - a[0]), constantOf<T>(a[1]) + scale(tau, b[1] - a[1])};
}

// The point of edge k at tau and the edge's direction in the reference domain (corner k + 1 − corner k).
[[nodiscard]] constexpr std::pair<ReferencePoint, ReferencePoint> referenceEdgePointAndDirection(
    std::size_t cornerCount, std::size_t k, double tau) noexcept
{
    const ReferencePoint a = referenceCorner(cornerCount, k);
    const ReferencePoint b = referenceCorner(cornerCount, k + 1);
    const ReferencePoint direction{b[0] - a[0], b[1] - a[1]};
    return {{a[0] + tau * direction[0], a[1] + tau * direction[1]}, direction};
}

// The orthonormal tangent frame (e1, e2) of a surface point of unit normal n that follows two reference axes x and y
// of the element: e1 = unit(P x + (P y) × n), P the projection on the tangent plane, and e2 = n × e1. (P y) × n is
// P y turned by −90° about n, so both axes count alike: on a flat element with x ⟂ y it is x itself, and on a
// curved one the frame turns with the surface without favouring x.
[[nodiscard]] inline std::array<std::array<double, 3>, 2> tangentFrameXy(const std::array<double, 3>& n,
                                                                         const std::array<double, 3>& x,
                                                                         const std::array<double, 3>& y)
{
    const std::array<double, 3> px = sub3(x, scl3(n, dot3(x, n)));
    const std::array<double, 3> py = sub3(y, scl3(n, dot3(y, n)));
    const std::array<double, 3> e1 = unit3(add3(px, cross3(py, n)));
    return {e1, cross3(n, e1)};
}

} // namespace oos::elements::mixedHybrid
