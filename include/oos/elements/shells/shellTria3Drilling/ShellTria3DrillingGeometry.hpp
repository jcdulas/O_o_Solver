#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"

#include <array>

namespace oos::elements
{

// A triangle is degenerate when ‖(x2 − x1) × (x3 − x1)‖ ≤ this factor · ‖x2 − x1‖·‖x3 − x1‖ (collinear or coincident
// grid points: no normal).
inline constexpr double TRIA_DEGENERATE_FACTOR = 1.0e-12;
// Two edges of a triangle are of one length when they differ by at most this share of it: the round-off of lengths
// computed from the corners in another order (1e-16), far below any meshed difference.
inline constexpr double TRIA_EDGE_TIE_TOLERANCE = 1.0e-10;

// The geometry of a triangular plate (ShellTria3DrillingPlate) in its element frame: x = (x2 − x1)/‖x2 − x1‖,
// n = (x2 − x1) × (x3 − x1) normalised, y = n × x; the plane coordinates of the corners, the area, the derivatives of
// the linear shape functions, and the widths A (the mean width A_e / B) and B (the height y_max − y_min).
struct ShellTria3DrillingGeometry
{
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, n in the basic system
    base::Vec3 centre;                        // the centroid, in basic
    std::array<base::Vec3, 3> positions{};    // G1 to G3, in basic
    std::array<double, 3> x{};                // (X_i − centre)·x
    std::array<double, 3> y{};                // (X_i − centre)·y
    double area = 0.0;                        // A_e
    double widthA = 0.0;                      // A = A_e / B
    double widthB = 0.0;                      // B = y_max − y_min
    std::array<double, 3> dX{};               // ∂N_i/∂x of the linear shape functions (constant)
    std::array<double, 3> dY{};               // ∂N_i/∂y
};

struct ShellTria3DrillingGeometryResult
{
    ShellTria3DrillingGeometry geometry;
    bool isDegenerate = false;
};

// The geometry of a triangle from the positions of G1 to G3 in the basic system.
[[nodiscard]] ShellTria3DrillingGeometryResult buildShellTria3DrillingGeometry(const std::array<base::Vec3, 3>& positions);

} // namespace oos::elements
