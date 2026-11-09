#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ShapeFunctions.hpp"

#include <array>
#include <cstdint>

namespace oos::elements
{

// The mean plane is degenerate when ‖d1 × d2‖ ≤ this factor · ‖d1‖·‖d2‖ (parallel
// diagonals: the four points are on a line or two of them coincide).
inline constexpr double QUAD_DEGENERATE_NORMAL_FACTOR = 1.0e-12;
// When ‖e31 − e42‖ is below this, the bisector is undefined and x = x_u.
inline constexpr double QUAD_BISECTOR_TOLERANCE = 1.0e-15;
// The element is warped when max |z_i^d| > this factor · max(extent in x, extent in y).
inline constexpr double QUAD_WARP_TOLERANCE = 1.0e-10;
// A warp max |z_i^d| / diagonal above this is reported.
inline constexpr double QUAD_WARP_REPORT_RATIO = 0.05;
// A corner whose Jacobian determinant is at or below this fraction of the area / 4 (its value on a parallelogram)
// makes the bilinear map fold or collapse there: a re-entrant corner, or two coincident grid points. Such an element
// has no valid isoparametric map and is refused (a collapsed ShellQuad4 is a ShellTria3 written as a quadrilateral).
inline constexpr double QUAD_MIN_CORNER_JACOBIAN_RATIO = 1.0e-10;

// The geometry of a quadrilateral plate (ShellQuad4DrillingPlate) in its element frame: the frame (x, y, n), the plane
// coordinates of the four corners, their heights off the mean plane, the half-widths A and B, and the Jacobian at the
// centre and the corners.
struct ShellQuad4DrillingGeometry
{
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, n in the basic system
    base::Vec3 centre;                        // the mean of the four grid points, in basic
    std::array<double, 4> x{};                // (X_i − centre)·x
    std::array<double, 4> y{};                // (X_i − centre)·y
    std::array<double, 4> zDeviation{};       // z_i^d = z_i − z̄; exactly 0 when the element is not warped
    bool isWarped = false;
    double warpRatio = 0.0;                       // max |z_i^d| / the mean length of the two diagonals
    double widthA = 0.0;                          // A = ½ |x2 + x3 − x1 − x4|
    double widthB = 0.0;                          // B = ½ |y3 + y4 − y1 − y2|
    double edgeAngle = 0.0;                       // φ = atan2(y2 − y1, x2 − x1): the edge G1→G2 from the x axis
    PlaneJacobian centreJacobian;                 // J0 = J(0, 0)
    std::array<PlaneJacobian, 4> cornerJacobians; // J(ξ_i, η_i) at corner i

    // The element area 4·det J0 (exact for any plane quadrilateral: det J is bilinear with a zero mean of its
    // non-constant terms over the parent square).
    [[nodiscard]] double area() const noexcept { return 4.0 * centreJacobian.determinant; }
    // The Jacobian of the plane map at (ξ, η).
    [[nodiscard]] PlaneJacobian jacobianAt(double xi, double eta) const
    {
        return planeJacobianAt(bilinearShapeAt(xi, eta), x, y);
    }
};

// Why a quadrilateral has no geometry.
enum class QuadGeometryProblem : std::uint8_t
{
    None,
    DegenerateNormal, // parallel or null diagonals
    FoldedCorner      // a corner Jacobian at or below QUAD_MIN_CORNER_JACOBIAN_RATIO (re-entrant or collapsed)
};

struct ShellQuad4DrillingGeometryResult
{
    ShellQuad4DrillingGeometry geometry;
    QuadGeometryProblem problem = QuadGeometryProblem::None;
    std::int32_t corner = 0; // the corner (0 to 3) of a FoldedCorner
};

// The geometry of a quadrilateral from the positions of G1 to G4 in the basic system.
[[nodiscard]] ShellQuad4DrillingGeometryResult buildShellQuad4DrillingGeometry(const std::array<base::Vec3, 4>& positions);

} // namespace oos::elements
