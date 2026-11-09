#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>

namespace oos::assembly
{

// The values of the shape functions of a face at one point, one per point of the face (in the face's order), the
// values of absent mid-side points 0.
struct FaceShapeValues
{
    std::array<double, 8> values{};
    std::size_t count = 0; // the points of the face: 3, 4, 6 or 8
};

// Local coordinates on a face.
struct FaceCoordinates
{
    double xi = 0.0;
    double eta = 0.0;
};

// The shape functions of the face of a shell or solid at the local coordinates (ξ, η) of a point on it (glued
// contact: u_s = Σ N_i(ξ, η)·u_i), for the faces of O_o_Reader's contactNodePairs: `face` the grid
// points in the element's order, the corners first, then the mid-side points in the order of the edges, 0 for an
// absent mid-side point. The same functions place the points of the face, x(ξ, η) = Σ N_i(ξ, η)·x_i (GlueSearch).
//
// - Quadrilateral (4 or 8 points), corner i at (ξ_i, η_i) = (−1, −1), (1, −1), (1, 1), (−1, 1); the mid-side points
//   on the edges 1-2, 2-3, 3-4, 4-1. The serendipity functions with a variable number of mid-side points (K.J. Bathe,
//   Finite Element Procedures, 1996, §5.3.1, the elements of 4 to 9 variable-number nodes):
//       N_5 = ½(1 − ξ²)(1 − η),  N_6 = ½(1 + ξ)(1 − η²),  N_7 = ½(1 − ξ²)(1 + η),  N_8 = ½(1 − ξ)(1 − η²),
//       N_i = ¼(1 + ξ·ξ_i)(1 + η·η_i) − ½·(N of the two mid-side points of corner i that are present),
//   which give the bilinear functions with no mid-side point and the 8-node serendipity ones with all four.
// - Triangle (3 or 6 points): ξ = L2, η = L3, L1 = 1 − L2 − L3; the mid-side points on the edges 1-2, 2-3, 3-1:
//       N_4 = 4·L1·L2,  N_5 = 4·L2·L3,  N_6 = 4·L3·L1,  N_i = L_i − ½·(N of the two mid-side points of corner i),
//   the linear functions with no mid-side point and the quadratic ones, L_i(2·L_i − 1), with all three (same source).
//
// The values are those of the formulas wherever (ξ, η) lies: outside the face they extrapolate (a negative weight or
// one above 1), which the caller avoids (GlueSearch ties at coordinates brought onto the face by clampToFace). Returns
// the values, or the reason the face cannot interpolate: a number of points other than 3, 4, 6, 8, an absent corner,
// coordinates that are not finite numbers.
[[nodiscard]] std::variant<FaceShapeValues, std::string> faceShapeFunctions(std::span<const std::int32_t> face,
                                                                            double xi, double eta);

// The local coordinates of a point brought back onto its face, for a face of `pointCount` points: on a quadrilateral ξ
// and η clamped to [−1, 1] each (the closest point of the parametric square); on a triangle a negative area
// coordinate set to 0 and L2, L3 scaled down when L2 + L3 exceeds 1 (onto the edge 2-3). Inside the face the
// coordinates are returned as they are.
[[nodiscard]] FaceCoordinates clampToFace(std::size_t pointCount, FaceCoordinates coordinates) noexcept;

} // namespace oos::assembly
