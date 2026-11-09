#pragma once

#include <array>
#include <cmath>

namespace oos::elements
{

// A point of a quadrature rule on the parent square [−1, 1]² (ξ, η) and its weight.
struct QuadraturePoint
{
    double xi = 0.0;
    double eta = 0.0;
    double weight = 0.0;
};

// 1/√3, the abscissa of the two-point Gauss-Legendre rule (exact for cubics on [−1, 1]).
inline const double GAUSS_TWO_POINT_ABSCISSA = 1.0 / std::sqrt(3.0);

// The 2 × 2 Gauss-Legendre rule on [−1, 1]², points at (±1/√3, ±1/√3), weights 1. The points are listed in the order
// of the corners of a quadrilateral (CQUAD4 G1 to G4: (−,−), (+,−), (+,+), (−,+)), so that point k is the one nearest
// to corner k: the bilinear extrapolation from the points to the corners (shells' corner outputs) relies on it.
[[nodiscard]] inline std::array<QuadraturePoint, 4> gaussRule2x2()
{
    const double g = GAUSS_TWO_POINT_ABSCISSA;
    return {QuadraturePoint{-g, -g, 1.0}, QuadraturePoint{g, -g, 1.0}, QuadraturePoint{g, g, 1.0},
            QuadraturePoint{-g, g, 1.0}};
}

} // namespace oos::elements
