#pragma once

#include "oos/elements/core/SolidShapeFunctions.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace oos::elements
{

// A point of a quadrature rule on a parent solid (ParentPoint coordinates) and its weight.
struct SolidQuadraturePoint
{
    ParentPoint point;
    double weight = 0.0;
};

// A quadrature rule on a parent solid: at most 64 points (the 4 × 4 × 4 rules). An immutable value; the functions
// below build each rule once (function-local constants) and hand out references to it.
struct SolidRule
{
    std::size_t count = 0;
    std::array<SolidQuadraturePoint, 64> points{};

    [[nodiscard]] std::span<const SolidQuadraturePoint> all() const noexcept { return {points.data(), count}; }
};

// The n × n × n Gauss-Legendre product rule on [−1, 1]³, n from 1 to 4 (exact for polynomials of degree 2n − 1 in each
// coordinate). Throws std::invalid_argument for another n.
[[nodiscard]] const SolidRule& hexahedronRule(std::size_t pointsPerAxis);

// A rule on the parent tetrahedron (volume coordinates L2, L3, L4; weights summing to its volume 1/6) exact for
// polynomials of total degree `degree` (1 to 5): the centroid for degree 1, the classic four-point rule
// L = (0.5854102, 0.1381966, ...) for degree 2 (Zienkiewicz & Taylor vol. 1, table 8.3), Keast's five-point rule for
// degree 3 (one negative weight), and for degrees 4 and 5 the conical product (collapsed-cube) rule of Stroud with
// Gauss-Legendre points (Stroud, Approximate Calculation of Multiple Integrals, 1971, §2.7). Throws
// std::invalid_argument for another degree.
[[nodiscard]] const SolidRule& tetrahedronRule(std::size_t degree);

// A rule on the parent pentahedron: a triangle rule exact for total degree `triangleDegree` (1 to 6; the centroid,
// the three mid-edge-interior points (1/6, 1/6), (2/3, 1/6), (1/6, 2/3) for degree 2, the collapsed Gauss product
// above that) times the `zetaPoints`-point Gauss-Legendre rule along ζ (1 to 4). Throws std::invalid_argument
// outside these ranges.
[[nodiscard]] const SolidRule& pentahedronRule(std::size_t triangleDegree, std::size_t zetaPoints);

// The abscissae and weights of the n-point Gauss-Legendre rule on [−1, 1], n from 1 to 4 (Abramowitz & Stegun 25.4.29).
struct GaussLegendreRule
{
    std::size_t count = 0;
    std::array<double, 4> abscissae{};
    std::array<double, 4> weights{};
};
[[nodiscard]] GaussLegendreRule gaussLegendreRule(std::size_t points);

} // namespace oos::elements
