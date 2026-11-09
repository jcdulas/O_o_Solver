#include "oos/elements/shells/HigherOrderShapes.hpp"

#include <cmath>

namespace oos::elements
{

ShapeValues<8> SerendipityQuadShape::at(double xi, double eta)
{
    ShapeValues<8> s;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const double a = XI[i];
        const double b = ETA[i];
        s.n[i] = 0.25 * (1.0 + a * xi) * (1.0 + b * eta) * (a * xi + b * eta - 1.0);
        s.dXi[i] = 0.25 * a * (1.0 + b * eta) * (2.0 * a * xi + b * eta);
        s.dEta[i] = 0.25 * b * (1.0 + a * xi) * (a * xi + 2.0 * b * eta);
    }
    for (std::size_t i = 4; i < 8; ++i)
    {
        const double a = XI[i];
        const double b = ETA[i];
        if (a == 0.0)
        {
            s.n[i] = 0.5 * (1.0 - xi * xi) * (1.0 + b * eta);
            s.dXi[i] = -xi * (1.0 + b * eta);
            s.dEta[i] = 0.5 * b * (1.0 - xi * xi);
        }
        else
        {
            s.n[i] = 0.5 * (1.0 + a * xi) * (1.0 - eta * eta);
            s.dXi[i] = 0.5 * a * (1.0 - eta * eta);
            s.dEta[i] = -eta * (1.0 + a * xi);
        }
    }
    return s;
}

std::array<QuadraturePoint, 9> SerendipityQuadShape::massRule()
{
    // The three-point Gauss-Legendre rule in each direction: abscissae 0, ±√(3/5), weights 8/9, 5/9.
    const double abscissa = std::sqrt(0.6);
    const std::array<double, 3> points{-abscissa, 0.0, abscissa};
    const std::array<double, 3> weights{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    std::array<QuadraturePoint, 9> rule{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            rule[3 * i + j] = QuadraturePoint{points[j], points[i], weights[i] * weights[j]};
    return rule;
}

std::array<QuadraturePoint, 7> QuadraticTriangleShape::massRule()
{
    // In the area coordinates: the centroid, and the two orbits (a, a, 1 − 2a) of a = (6 ∓ √15)/21; the weights are
    // those of the reference rule (sum 1) times the parent area ½.
    const double root = std::sqrt(15.0);
    const double a1 = (6.0 - root) / 21.0;
    const double b1 = 1.0 - 2.0 * a1;
    const double w1 = (155.0 - root) / 2400.0;
    const double a2 = (6.0 + root) / 21.0;
    const double b2 = 1.0 - 2.0 * a2;
    const double w2 = (155.0 + root) / 2400.0;
    constexpr double CENTROID_WEIGHT = 9.0 / 80.0;
    return {QuadraturePoint{1.0 / 3.0, 1.0 / 3.0, CENTROID_WEIGHT},
            QuadraturePoint{a1, a1, w1},
            QuadraturePoint{b1, a1, w1},
            QuadraturePoint{a1, b1, w1},
            QuadraturePoint{a2, a2, w2},
            QuadraturePoint{b2, a2, w2},
            QuadraturePoint{a2, b2, w2}};
}

ShapeValues<6> QuadraticTriangleShape::at(double xi, double eta)
{
    // L1 = 1 − ξ − η, L2 = ξ, L3 = η; ∂L1/∂ξ = ∂L1/∂η = −1.
    const double l1 = 1.0 - xi - eta;
    const double l2 = xi;
    const double l3 = eta;
    ShapeValues<6> s;
    s.n = {l1 * (2.0 * l1 - 1.0), l2 * (2.0 * l2 - 1.0), l3 * (2.0 * l3 - 1.0),
           4.0 * l1 * l2,         4.0 * l2 * l3,         4.0 * l3 * l1};
    s.dXi = {-(4.0 * l1 - 1.0), 4.0 * l2 - 1.0, 0.0, 4.0 * (l1 - l2), 4.0 * l3, -4.0 * l3};
    s.dEta = {-(4.0 * l1 - 1.0), 0.0, 4.0 * l3 - 1.0, -4.0 * l2, 4.0 * l2, 4.0 * (l1 - l3)};
    return s;
}

std::array<QuadraturePoint, 3> QuadraticTriangleShape::rule()
{
    constexpr double SIXTH = 1.0 / 6.0;
    return {QuadraturePoint{SIXTH, SIXTH, SIXTH}, QuadraturePoint{2.0 / 3.0, SIXTH, SIXTH},
            QuadraturePoint{SIXTH, 2.0 / 3.0, SIXTH}};
}

} // namespace oos::elements
