#pragma once

#include "oos/elements/core/IntegrationRules.hpp"

#include <array>
#include <cstddef>

namespace oos::elements
{

// The shape functions and integration rules of the higher-order shells, each a policy of
// MindlinShell: N nodes (the corners first, in the order of the card, then the edge nodes), CORNERS corners, the shape
// functions and their parent derivatives at a point, the parent coordinates of the nodes and of the centre, the
// integration rule of the stiffness (rule(): the reduced rule of the references of MindlinShell) and the full rule of
// the mass and the pressure loads (massRule(): exact for N_i N_j times a linear thickness on a flat element of
// straight edges).

// The values and parent derivatives of the shape functions at a point.
template <std::size_t N>
struct ShapeValues
{
    std::array<double, N> n{};
    std::array<double, N> dXi{};
    std::array<double, N> dEta{};
};

// ShellQuad8: the eight-node serendipity quadrilateral (Zienkiewicz & Taylor, The Finite Element Method, vol. 1, §8.5):
// corners N_i = ¼(1 + ξξ_i)(1 + ηη_i)(ξξ_i + ηη_i − 1), edge nodes N_i = ½(1 − ξ²)(1 + ηη_i) (η_i = ±1) or
// ½(1 + ξξ_i)(1 − η²) (ξ_i = ±1); G5 on G1-G2, G6 on G2-G3, G7 on G3-G4, G8 on G4-G1. Stiffness: the uniform
// reduced 2 × 2 Gauss rule (Zienkiewicz, Taylor & Too 1971); mass and pressure: the 3 × 3 Gauss rule (exact to degree
// five in each direction).
struct SerendipityQuadShape
{
    static constexpr std::size_t NODES = 8;
    static constexpr std::size_t CORNERS = 4;
    static constexpr std::array<double, 8> XI{-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0};
    static constexpr std::array<double, 8> ETA{-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0};
    static constexpr double CENTRE_XI = 0.0;
    static constexpr double CENTRE_ETA = 0.0;

    [[nodiscard]] static ShapeValues<8> at(double xi, double eta);
    [[nodiscard]] static std::array<QuadraturePoint, 4> rule() { return gaussRule2x2(); }
    [[nodiscard]] static std::array<QuadraturePoint, 9> massRule();
    // The parent area of the element (the weights of its rule sum to it): 4.
    static constexpr double PARENT_AREA = 4.0;
};

// ShellTria6: the six-node quadratic triangle in the area coordinates L1 = 1 − ξ − η, L2 = ξ, L3 = η (Zienkiewicz &
// Taylor vol. 1, §8.8): corners N_i = L_i(2L_i − 1), edge nodes N = 4 L_i L_j; G4 on G1-G2, G5 on G2-G3, G6 on G3-G1.
// Stiffness: the three-point rule at (1/6, 1/6), (2/3, 1/6), (1/6, 2/3), weights 1/6 each (exact for quadratics: the
// membrane and the bending exactly, the transverse shear reduced). Mass and pressure: the seven-point rule exact for
// quintics (Hammer, Marlowe & Stroud, "Numerical integration over simplexes and cones", Math. Tables Aids Comput. 10
// (1956) 130-137; tabulated by Cowper, IJNME 7 (1973) 405-408).
struct QuadraticTriangleShape
{
    static constexpr std::size_t NODES = 6;
    static constexpr std::size_t CORNERS = 3;
    static constexpr std::array<double, 6> XI{0.0, 1.0, 0.0, 0.5, 0.5, 0.0};
    static constexpr std::array<double, 6> ETA{0.0, 0.0, 1.0, 0.0, 0.5, 0.5};
    static constexpr double CENTRE_XI = 1.0 / 3.0;
    static constexpr double CENTRE_ETA = 1.0 / 3.0;

    [[nodiscard]] static ShapeValues<6> at(double xi, double eta);
    [[nodiscard]] static std::array<QuadraturePoint, 3> rule();
    [[nodiscard]] static std::array<QuadraturePoint, 7> massRule();
    static constexpr double PARENT_AREA = 0.5;
};

} // namespace oos::elements
