#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The parent shapes of the solid elements, with and without mid-side nodes (CTETRA, SolidPenta, SolidPyramid, SolidHexa).
// The nodes are numbered as on the cards: the corners first, then the mid-side nodes edge by edge (solidEdges()).
enum class SolidShape : std::uint8_t
{
    Tetra4,
    Tetra10,
    Penta6,
    Penta15,
    Pyram5,
    Pyram13,
    Hexa8,
    Hexa20
};

// The most nodes of a solid element (CHEXA with its twelve mid-side nodes).
inline constexpr std::size_t MAX_SOLID_NODES = 20;

// A point of the parent element. Its coordinates depend on the shape:
// - Tetra: the volume coordinates (L2, L3, L4) of corners G2, G3, G4, L1 = 1 − L2 − L3 − L4;
// - Penta: the area coordinates (L2, L3) of G2, G3 in the triangles G1-G2-G3 and G4-G5-G6, and ζ ∈ [−1, 1] from the
//   face G1-G2-G3 (ζ = −1) to the face G4-G5-G6 (ζ = 1);
// - Hexa and Pyram: (ξ, η, ζ) ∈ [−1, 1]³, the pyramid being the hexahedron whose face ζ = 1 is collapsed into the
//   apex G5 (Zienkiewicz & Taylor, The Finite Element Method, vol. 1, §8.9: degenerate elements).
struct ParentPoint
{
    double xi = 0.0;
    double eta = 0.0;
    double zeta = 0.0;
};

// The shape functions of a solid at a parent point and their derivatives with respect to the parent coordinates;
// the first `count` entries are used.
struct SolidShapeValues
{
    std::size_t count = 0;
    std::array<double, MAX_SOLID_NODES> n{};
    std::array<double, MAX_SOLID_NODES> dXi{};
    std::array<double, MAX_SOLID_NODES> dEta{};
    std::array<double, MAX_SOLID_NODES> dZeta{};
};

// The isoparametric shape functions (Zienkiewicz & Taylor vol. 1 §8.7-8.9):
// - Tetra4 N_i = L_i; Tetra10 N_i = L_i(2L_i − 1) at the corners, 4 L_a L_b at the mid-side node of edge a-b;
// - Penta6 N = L_i(1 ∓ ζ)/2; Penta15 ½L_i(2L_i − 1)(1 ∓ ζ) − ½L_i(1 − ζ²) at the corners, 2 L_a L_b (1 ∓ ζ) at the
//   mid-side nodes of the triangles, L_i (1 − ζ²) at those of the vertical edges;
// - Hexa8 ⅛(1 + ξ_iξ)(1 + η_iη)(1 + ζ_iζ); Hexa20 the serendipity functions ⅛(1 + ξ_iξ)(1 + η_iη)(1 + ζ_iζ)
//   (ξ_iξ + η_iη + ζ_iζ − 2) at the corners and ¼(1 − ξ²)(1 + η_iη)(1 + ζ_iζ) (and permutations) at the mid-side nodes;
// - Pyram5 and Pyram13 the Hexa8 and Hexa20 functions with those of the collapsed face ζ = 1 added into the apex.
[[nodiscard]] SolidShapeValues solidShapeAt(SolidShape shape, ParentPoint point);

[[nodiscard]] std::size_t solidNodeCount(SolidShape shape) noexcept;
[[nodiscard]] std::size_t solidCornerCount(SolidShape shape) noexcept;
// Whether the shape has its mid-side nodes.
[[nodiscard]] bool isQuadraticSolid(SolidShape shape) noexcept;

// The parent coordinates of the corners, in the order of the card.
[[nodiscard]] std::span<const ParentPoint> solidCornerPoints(SolidShape shape) noexcept;
// The parent point taken as the centre of the element (its "CENTER" output row): the centroid of the parent element
// (L = ¼ for a tetrahedron, (⅓, ⅓, 0) for a pentahedron, (0, 0, 0) for a hexahedron and for the collapsed pyramid).
[[nodiscard]] ParentPoint solidCentre(SolidShape shape) noexcept;

// An edge of a solid: its two corners and the mid-side node on it (indices into the card's node list, from 0).
struct SolidEdge
{
    std::uint8_t first = 0;
    std::uint8_t second = 0;
    std::uint8_t middle = 0;
};
// The edges in the order of the mid-side nodes of the card (CTETRA G5-G10, SolidPenta G7-G15, SolidPyramid G6-G13,
// SolidHexa G9-G20), for either shape of a family (the linear shape gives the same edges, their middle being where the
// mid-side node would be).
[[nodiscard]] std::span<const SolidEdge> solidEdges(SolidShape shape) noexcept;

// A face of a solid: its corners (3 or 4, indices from 0), ordered so that (c1 − c0) × (c2 − c0) points out of a
// parent element of positive Jacobian.
struct SolidFace
{
    std::uint8_t cornerCount = 0;
    std::array<std::uint8_t, 4> corners{};
};
[[nodiscard]] std::span<const SolidFace> solidFaces(SolidShape shape) noexcept;

} // namespace oos::elements
