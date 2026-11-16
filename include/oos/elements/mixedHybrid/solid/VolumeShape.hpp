#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The four shapes of the solid elements of the mixed hybrid formulation (document 35 of the reference; the
// specification, section 3.3) and what is written on their reference elements: corners, edges and faces, the space of
// each shape (SolidSpaces.hpp), the bubble factor, the rule, and the functions of the faces.
//
// | shape       | reference                               | space of degree n      | faces                         |
// |-------------|-----------------------------------------|------------------------|-------------------------------|
// | tetrahedron | the corners below                       | P_n                    | 4 triangles                   |
// | pyramid     | base [−1, 1]² at z = −1, apex (0, 0, 1) | rational (Bergot 2010) | 1 quadrilateral, 4 triangles  |
// | pentahedron | reference triangle × [−1, 1]            | P_n(triangle) ⊗ P_n(z) | 2 triangles, 3 quadrilaterals |
// | hexahedron  | [−1, 1]³                                | Q_n                    | 6 quadrilaterals              |
//
// Corner orders (those of the input format):
// - tetrahedron (SolidTetra G1-G4): (−1,−1,−1), (1,−1,−1), (−1,1,−1), (−1,−1,1);
// - pyramid (SolidPyramid G1-G4 the base, G5 the apex): (−1,−1,−1), (1,−1,−1), (1,1,−1), (−1,1,−1), (0,0,1);
// - pentahedron (SolidPenta G1-G3 one triangle, G4-G6 the other, G4 opposite G1): (−1,−1,−1), (1,−1,−1), (−1,1,−1),
//   then the same at z = 1;
// - hexahedron (SolidHexa G1-G4 one face, G5-G8 the opposite one, G5 opposite G1): (−1,−1,−1), (1,−1,−1), (1,1,−1),
//   (−1,1,−1), then the same at z = 1.
// The edges are listed in the order of the mid-side nodes of the cards (SolidTetra G5-G10, SolidPyramid G6-G13,
// SolidPenta G7-G15, SolidHexa G9-G20): mid-side node k is on edge k. The faces list their corners in a cyclic order.
//
// The functions of a face (faceFunctions) are written from the numbers of its grid points, so that two elements of any
// shapes that share a face see the same functions on it (document 35, section 3.1):
// - triangle, corners a < b < c by number, barycentric L: L_i at the corners; on the edge (i, j), i the lower number,
//   L_i L_j S_m(L_j − L_i, L_i + L_j), m = 0 … p − 2, S_m the homogeneous Legendre polynomial (scaledJacobi); inside
//   L_a L_b L_c (L_b − L_a)^m (L_c − L_a)^n, m + n ≤ pf − 3 (those of the tetrahedron of document 28);
// - quadrilateral: bilinear at the corners; on an edge s (1 − s) P_m(2s − 1), s measured from its lower number, times
//   the linear function that is 1 on the edge and 0 on the opposite edge; inside S (1 − S) T (1 − T) P_m(2S − 1)
//   P_n(2T − 1), m, n ≤ pf − 2, S from the lowest number of the face towards the lower of its two neighbours, T
//   towards the other.
// An edge sees s (1 − s) P_m(2s − 1) from every face and element around it (on an edge L_i + L_j = 1 and
// L_j − L_i = 2s − 1 with s = L_j measured from the lower number i), and the functions of a quadrilateral do not depend
// on the corner it is given from nor on its direction.
//
// Every space has P_n on its triangular faces and Q_n on its quadrilateral faces: the traces of the space of degree p
// are the spans of these face functions, so that two elements that share a face are conforming there.
//
// Thread-safety: constant tables and pure functions.

// Reference: volume::Shape
enum class VolumeShape
{
    Tetrahedron,
    Pyramid,
    Pentahedron,
    Hexahedron,
};

// An edge of a shape: its two corners.
using ShapeEdge = std::array<std::size_t, 2>;

// A face of a shape: its corners in a cyclic order (3 or 4 of them).
struct ShapeFace
{
    std::size_t cornerCount = 0;
    std::array<std::size_t, 4> corners{};

    [[nodiscard]] constexpr std::span<const std::size_t> cornerSpan() const { return {corners.data(), cornerCount}; }
};

// The shape of `cornerCount` corners (4, 5, 6 or 8); empty otherwise.
// Reference: volume::Shape::of
[[nodiscard]] std::optional<VolumeShape> volumeShapeOfCornerCount(std::size_t cornerCount);

// "tetrahedron", "pyramid", "pentahedron", "hexahedron".
// Reference: volume::Shape::name
[[nodiscard]] const char* shapeName(VolumeShape shape);

// The corners of the reference element, in the order of the input format.
// Reference: volume::Shape::vertices (TET_V, PYR_V, PEN_V, HEX_V)
[[nodiscard]] std::span<const SolidPoint> shapeVertices(VolumeShape shape);

// The number of corners: 4, 5, 6, 8.
// Reference: volume::Shape::nv
[[nodiscard]] std::size_t shapeCornerCount(VolumeShape shape);

// The edges in the order of the mid-side nodes of the cards.
// Reference: volume::Shape::edges (TET_E, PYR_E, PEN_E, HEX_E)
[[nodiscard]] std::span<const ShapeEdge> shapeEdges(VolumeShape shape);

// The faces, their corners in a cyclic order:
//   tetrahedron (1,2,3) (0,2,3) (0,1,3) (0,1,2): face k opposite corner k;
//   pyramid (0,1,2,3) (0,1,4) (1,2,4) (2,3,4) (3,0,4);
//   pentahedron (0,1,2) (3,4,5) (0,1,4,3) (1,2,5,4) (2,0,3,5);
//   hexahedron (0,1,2,3) (4,5,6,7) (0,1,5,4) (1,2,6,5) (2,3,7,6) (3,0,4,7).
// Reference: volume::Shape::faces (TET_F, PYR_F, PEN_F, HEX_F)
[[nodiscard]] std::span<const ShapeFace> shapeFaces(VolumeShape shape);

// The reference centre: (−1/2, −1/2, −1/2), (0, 0, −1/2), (−1/3, −1/3, 0), (0, 0, 0).
// Reference: volume::Shape::centre
[[nodiscard]] SolidPoint shapeCentre(VolumeShape shape);

// The edge of the shape between the corners a and b (either order); empty when they have none.
[[nodiscard]] std::optional<std::size_t> shapeEdgeIndex(VolumeShape shape, std::size_t a, std::size_t b);

// The functions of an edge for the trace degree p: p − 1 (none for p < 2).
// Reference: volume::edge_count
[[nodiscard]] std::size_t edgeFunctionCount(std::size_t traceDegree);

// The functions inside a face of `cornerCount` corners for the face degree pf: (pf − 1)(pf − 2) / 2 for a triangle
// (none for pf < 3), (pf − 1)² for a quadrilateral (none for pf < 2).
// Reference: volume::face_count
[[nodiscard]] std::size_t faceFunctionCount(std::size_t faceDegree, std::size_t cornerCount);

// The dimension of the space of degree n of the shape.
// Reference: volume::space_dim
[[nodiscard]] std::size_t volumeSpaceDimension(VolumeShape shape, std::size_t n);

// The level (degree) of each function of volumeSpace(shape, n), in its order.
// Reference: volume::levels
[[nodiscard]] std::vector<std::size_t> volumeLevels(VolumeShape shape, std::size_t n);

// The rule of n points per direction on the reference element (positive weights, summing to its volume).
// Reference: volume::rule
[[nodiscard]] std::vector<VolumeRulePoint> volumeRule(VolumeShape shape, std::size_t n);

// The basis of the space of degree n of the shape at the reference point r (SolidSpaces.hpp).
// Reference: volume::space
template <Scalar T>
[[nodiscard]] std::vector<T> volumeSpace(VolumeShape shape, std::size_t n, const std::array<T, 3>& r)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return tetrahedronSpace(n, r);
    case VolumeShape::Pyramid:
        return pyramidSpace(n, r);
    case VolumeShape::Pentahedron:
        return pentahedronSpace(n, r);
    case VolumeShape::Hexahedron:
        break;
    }
    return hexahedronSpace(n, r);
}

// The function that vanishes on the faces of the shape, the factor of its bubbles (document 35, section 3.3):
//   tetrahedron   256 L0 L1 L2 L3,  L = (1 − Σ (1 + r_i)/2, (1 + r_0)/2, (1 + r_1)/2, (1 + r_2)/2);
//   pyramid       4 q² (1 − a²)(1 − b²)(1 + z) = 4 (q² − x²)(q² − y²)(1 + z) / q²,  q = (1 − z)/2, a = x/q, b = y/q;
//   pentahedron   27 L0 L1 L2 (1 − z²),  L = (−(x + y)/2, (1 + x)/2, (1 + y)/2);
//   hexahedron    (1 − x²)(1 − y²)(1 − z²).
// Each is 1 at the centre of its reference element (or near it). The factor of the pyramid holds q² and not q⁴: the
// interior functions of its space of degree 4 have q², and with q⁴ the bubbles do not hold them (document 35, 3.3).
// Zero at the apex of the pyramid (|q| below PYRAMID_APEX_TOLERANCE).
// Reference: volume::bubble
template <Scalar T>
[[nodiscard]] T volumeBubble(VolumeShape shape, const std::array<T, 3>& r)
{
    const T one = constantOf<T>(1.0);
    const T half = constantOf<T>(0.5);
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
    {
        const std::array<T, 4> l{one - (one + r[0]) * half - (one + r[1]) * half - (one + r[2]) * half,
                                 (one + r[0]) * half, (one + r[1]) * half, (one + r[2]) * half};
        return scale(l[0] * l[1] * l[2] * l[3], 256.0);
    }
    case VolumeShape::Pentahedron:
    {
        const std::array<T, 3> l{-(r[0] + r[1]) * half, (one + r[0]) * half, (one + r[1]) * half};
        return scale(l[0] * l[1] * l[2] * (one - r[2] * r[2]), 27.0);
    }
    case VolumeShape::Pyramid:
    {
        const T q = (one - r[2]) * half;
        if (std::fabs(valueOf(q)) < PYRAMID_APEX_TOLERANCE)
            return constantOf<T>(0.0);
        return scale((q * q - r[0] * r[0]) * (q * q - r[1] * r[1]) * (one + r[2]) / (q * q), 4.0);
    }
    case VolumeShape::Hexahedron:
        break;
    }
    return (one - r[0] * r[0]) * (one - r[1] * r[1]) * (one - r[2] * r[2]);
}

// The reference point of the element at the coordinates c of its face `face` (its corners, local to the shape), by
// the weights of the corners of the face (faceWeights: barycentric on a triangle, bilinear on a quadrilateral).
// Reference: volume::face_point
template <Scalar T>
[[nodiscard]] std::array<T, 3> facePoint(VolumeShape shape, std::span<const std::size_t> face,
                                         const std::array<T, 2>& c)
{
    const std::vector<T> weights = faceWeights(face.size(), c);
    const std::span<const SolidPoint> vertices = shapeVertices(shape);
    std::array<T, 3> r{constantOf<T>(0.0), constantOf<T>(0.0), constantOf<T>(0.0)};
    for (std::size_t k = 0; k < face.size(); ++k)
        for (std::size_t m = 0; m < 3; ++m)
            r[m] = r[m] + scale(weights[k], vertices[face[k]][m]);
    return r;
}

namespace detail
{

// The functions of the edges and of the inside of a triangular face (see the head of this file), appended to `out`;
// `l` the barycentric coordinates of its corners, `g` their numbers.
template <Scalar T>
void appendTriangleFaceFunctions(std::span<const base::EntityId> g, std::size_t p, std::size_t pf,
                                 const std::vector<T>& l, std::vector<T>& out)
{
    const T one = constantOf<T>(1.0);
    const std::size_t perEdge = edgeFunctionCount(p);
    for (std::size_t k = 0; k < 3; ++k)
    {
        // edge k from corner k to corner k + 1, written from its lower number i
        const std::size_t a = k;
        const std::size_t b = (k + 1) % 3;
        const std::size_t i = g[a] < g[b] ? a : b;
        const std::size_t j = g[a] < g[b] ? b : a;
        if (perEdge == 0)
            continue;
        const std::vector<T> s = scaledJacobi(p - 2, 0.0, l[j] - l[i], l[i] + l[j]);
        for (std::size_t m = 0; m < perEdge; ++m)
            out.push_back(l[i] * l[j] * s[m]);
    }
    if (pf < 3)
        return;
    std::array<std::size_t, 3> byNumber{0, 1, 2};
    std::sort(byNumber.begin(), byNumber.end(), [&](std::size_t x, std::size_t y) { return g[x] < g[y]; });
    const auto [i, j, k] = byNumber;
    const T x = l[j] - l[i];
    const T y = l[k] - l[i];
    const T faceBubble = l[i] * l[j] * l[k];
    for (std::size_t total = 0; total + 3 <= pf; ++total)
    {
        T xPower = one;
        for (std::size_t m = 0; m <= total; ++m)
        {
            T yPower = one;
            for (std::size_t n = 0; n < total - m; ++n)
                yPower = yPower * y;
            out.push_back(faceBubble * xPower * yPower);
            xPower = xPower * x;
        }
    }
}

// The functions of the edges and of the inside of a quadrilateral face at (s, t) ∈ [0, 1]² (see the head of this
// file), appended to `out`; `g` the numbers of its corners (0,0), (1,0), (1,1), (0,1).
template <Scalar T>
void appendQuadrilateralFaceFunctions(std::span<const base::EntityId> g, std::size_t p, std::size_t pf, const T& s,
                                      const T& t, std::vector<T>& out)
{
    const T one = constantOf<T>(1.0);
    const std::size_t perEdge = edgeFunctionCount(p);
    // edge k: its parameter from corner k towards corner k + 1, and the linear function across it (1 on the edge, 0 on
    // the opposite edge)
    const std::array<T, 4> along{s, t, one - s, one - t};
    const std::array<T, 4> across{one - t, s, t, one - s};
    for (std::size_t k = 0; k < 4; ++k)
    {
        if (perEdge == 0)
            continue;
        const bool lowerFirst = g[k] < g[(k + 1) % 4];
        const T se = lowerFirst ? along[k] : one - along[k];
        const std::vector<T> legendre = legendreAll(p - 2, scale(se, 2.0) - one);
        for (std::size_t m = 0; m < perEdge; ++m)
            out.push_back(se * (one - se) * legendre[m] * across[k]);
    }
    if (pf < 2)
        return;
    // the frame of the face: from its lowest number k0 towards the lower of its two neighbours (S), then the other (T)
    constexpr std::array<std::array<double, 2>, 4> CORNER{{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
    std::size_t k0 = 0;
    for (std::size_t k = 1; k < 4; ++k)
        if (g[k] < g[k0])
            k0 = k;
    const std::size_t next = (k0 + 1) % 4;
    const std::size_t previous = (k0 + 3) % 4;
    const std::size_t ks = g[next] < g[previous] ? next : previous;
    const std::size_t kt = g[next] < g[previous] ? previous : next;
    const std::array<double, 2> es{CORNER[ks][0] - CORNER[k0][0], CORNER[ks][1] - CORNER[k0][1]};
    const std::array<double, 2> et{CORNER[kt][0] - CORNER[k0][0], CORNER[kt][1] - CORNER[k0][1]};
    const T ds = s - constantOf<T>(CORNER[k0][0]);
    const T dt = t - constantOf<T>(CORNER[k0][1]);
    const T sc = scale(ds, es[0]) + scale(dt, es[1]);
    const T tc = scale(ds, et[0]) + scale(dt, et[1]);
    const std::vector<T> ps = legendreAll(pf - 2, scale(sc, 2.0) - one);
    const std::vector<T> pt = legendreAll(pf - 2, scale(tc, 2.0) - one);
    const T faceBubble = sc * (one - sc) * tc * (one - tc);
    for (std::size_t m = 0; m + 1 < pf; ++m)
        for (std::size_t n = 0; n + 1 < pf; ++n)
            out.push_back(faceBubble * ps[m] * pt[n]);
}

} // namespace detail

// The functions of a face at its coordinates c (faceRule: (L1, L2) on a triangle, (s, t) ∈ [0, 1]² on a
// quadrilateral), for the numbers g of its corners in the cyclic order the face is given in, the trace degree p and
// the face degree pf. In this order: one per corner (faceWeights), then edgeFunctionCount(p) per edge (edge k from
// corner k to corner k + 1), then faceFunctionCount(pf, n) inside (triangle: by total degree m + n, then m;
// quadrilateral: m along S outer, n along T inner, at m (pf − 1) + n).
// Reference: volume::face_functions
template <Scalar T>
[[nodiscard]] std::vector<T> faceFunctions(std::span<const base::EntityId> g, std::size_t p, std::size_t pf,
                                           const std::array<T, 2>& c)
{
    std::vector<T> out = faceWeights(g.size(), c);
    if (g.size() == 3)
    {
        const std::vector<T> l = out;
        detail::appendTriangleFaceFunctions(g, p, pf, l, out);
    }
    else
    {
        detail::appendQuadrilateralFaceFunctions(g, p, pf, c[0], c[1], out);
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
