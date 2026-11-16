#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The projections that carry a curve on the functions of an edge of the solids, and a surface on the functions of a
// face (document 35 of the reference, section 3.4; document 34, section 8.2). The geometry of the solids (SolidPart)
// and, later, the ties with the shells use them: the coefficients of an edge are the L2 projection, on its functions
// s (1 − s) P_m(2s − 1), of what the curve adds to the chord between its two grid points; the coefficients of a face
// are the L2 projection, on its interior functions, of what the surface adds to its grid points and its edges.
//
// Conventions: an edge is parametrised by s ∈ [0, 1] from its lower grid point; a face by the coordinates of faceRule
// for the numbers of its grid points in the cyclic order it is given in (faceFunctions of VolumeShape.hpp).
//
// Thread-safety: plain values and pure functions.

// The functions of an edge at the parameter s from its lower grid point, s (1 − s) P_m(2s − 1), m = 0 … p − 2, with
// P_m the Legendre polynomial (the homogeneous Jacobi polynomial of weight 0 at q = 1); none for p < 2.
// Reference: src/tdnns/solid.rs::edge_functions
[[nodiscard]] std::vector<double> edgeFunctions(std::size_t traceDegree, double s);

// The projection on the functions of an edge of what a field adds to the linear interpolation of its two end values:
// coefficient m is Σ_q a(m, q) f(s_q) + a0[m] f(0) + a1[m] f(1), with the 12-point Gauss-Legendre rule on [0, 1]:
//   a = G⁻¹ (w_q b_n(s_q)),   G_mn = Σ_q w_q b_m(s_q) b_n(s_q),
//   a0[m] = −Σ_q a(m, q) (1 − s_q),   a1[m] = −Σ_q a(m, q) s_q   (the linear part removed).
// Reference: src/tdnns/solid.rs::EdgeProjection, edge_projection
struct EdgeProjection
{
    std::vector<double> parameters; // s_q
    Matrix weights;                 // a: edgeFunctionCount(p) × points
    std::vector<double> lowerWeights;
    std::vector<double> upperWeights;

    [[nodiscard]] static EdgeProjection of(std::size_t traceDegree);

    // The coefficients of the curve whose points at `parameters` are `values` (one per parameter), from the grid
    // point at `lower` (s = 0) to the one at `upper` (s = 1).
    [[nodiscard]] std::vector<SolidPoint> project(std::span<const SolidPoint> values, const SolidPoint& lower,
                                                  const SolidPoint& upper) const;
};

// The projection on the interior functions of a face of what a field adds to its grid points and to the functions of
// its edges: the points of the face rule of 10 points per direction (faceRule), the weights b(m, q) of the value at
// point q in coefficient m,
//   b = G⁻¹ (w_q φ_n(c_q)),   G_mn = Σ_q w_q φ_m(c_q) φ_n(c_q)   (φ the interior functions of the face),
// and the values of all the functions of the face at each point (faceFunctions: the grid points, then the edges, then
// the interior).
// Reference: src/tdnns/solid.rs::FaceProjection, face_projection
struct FaceProjection
{
    std::vector<std::array<double, 2>> points;
    Matrix weights; // b: faceFunctionCount × points
    std::vector<std::vector<double>> functionValues;
    std::size_t perEdge = 0;       // ec
    std::size_t interiorCount = 0; // fc

    // `grids`: the numbers of the grid points of the face in its cyclic order.
    [[nodiscard]] static FaceProjection of(std::span<const base::EntityId> grids, std::size_t traceDegree,
                                           std::size_t faceDegree);

    // The coefficients of the surface whose points at `points` are `values`: what remains of each value once the
    // grid points (`gridPositions`, the cyclic order of the face) and the edges (`edgeCoefficients[k]`, edge k from
    // grid point k to grid point k + 1) are taken out, projected on the interior functions.
    [[nodiscard]] std::vector<SolidPoint> project(std::span<const SolidPoint> values,
                                                  std::span<const SolidPoint> gridPositions,
                                                  std::span<const std::vector<SolidPoint>> edgeCoefficients) const;
};

} // namespace oos::elements::mixedHybrid
