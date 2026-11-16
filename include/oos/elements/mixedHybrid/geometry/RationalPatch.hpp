#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// The surfaces of a quadrilateral read from the mesh: four grids and one conic per edge (Surface.hpp, "edge conics").
//
// RationalPatch is the surface of every quadrilateral whose geometry is read from the mesh: the rational biquadratic
// Bézier patch (one patch of a NURBS surface; L. Piegl, W. Tiller, The NURBS Book, 2nd ed., Springer 1997, chapter 4)
//   X(ξ, η) = Σ_ij w_ij B_i(s) B_j(t) P_ij / Σ_ij w_ij B_i(s) B_j(t),   s = (1 + ξ)/2,  t = (1 + η)/2,
// B_i the quadratic Bernstein polynomials. Its corners are the grids (weight 1), its edge control points and weights
// those of the edge conics, and its centre control point the only freedom of its interior (fitCentre). Every
// iso-parametric line is a conic: the patch cannot oscillate. On a flat element with straight edges it is the
// bilinear patch. Document 25 of the reference, 3.1.
//
// CoonsPatch is the Coons combination of the same four edge conics, kept for single-element studies (the elements take
// RationalPatch).
//
// Invariants: corners counter-clockwise in the reference square, edge k from grid k to grid k + 1. Thread-safety:
// plain value types; evaluate() is pure, fitCentre() mutates its patch only.

// Reference: src/surface.rs::QuadPatch.
struct CoonsPatch
{
    std::array<Vector3, 4> corners{};
    // The bulge of each edge conic (half the offset of its control point from the middle of its chord).
    std::array<Vector3, 4> bulges{};

    // The bilinear patch through the four grids (straight edges).
    [[nodiscard]] static CoonsPatch bilinear(const std::array<Vector3, 4>& corners) { return {corners, {}}; }

    [[nodiscard]] static constexpr std::size_t cornerCount() { return 4; }

    // X = Σ_k W_k(ξ, η) C_k(τ_k) − Σ_k N_k(ξ, η) X_k: the edge curves C_k blended linearly across the patch, less the
    // bilinear interpolation of the corners (the Coons patch; W. J. Gordon, Blending-function methods of bivariate
    // and multivariate interpolation, SIAM J. Numer. Anal. 8 (1971) 158-177). Each C_k is the conic of the edge with
    // the weight of its control polygon, written as X_k + g(τ) chord + h(τ) 2 bulge.
    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        const T one = constantOf<T>(1.0);
        const T quarter = constantOf<T>(0.25);
        const T half = constantOf<T>(0.5);
        const T um = one - xi;
        const T up = one + xi;
        const T vm = one - eta;
        const T vp = one + eta;
        const std::array<T, 4> bilinearShape{um * vm * quarter, up * vm * quarter, up * vp * quarter,
                                             um * vp * quarter};
        // the parameter of each edge at the point, and the blending weight of that edge
        const std::array<T, 4> tau{up * half, vp * half, um * half, vm * half};
        const std::array<T, 4> blend{vm * half, up * half, vp * half, um * half};
        Vec3Of<T> out{constantOf<T>(0.0), constantOf<T>(0.0), constantOf<T>(0.0)};
        for (std::size_t k = 0; k < 4; ++k)
        {
            const Vector3& p0 = corners[k];
            const Vector3 chord = sub3(corners[(k + 1) % 4], p0);
            const Vector3 control = scl3(bulges[k], 2.0);
            const double w = controlPolygonWeight(chord, bulges[k]);
            const T a = tau[k] * (one - tau[k]);
            const T denominator = one - scale(a, 2.0 * (1.0 - w));
            const T g = (scale(a, w) + tau[k] * tau[k]) / denominator;
            const T h = scale(a, 2.0 * w) / denominator;
            for (std::size_t i = 0; i < 3; ++i)
                out[i] = out[i] + blend[k] * (constantOf<T>(p0[i]) + scale(g, chord[i]) + scale(h, control[i])) -
                         scale(bilinearShape[k], p0[i]);
        }
        return out;
    }
};

// Reference: src/surface.rs::RationalPatch.
struct RationalPatch
{
    // controlPoints[i][j]: i along ξ, j along η.
    std::array<std::array<Vector3, 3>, 3> controlPoints{};
    std::array<std::array<double, 3>, 3> weights{};
    std::array<Vector3, 4> corners{};
    std::array<Vector3, 4> bulges{};

    // The patch of the four grids and edge bulges, every edge conic with the weight of its control polygon (the
    // circular arc), and the centre of the rule of fromWeightedEdges.
    // Reference: surface.rs::RationalPatch::from_edges.
    [[nodiscard]] static RationalPatch fromEdges(const std::array<Vector3, 4>& corners,
                                                 const std::array<Vector3, 4>& bulges);

    // The same patch with the weights of the edge conics that are not those of their control polygons (NaN: the
    // circular arc), the edges shared with triangles (Topology::weights). The centre control point is the Coons
    // combination of the eight others,
    //   P_11 = (P_10 + P_12)/2 + (P_01 + P_21)/2 − (X_0 + X_1 + X_2 + X_3)/4,
    // and its weight the product of the mean weights of the two pairs of opposite edges,
    //   w_11 = (w_10 + w_12)/4 · (w_01 + w_21),
    // which is exact on a cylinder.
    // Reference: surface.rs::RationalPatch::from_weighted_edges.
    [[nodiscard]] static RationalPatch fromWeightedEdges(const std::array<Vector3, 4>& corners,
                                                         const std::array<Vector3, 4>& bulges,
                                                         const std::array<double, 4>& edgeWeights);

    // Moves the centre control point so that the patch is as smooth as its data allow (document 25 of the reference,
    // 3.2, rule 6). The data hold more than the edges: the normals at the four grids. Along each edge the target
    // normal is the one carried from one grid to the other without twisting about the edge (the edge is planar: a
    // constant angle with its plane), plus the uniform twist that reaches the normal of the other grid. On an edge
    // the inward derivative of the patch is affine in the centre control point, so its three coordinates enter a
    // linear least-squares problem. Its move along the normal of the patch at its centre is fixed by it; its move in
    // the surface is weighed against the misfit of the normals (see the source). Nothing moves on a flat element.
    // Reference: surface.rs::RationalPatch::fit_centre.
    void fitCentre();

    [[nodiscard]] static constexpr std::size_t cornerCount() { return 4; }

    // Reference: surface.rs::RationalPatch (Surface::eval).
    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        const T one = constantOf<T>(1.0);
        const T half = constantOf<T>(0.5);
        const T s = (one + xi) * half;
        const T t = (one + eta) * half;
        const std::array<T, 3> bs{(one - s) * (one - s), scale(s * (one - s), 2.0), s * s};
        const std::array<T, 3> bt{(one - t) * (one - t), scale(t * (one - t), 2.0), t * t};
        Vec3Of<T> numerator{constantOf<T>(0.0), constantOf<T>(0.0), constantOf<T>(0.0)};
        T denominator = constantOf<T>(0.0);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
            {
                const T b = scale(bs[i] * bt[j], weights[i][j]);
                denominator = denominator + b;
                for (std::size_t c = 0; c < 3; ++c)
                    numerator[c] = numerator[c] + scale(b, controlPoints[i][j][c]);
            }
        return {numerator[0] / denominator, numerator[1] / denominator, numerator[2] / denominator};
    }

    // The homogeneous numerator N and denominator D at the Bézier parameters (s, t), with their derivatives in s and
    // t.
    struct Homogeneous
    {
        Vector3 numerator{};
        double denominator = 0.0;
        std::array<Vector3, 2> numeratorDerivatives{};
        std::array<double, 2> denominatorDerivatives{};
    };
    // Reference: surface.rs::RationalPatch::homogeneous.
    [[nodiscard]] Homogeneous homogeneous(double s, double t) const;
};

} // namespace oos::elements::mixedHybrid
