#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// The surface of a triangle read from the mesh, on three grids and three edge conics (Surface.hpp, "edge conics"):
// the rational quadratic Bézier triangle of the grids (weight 1) and of the control points and weights of the edges,
// plus an interior term,
//   X = [Σ_k L_k² X_k + 2 Σ_k w_k L_k L_k+1 P_k] / [Σ_k L_k² + 2 Σ_k w_k L_k L_k+1]
//       + L_0 L_1 L_2 (L_0 c_0 + L_1 c_1 + L_2 c_2),
// L the barycentric coordinates (ReferenceShapes.hpp). Its edges are the conics of the edges, with the same parameter
// as those of RationalPatch (the coordinate of the end grid), and its tangent plane at a grid is that of the two
// edges there. The three vectors c are the only freedom of the interior (fitInterior, fitInteriorRational); they
// vanish on a flat element, which is then the plane triangle on its edge curves. Document 26 of the reference, 3.1;
// the rational Bézier triangle is that of G. Farin, Curves and Surfaces for CAGD, 5th ed., Morgan Kaufmann 2002,
// chapter 17.
//
// Invariants: corners counter-clockwise in the reference triangle, edge k from grid k to grid k + 1. Thread-safety:
// a plain value type; evaluate() is pure, the fits mutate their patch only.
//
// Reference: src/surface.rs::TriPatch.
struct TrianglePatch
{
    std::array<Vector3, 3> corners{};
    std::array<Vector3, 3> bulges{};
    // The control point and the weight of each edge conic.
    std::array<Vector3, 3> controlPoints{};
    std::array<double, 3> weights{};
    // The interior vectors c_k.
    std::array<Vector3, 3> interior{};
    // The interior term inside the rational form, X = (numerator + L0 L1 L2 Σ L_k c_k) / denominator
    // (fitInteriorRational, the shell of document 34 of the reference): the surface is then a rational map of degree
    // 4 over the quadratic denominator. false: added outside (fitInterior).
    bool interiorInsideRationalForm = false;

    // The patch of the three grids and edge bulges. `edgeWeights`: the weights of the edge conics that are not those
    // of their control polygons (NaN: the circular arc).
    // Reference: surface.rs::TriPatch::from_edges.
    [[nodiscard]] static TrianglePatch fromEdges(const std::array<Vector3, 3>& corners,
                                                 const std::array<Vector3, 3>& bulges,
                                                 const std::array<double, 3>& edgeWeights);

    // The interior that makes the patch as smooth as its data allow, as RationalPatch::fitCentre does: the data hold
    // the normals at the three grids, and the normal of the patch along each edge is brought to a target. On the edge
    // from grid i to grid j, of parameter τ, the target is the direction
    //   (1 − τ)² n_i + τ (1 − τ) (n_i + n_j) / w + τ² n_j,   w = cos(half the angle of n_i, n_j),
    // less its part along the edge: the normal turns from n_i to n_j about a fixed axis with the parameter of a
    // circular arc. That is the normal of a sphere along an arc of a great circle and that of a cylinder along any
    // plane section: on both the rational triangle is left as it is. The inward derivative of the patch on the edge
    // is that of the rational triangle plus L_i L_j (L_i c_i + L_j c_j) times a constant: with the vectors c along
    // the normal of the patch at its centre, three amplitudes enter a linear least-squares problem, always well
    // posed. Nothing moves on a flat element.
    // Reference: surface.rs::TriPatch::fit_interior.
    void fitInterior();

    // The interior inside the rational form (interiorInsideRationalForm): the criterion of fitInterior, the interior
    // term divided by the denominator of the edge conic.
    // Reference: surface.rs::TriPatch::fit_interior_rational.
    void fitInteriorRational();

    [[nodiscard]] static constexpr std::size_t cornerCount() { return 3; }

    // Reference: surface.rs::TriPatch (Surface::eval).
    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        const std::array<T, 3> l = barycentric(xi, eta);
        Vec3Of<T> numerator{constantOf<T>(0.0), constantOf<T>(0.0), constantOf<T>(0.0)};
        T denominator = constantOf<T>(0.0);
        for (std::size_t k = 0; k < 3; ++k)
        {
            const T b = l[k] * l[k];
            const T e = scale(l[k] * l[(k + 1) % 3], 2.0 * weights[k]);
            denominator = denominator + b + e;
            for (std::size_t c = 0; c < 3; ++c)
                numerator[c] = numerator[c] + scale(b, corners[k][c]) + scale(e, controlPoints[k][c]);
        }
        const T bubble = l[0] * l[1] * l[2];
        Vec3Of<T> out{};
        for (std::size_t c = 0; c < 3; ++c)
        {
            const T term =
                bubble * (scale(l[0], interior[0][c]) + scale(l[1], interior[1][c]) + scale(l[2], interior[2][c]));
            out[c] =
                interiorInsideRationalForm ? (numerator[c] + term) / denominator : numerator[c] / denominator + term;
        }
        return out;
    }

private:
    // The fit of both forms: `insideRationalForm` divides the interior term by the denominator of the edge conic.
    void fitInteriorTerms(bool insideRationalForm);
};

} // namespace oos::elements::mixedHybrid
