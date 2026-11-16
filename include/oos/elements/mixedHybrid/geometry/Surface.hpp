#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"

#include <array>
#include <concepts>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// The reference geometry of a shell element of the mixed hybrid formulation. A surface is any smooth map
// (ξ, η) → X of the reference domain (the square or the triangle of ReferenceShapes.hpp), written once as a template
// on its scalar type; its tangents, its unit normal and their derivatives follow by nested forward differentiation
// (DualScalar.hpp), exactly. Nothing classifies a surface: the kernels only ask for point values.
//
// Invariants: a surface is a small value type; evaluate() is pure. Thread-safety: the types and functions of this
// header are plain values and pure functions.
//
// Reference: src/surface.rs (the Surface trait and the point quantities).

// A point of space or a vector.
using Vector3 = std::array<double, 3>;

// What a surface provides: its corner count (4, the reference square, or 3, the reference triangle) and its point
//   template <Scalar T> Vec3Of<T> evaluate(const T& xi, const T& eta) const;
// Reference: surface.rs::Surface.
template <class S>
concept Surface = std::copyable<S> && requires(const S& surface, double x, Dual<double> d) {
    { surface.cornerCount() } -> std::convertible_to<std::size_t>;
    { surface.evaluate(x, x) } -> std::same_as<Vec3Of<double>>;
    { surface.evaluate(d, d) } -> std::same_as<Vec3Of<Dual<double>>>;
};

// The position and the natural tangents a_0 = ∂X/∂ξ, a_1 = ∂X/∂η.
template <class T>
struct SurfaceTangents
{
    Vec3Of<T> position{};
    std::array<Vec3Of<T>, 2> tangents{};
};

// Reference: surface.rs::tangents.
template <Surface S, Scalar T>
[[nodiscard]] SurfaceTangents<T> tangents(const S& surface, const T& xi, const T& eta)
{
    const Vec3Of<Dual<T>> p = surface.evaluate(Dual<T>::variable(xi, 0), Dual<T>::variable(eta, 1));
    SurfaceTangents<T> out;
    for (std::size_t i = 0; i < 3; ++i)
    {
        out.position[i] = p[i].value;
        out.tangents[0][i] = p[i].derivatives[0];
        out.tangents[1][i] = p[i].derivatives[1];
    }
    return out;
}

// The position, the tangents, the unit normal n = a_0 × a_1 / |a_0 × a_1| and the area element J = |a_0 × a_1| at a
// (possibly dual) point.
// Reference: surface.rs::Frame.
template <class T>
struct SurfacePointFrame
{
    Vec3Of<T> position{};
    std::array<Vec3Of<T>, 2> tangents{};
    Vec3Of<T> normal{};
    T jacobian{};
};

// Reference: surface.rs::frame.
template <Surface S, Scalar T>
[[nodiscard]] SurfacePointFrame<T> frame(const S& surface, const T& xi, const T& eta)
{
    const SurfaceTangents<T> t = tangents(surface, xi, eta);
    const Vec3Of<T> c = cross3(t.tangents[0], t.tangents[1]);
    const T jacobian = norm3(c);
    return {t.position, t.tangents, scl3(c, constantOf<T>(1.0) / jacobian), jacobian};
}

// Everything the kernels need at a real point of a surface.
// Reference: surface.rs::PointGeom.
struct SurfacePointGeometry
{
    Vector3 position{};
    // tangents[α] = ∂X/∂ξ_α.
    std::array<Vector3, 2> tangents{};
    // tangentDerivatives[α][β] = ∂a_α/∂ξ_β.
    std::array<std::array<Vector3, 2>, 2> tangentDerivatives{};
    Vector3 normal{};
    // normalDerivatives[α] = ∂n/∂ξ_α.
    std::array<Vector3, 2> normalDerivatives{};
    double jacobian = 0.0;
};

// Reference: surface.rs::point_geom.
template <Surface S>
[[nodiscard]] SurfacePointGeometry pointGeometry(const S& surface, double xi, double eta)
{
    const SurfacePointFrame<Dual<double>> f =
        frame(surface, Dual<double>::variable(xi, 0), Dual<double>::variable(eta, 1));
    const auto value = [](const Vec3Of<Dual<double>>& w) { return Vector3{w[0].value, w[1].value, w[2].value}; };
    const auto derivative = [](const Vec3Of<Dual<double>>& w, std::size_t b)
    { return Vector3{w[0].derivatives[b], w[1].derivatives[b], w[2].derivatives[b]}; };
    SurfacePointGeometry g;
    g.position = value(f.position);
    g.tangents = {value(f.tangents[0]), value(f.tangents[1])};
    g.tangentDerivatives = {{{derivative(f.tangents[0], 0), derivative(f.tangents[0], 1)},
                             {derivative(f.tangents[1], 0), derivative(f.tangents[1], 1)}}};
    g.normal = value(f.normal);
    g.normalDerivatives = {derivative(f.normal, 0), derivative(f.normal, 1)};
    g.jacobian = f.jacobian.value;
    return g;
}

// The orthonormal tangent frame (e1, e2) at a point of unit normal n: e1 the projection of the reference direction on
// the tangent plane, e2 = n × e1.
// Reference: surface.rs::tangent_frame.
[[nodiscard]] std::array<Vector3, 2> tangentFrame(const Vector3& normal, const Vector3& reference);

// The physical quantities of a point geometry in a tangent frame.
// Reference: surface.rs::PhysGeom.
struct PhysicalGeometry
{
    // The tangent frame (e1, e2).
    std::array<Vector3, 2> frame{};
    // components[I][α] = e_I · a_α.
    std::array<std::array<double, 2>, 2> components{};
    // inverse[α][I] = ∂ξ_α/∂s_I, the inverse of `components`.
    std::array<std::array<double, 2>, 2> inverse{};
    // normalGradient[J] = ∂n/∂s_J.
    std::array<Vector3, 2> normalGradient{};
};

// The quantities in the frame of the reference direction `reference` (tangentFrame).
// Reference: surface.rs::phys_geom.
[[nodiscard]] PhysicalGeometry physicalGeometry(const SurfacePointGeometry& g, const Vector3& reference);

// The quantities in the frame of the two reference axes x and y of an element (tangentFrameXy).
// Reference: surface.rs::phys_geom_xy.
[[nodiscard]] PhysicalGeometry physicalGeometryXy(const SurfacePointGeometry& g, const Vector3& x, const Vector3& y);

// The quantities in a given tangent frame e = axes: components c_Iα = e_I · a_α, their inverse, and ∂n/∂s_J =
// Σ_α ∂n/∂ξ_α ∂ξ_α/∂s_J.
// Reference: surface.rs::phys_geom_in.
[[nodiscard]] PhysicalGeometry physicalGeometryInFrame(const SurfacePointGeometry& g,
                                                       const std::array<Vector3, 2>& axes);

// The position and the velocity dX/dτ on edge k of a surface at τ in [0, 1] (from corner k to corner k + 1):
// dX/dτ = a_0 d_0 + a_1 d_1, d the edge of the reference domain.
template <class T>
struct EdgePoint
{
    Vec3Of<T> position{};
    Vec3Of<T> velocity{};
};

// Reference: surface.rs::edge_eval.
template <Surface S, Scalar T>
[[nodiscard]] EdgePoint<T> edgeEvaluation(const S& surface, std::size_t k, const T& tau)
{
    const std::size_t cornerCount = surface.cornerCount();
    const std::array<T, 2> uv = referenceEdgePoint(cornerCount, k, tau);
    const SurfaceTangents<T> t = tangents(surface, uv[0], uv[1]);
    const ReferencePoint p = referenceCorner(cornerCount, k);
    const ReferencePoint q = referenceCorner(cornerCount, k + 1);
    const std::array<double, 2> d{q[0] - p[0], q[1] - p[1]};
    return {t.position, add3(scl3(t.tangents[0], constantOf<T>(d[0])), scl3(t.tangents[1], constantOf<T>(d[1])))};
}

// ---- edge conics ------------------------------------------------------------------------------------------------
//
// The curve of an edge read from the mesh is the conic of its two grids X_a, X_b, its control point P and its weight
// w (document 25 of the reference, 3.1):
//   X(τ) = [(1 − τ)² X_a + 2 w τ (1 − τ) P + τ² X_b] / [(1 − τ)² + 2 w τ (1 − τ) + τ²],
// P = (X_a + X_b)/2 + 2 b, b the bulge of the edge (half the offset of the control point from the middle of the
// chord). With the weight of the control polygon, the sine of half its angle at P, the conic is the circular arc
// whose end slopes are those of the polygon (L. Piegl, W. Tiller, The NURBS Book, 2nd ed., Springer 1997, 7.3).

// The weight of the control polygon of an edge of chord X_b − X_a and bulge b: with a_1 = chord/2 + 2b and
// a_2 = chord/2 − 2b, cos γ = −a_1·a_2 / (|a_1| |a_2|) and w = √(max(0, (1 − cos γ)/2)): 1 on a straight edge.
[[nodiscard]] double controlPolygonWeight(const Vector3& chord, const Vector3& bulge);

} // namespace oos::elements::mixedHybrid
