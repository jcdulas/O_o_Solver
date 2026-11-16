#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// Surfaces given by formulas, separate from the reading of the mesh: patches of a cylinder and of a sphere (exact
// geometry, for validation and for elements whose surface is given), and the parallel surface X + d n of an offset
// element.
//
// Invariants: the analytic patches are quadrilaterals (reference square); their angle and axial coordinates are
// bilinear in (ξ, η). Thread-safety: plain value types; evaluate() is pure.

namespace detail
{
// The bilinear interpolation of four corner values c_k at (ξ, η): Σ_k N_k c_k, N_k = (1 ± ξ)(1 ± η)/4.
// Reference: src/surface.rs::bilin.
template <Scalar T>
[[nodiscard]] T bilinearInCorners(const std::array<double, 4>& c, const T& xi, const T& eta)
{
    const T one = constantOf<T>(1.0);
    const T um = one - xi;
    const T up = one + xi;
    const T vm = one - eta;
    const T vp = one + eta;
    return scale(um * vm, 0.25 * c[0]) + scale(up * vm, 0.25 * c[1]) + scale(up * vp, 0.25 * c[2]) +
           scale(um * vp, 0.25 * c[3]);
}
} // namespace detail

// A patch of the cylinder X = centre + z axis + R (cos θ e1 + sin θ e2), θ and z bilinear in (ξ, η).
// Reference: src/surface.rs::CylinderPatch.
struct CylinderPatch
{
    Vector3 centre{};
    Vector3 axis{};
    Vector3 e1{};
    Vector3 e2{};
    double radius = 0.0;
    // θ and z at the four corners.
    std::array<double, 4> angle{};
    std::array<double, 4> z{};

    [[nodiscard]] static constexpr std::size_t cornerCount() { return 4; }

    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        const T height = detail::bilinearInCorners(z, xi, eta);
        const T theta = detail::bilinearInCorners(angle, xi, eta);
        const T c = cos(theta);
        const T s = sin(theta);
        Vec3Of<T> out{};
        for (std::size_t i = 0; i < 3; ++i)
            out[i] =
                constantOf<T>(centre[i]) + scale(height, axis[i]) + scale(c, radius * e1[i]) + scale(s, radius * e2[i]);
        return out;
    }
};

// A patch of the sphere X = centre + R (sin φ cos θ e1 + sin φ sin θ e2 + cos φ axis), the colatitude φ and the
// longitude θ bilinear in (ξ, η).
// Reference: src/surface.rs::SpherePatch.
struct SpherePatch
{
    Vector3 centre{};
    Vector3 axis{};
    Vector3 e1{};
    Vector3 e2{};
    double radius = 0.0;
    // φ and θ at the four corners.
    std::array<double, 4> phi{};
    std::array<double, 4> theta{};

    [[nodiscard]] static constexpr std::size_t cornerCount() { return 4; }

    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        const T p = detail::bilinearInCorners(phi, xi, eta);
        const T t = detail::bilinearInCorners(theta, xi, eta);
        const T sp = sin(p);
        const T cp = cos(p);
        const T st = sin(t);
        const T ct = cos(t);
        Vec3Of<T> out{};
        for (std::size_t i = 0; i < 3; ++i)
            out[i] = constantOf<T>(centre[i]) + scale(sp * ct, radius * e1[i]) + scale(sp * st, radius * e2[i]) +
                     scale(cp, radius * axis[i]);
        return out;
    }
};

// The parallel surface X + d n of a base surface: the reference surface of an offset element, valid on any base.
// Reference: src/surface.rs::Offset.
template <Surface S>
struct OffsetSurface
{
    S base{};
    double distance = 0.0;

    [[nodiscard]] std::size_t cornerCount() const { return base.cornerCount(); }

    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        if (distance == 0.0)
            return base.evaluate(xi, eta);
        const SurfacePointFrame<T> f = frame(base, xi, eta);
        return add3(f.position, scl3(f.normal, constantOf<T>(distance)));
    }
};

} // namespace oos::elements::mixedHybrid
