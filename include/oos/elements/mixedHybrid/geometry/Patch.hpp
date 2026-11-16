#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/AnalyticSurfaces.hpp"
#include "oos/elements/mixedHybrid/geometry/RationalPatch.hpp"
#include "oos/elements/mixedHybrid/geometry/TrianglePatch.hpp"

#include <cstddef>
#include <variant>

namespace oos::elements::mixedHybrid
{

// The surface of one shell element: the rational patch of a quadrilateral or the rational triangle read from the
// mesh, or a cylinder or sphere patch given explicitly (validation). It is a Surface (Surface.hpp) whatever its kind.
//
// Invariants: a triangle has 3 corners (the reference triangle), every other kind 4 (the reference square).
// Thread-safety: a plain value type; evaluate() is pure.
//
// Reference: src/model.rs::Patch.
struct Patch
{
    std::variant<RationalPatch, CylinderPatch, SpherePatch, TrianglePatch> shape;

    // Implicit on purpose: a patch of any kind is a Patch.
    Patch() = default;
    Patch(const RationalPatch& p) : shape(p) {}
    Patch(const CylinderPatch& p) : shape(p) {}
    Patch(const SpherePatch& p) : shape(p) {}
    Patch(const TrianglePatch& p) : shape(p) {}

    [[nodiscard]] std::size_t cornerCount() const { return std::holds_alternative<TrianglePatch>(shape) ? 3 : 4; }

    template <Scalar T>
    [[nodiscard]] Vec3Of<T> evaluate(const T& xi, const T& eta) const
    {
        return std::visit([&](const auto& s) { return s.evaluate(xi, eta); }, shape);
    }

    // The patch as the given kind, or null.
    [[nodiscard]] const RationalPatch* rational() const { return std::get_if<RationalPatch>(&shape); }
    [[nodiscard]] const TrianglePatch* triangle() const { return std::get_if<TrianglePatch>(&shape); }
};

} // namespace oos::elements::mixedHybrid
