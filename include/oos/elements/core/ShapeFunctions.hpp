#pragma once

#include "oos/base/math/MatN.hpp"

#include <array>

namespace oos::elements
{

// The parent coordinates (ξ_i, η_i) of the corners of a four-node quadrilateral, in the order G1 to G4 of the card:
// (−1, −1), (1, −1), (1, 1), (−1, 1).
inline constexpr std::array<double, 4> QUAD_CORNER_XI{-1.0, 1.0, 1.0, -1.0};
inline constexpr std::array<double, 4> QUAD_CORNER_ETA{-1.0, -1.0, 1.0, 1.0};

// The bilinear shape functions N_i = ¼(1 + ξ_i ξ)(1 + η_i η) of a four-node quadrilateral at a point (ξ, η) and their
// derivatives with respect to ξ and η.
struct BilinearShape
{
    std::array<double, 4> n{};
    std::array<double, 4> dXi{};
    std::array<double, 4> dEta{};
};
[[nodiscard]] BilinearShape bilinearShapeAt(double xi, double eta);

// The Jacobian of the map (ξ, η) → (x, y) of a plane quadrilateral at a point, with the physical derivatives of the
// shape functions there. Convention (used by every shell formula):
// J = [[∂x/∂ξ, ∂y/∂ξ], [∂x/∂η, ∂y/∂η]], so that [∂/∂x; ∂/∂y] = J⁻¹·[∂/∂ξ; ∂/∂η].
struct PlaneJacobian
{
    base::MatN<2> j;
    base::MatN<2> inverse;
    double determinant = 0.0;
    std::array<double, 4> dX{}; // ∂N_i/∂x
    std::array<double, 4> dY{}; // ∂N_i/∂y
};
// Precondition: the Jacobian is not singular at the point (the element's geometry check refuses such a shape).
[[nodiscard]] PlaneJacobian planeJacobianAt(const BilinearShape& shape, const std::array<double, 4>& x,
                                            const std::array<double, 4>& y);

} // namespace oos::elements
