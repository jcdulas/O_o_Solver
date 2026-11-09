#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace oos::elements
{

// The most DOFs of a solid element: three translations at each of its 20 nodes.
inline constexpr std::size_t MAX_SOLID_DOFS = 3 * MAX_SOLID_NODES;

// A dense square matrix over the DOFs of one solid (u1, v1, w1, u2, ...), at most 60 × 60, row-major. A value type of
// fixed capacity (29 kB): the facets hold one per call, never one per element on the stack of a loop.
struct SolidMatrix
{
    std::size_t size = 0;
    std::array<double, MAX_SOLID_DOFS * MAX_SOLID_DOFS> values{};

    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const { return values[i * size + j]; }
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) { return values[i * size + j]; }
    // Sets the size and every term to 0.
    void reset(std::size_t newSize);
};

// A linear map from the DOFs of one solid to a Voigt 6-vector (a strain or a stress at one point): 6 rows of `columns`
// terms, row-major.
struct SolidOperator
{
    std::size_t columns = 0;
    std::array<double, 6 * MAX_SOLID_DOFS> values{};

    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const { return values[i * columns + j]; }
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) { return values[i * columns + j]; }
    void reset(std::size_t newColumns);
    // The Voigt vector of the DOF values u (u.size() is `columns`).
    [[nodiscard]] Voigt6 apply(std::span<const double> u) const;
};

// The Jacobian of the map from the parent element to the element at a parent point, and the gradients of the shape
// functions there. Convention: J(a, b) = ∂x_b/∂ξ_a (row a the derivatives along parent coordinate a), so that
// ∇_x N = J⁻¹·∇_ξ N.
struct ShapeGradients
{
    SolidShapeValues shape;
    base::Mat3 jacobian;
    double determinant = 0.0;
    std::array<base::Vec3, MAX_SOLID_NODES> gradients{}; // ∇_x N_i
};

// The Jacobian at a parent point of an element whose nodes are at `positions` (count of the shape). The gradients are
// left at 0 where the determinant is 0 (a collapsed corner): the caller checks the determinant first.
[[nodiscard]] ShapeGradients shapeGradientsAt(SolidShape shape, std::span<const base::Vec3> positions,
                                              ParentPoint point);

// The strain-displacement matrix B of the gradients: ε = B·u, engineering strains (εx, εy, εz, γxy, γyz, γzx) with
// γxy = ∂u/∂y + ∂v/∂x (Zienkiewicz & Taylor vol. 1 §6.2).
void fillStrainOperator(const ShapeGradients& gradients, SolidOperator& b);

// The Voigt engineering strain of the symmetric part of a displacement gradient G(i, j) = ∂u_i/∂x_j.
[[nodiscard]] Voigt6 voigtStrainOf(const base::Mat3& gradient);

// The 6 × 6 inverse of a positive definite constitutive matrix (the compliance S = D⁻¹).
[[nodiscard]] Mat6 complianceOf(const Mat6& elasticity);

} // namespace oos::elements
