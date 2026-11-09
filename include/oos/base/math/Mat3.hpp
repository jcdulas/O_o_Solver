#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"

namespace oos::base
{

// A 3 × 3 matrix: rotations between frames (the rows of a rotation are the axes of the target frame written in the
// source frame, as O_o_Reader's `displacementRotations` gives them), inertia tensors, material matrices, Jacobians.
using Mat3 = MatN<3>;

[[nodiscard]] constexpr Vec3 operator*(const Mat3& m, Vec3 v)
{
    return {m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2], m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
            m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]};
}

// The matrix whose rows are the three given vectors.
[[nodiscard]] constexpr Mat3 matrixFromRows(Vec3 r0, Vec3 r1, Vec3 r2)
{
    Mat3 m;
    for (std::size_t j = 0; j < 3; ++j)
    {
        m(0, j) = r0[j];
        m(1, j) = r1[j];
        m(2, j) = r2[j];
    }
    return m;
}

// Row i of m as a vector.
[[nodiscard]] constexpr Vec3 row(const Mat3& m, std::size_t i)
{
    return {m(i, 0), m(i, 1), m(i, 2)};
}

// Expansion along the first row.
[[nodiscard]] constexpr double determinant(const Mat3& m)
{
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
           m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

// The inverse by the adjugate, m⁻¹ = adj(m) / det(m). Precondition: det(m) is not zero; the caller has checked it
// against its own tolerance (an element refuses a degenerate Jacobian with a diagnostic before calling this).
[[nodiscard]] constexpr Mat3 inverse(const Mat3& m)
{
    const double scale = 1.0 / determinant(m);
    Mat3 r;
    r(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * scale;
    r(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * scale;
    r(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * scale;
    r(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * scale;
    r(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * scale;
    r(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * scale;
    r(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * scale;
    r(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * scale;
    r(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * scale;
    return r;
}

} // namespace oos::base
