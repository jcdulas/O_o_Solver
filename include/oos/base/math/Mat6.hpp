#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"

namespace oos::base
{

// A 6 × 6 matrix: the node block of a stiffness or mass matrix (3 translations then 3 rotations), a
// CBUSH stiffness, a rigid-body mass matrix.
using Mat6 = MatN<6>;

// The 6 × 6 transformation of a grid point's six components for a 3 × 3 rotation r: diag(r, r), translations and
// rotations turned alike.
[[nodiscard]] constexpr Mat6 rotationBlock(const Mat3& r)
{
    Mat6 t;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            t(i, j) = r(i, j);
            t(i + 3, j + 3) = r(i, j);
        }
    return t;
}

} // namespace oos::base
