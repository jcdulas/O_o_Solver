#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The polynomial spaces of degree n of the reference solids, in which the displacement and the stresses of a solid
// element of the mixed hybrid formulation are written, and the coordinates of their faces:
//
// | shape       | reference                                         | space of degree n                               |
// |-------------|---------------------------------------------------|-------------------------------------------------|
// | tetrahedron | (−1,−1,−1), (1,−1,−1), (−1,1,−1), (−1,−1,1)       | P_n (tetrahedronBasis)                          |
// | pyramid     | base [−1, 1]² at z = −1, apex (0, 0, 1)           | P_i(a) P_j(b) q^max(i,j) P_k^(2m+2,0)(z),       |
// |             |                                                   | a = x / q, b = y / q, q = (1 − z) / 2,          |
// |             |                                                   | m = max(i, j), m + k ≤ n                        |
// | pentahedron | reference triangle × [−1, 1]                      | P_n(triangle) ⊗ P_n(z)                          |
// | hexahedron  | [−1, 1]³                                          | Q_n                                             |
//
// The space of the pyramid is the rational space of M. Bergot, G. Cohen, M. Duruflé (Higher-order finite elements
// for hybrid meshes using new nodal pyramidal elements, Journal of Scientific Computing 42 (2010) 345-381): its
// traces are P_n on the triangular faces and Q_n on the base, so that every space has P_n on its triangular faces and
// Q_n on its quadrilateral faces. The `levels` functions give the degree of each function of a basis: the space of
// degree m ≤ n is spanned by the functions of level m at most.
//
// Thread-safety: pure functions.

// The functions P_0 … P_n at x (the Legendre polynomials as homogeneous Jacobi polynomials with q = 1).
template <Scalar T>
[[nodiscard]] std::vector<T> legendreAll(std::size_t n, const T& x)
{
    return scaledJacobi(n, 0.0, x, constantOf<T>(1.0));
}

// The orthogonal basis of P_n on the reference triangle (−1, −1), (1, −1), (−1, 1) at (u, v): the basis of
// triangleBasis, in the same order, written for any scalar type.
template <Scalar T>
[[nodiscard]] std::vector<T> triangleSpace(std::size_t n, const T& u, const T& v)
{
    const T one = constantOf<T>(1.0);
    const T half = constantOf<T>(0.5);
    const std::vector<T> si = scaledJacobi(n, 0.0, u + (one + v) * half, (one - v) * half);
    std::vector<T> out;
    out.reserve(totalDegreeCount2D(n));
    for (std::size_t total = 0; total <= n; ++total)
    {
        for (std::size_t i = 0; i <= total; ++i)
        {
            const std::vector<T> pj = scaledJacobi(total - i, static_cast<double>(2 * i + 1), v, one);
            out.push_back(si[i] * pj[total - i]);
        }
    }
    return out;
}

// Dimensions of the spaces of degree n.
[[nodiscard]] std::size_t tetrahedronSpaceDimension(std::size_t n);
[[nodiscard]] std::size_t pyramidSpaceDimension(std::size_t n);
[[nodiscard]] std::size_t pentahedronSpaceDimension(std::size_t n);
[[nodiscard]] std::size_t hexahedronSpaceDimension(std::size_t n);

// The basis of P_n on the reference tetrahedron at r: tetrahedronBasis.
template <Scalar T>
[[nodiscard]] std::vector<T> tetrahedronSpace(std::size_t n, const std::array<T, 3>& r)
{
    return tetrahedronBasis(n, r[0], r[1], r[2]);
}

// The basis of Q_n on [−1, 1]³ at r: P_i(x) P_j(y) P_k(z), k fastest, then j, then i.
template <Scalar T>
[[nodiscard]] std::vector<T> hexahedronSpace(std::size_t n, const std::array<T, 3>& r)
{
    const std::vector<T> a = legendreAll(n, r[0]);
    const std::vector<T> b = legendreAll(n, r[1]);
    const std::vector<T> c = legendreAll(n, r[2]);
    std::vector<T> out;
    out.reserve(hexahedronSpaceDimension(n));
    for (std::size_t i = 0; i <= n; ++i)
        for (std::size_t j = 0; j <= n; ++j)
            for (std::size_t k = 0; k <= n; ++k)
                out.push_back(a[i] * b[j] * c[k]);
    return out;
}

// The basis of P_n(triangle) ⊗ P_n(z) at r: each function of triangleSpace times P_0(z) … P_n(z).
template <Scalar T>
[[nodiscard]] std::vector<T> pentahedronSpace(std::size_t n, const std::array<T, 3>& r)
{
    const std::vector<T> triangle = triangleSpace(n, r[0], r[1]);
    const std::vector<T> c = legendreAll(n, r[2]);
    std::vector<T> out;
    out.reserve(triangle.size() * (n + 1));
    for (const T& t : triangle)
        for (const T& ck : c)
            out.push_back(t * ck);
    return out;
}

// Below this |q| a point of the pyramid is its apex, where the functions with min(i, j) > 0 vanish.
inline constexpr double PYRAMID_APEX_TOLERANCE = 1e-14;

// The rational space of the pyramid (Bergot, Cohen, Duruflé, 2010) at r, for m = 0 … n, then i, j ≤ m with
// max(i, j) = m, then k = 0 … n − m:
//   q^m P_i(a) P_j(b) P_k^(2m+2,0)(z) = (q^i P_i(x / q)) (q^j P_j(y / q)) / q^min(i,j) · P_k^(2m+2,0)(z).
// The homogeneous factors q^i P_i(x / q) are polynomials; the division by q^min(i,j) is the rational part, and the
// limit at the apex (q = 0) of a function with min(i, j) > 0 is zero.
template <Scalar T>
[[nodiscard]] std::vector<T> pyramidSpace(std::size_t n, const std::array<T, 3>& r)
{
    const T one = constantOf<T>(1.0);
    const T half = constantOf<T>(0.5);
    const T q = (one - r[2]) * half;
    const std::vector<T> sx = scaledJacobi(n, 0.0, r[0], q);
    const std::vector<T> sy = scaledJacobi(n, 0.0, r[1], q);
    const bool apex = std::fabs(valueOf(q)) < PYRAMID_APEX_TOLERANCE;
    std::vector<T> out;
    out.reserve(pyramidSpaceDimension(n));
    for (std::size_t m = 0; m <= n; ++m)
    {
        const std::vector<T> pk = scaledJacobi(n - m, static_cast<double>(2 * m + 2), r[2], one);
        for (std::size_t i = 0; i <= m; ++i)
        {
            for (std::size_t j = 0; j <= m; ++j)
            {
                if (std::max(i, j) != m)
                    continue;
                const std::size_t lowest = std::min(i, j);
                T f = sx[i] * sy[j];
                if (lowest > 0)
                {
                    if (apex)
                        f = constantOf<T>(0.0);
                    else
                        for (std::size_t power = 0; power < lowest; ++power)
                            f = f / q;
                }
                for (const T& pkk : pk)
                    out.push_back(f * pkk);
            }
        }
    }
    return out;
}

// The level (degree) of each function of the bases above, in their order.
[[nodiscard]] std::vector<std::size_t> tetrahedronLevels(std::size_t n);
[[nodiscard]] std::vector<std::size_t> pyramidLevels(std::size_t n);
[[nodiscard]] std::vector<std::size_t> pentahedronLevels(std::size_t n);
[[nodiscard]] std::vector<std::size_t> hexahedronLevels(std::size_t n);

// The barycentric coordinates (L0, L1, L2) of the point (u, v) of the reference triangle (−1, −1), (1, −1), (−1, 1):
// L0 = −(u + v) / 2, L1 = (1 + u) / 2, L2 = (1 + v) / 2.
template <Scalar T>
[[nodiscard]] std::array<T, 3> triangleBarycentric(const T& u, const T& v)
{
    const T one = constantOf<T>(1.0);
    const T half = constantOf<T>(0.5);
    return {-(u + v) * half, (one + u) * half, (one + v) * half};
}

// The weights of the corners of a face at its coordinates c: a triangle (3 corners, c = (L1, L2)) has the barycentric
// weights (1 − L1 − L2, L1, L2); a quadrilateral (4 corners, c = (s, t) in [0, 1]²) the bilinear ones
// ((1 − s)(1 − t), s (1 − t), s t, (1 − s) t).
template <Scalar T>
[[nodiscard]] std::vector<T> faceWeights(std::size_t cornerCount, const std::array<T, 2>& c)
{
    const T one = constantOf<T>(1.0);
    if (cornerCount == 3)
        return {one - c[0] - c[1], c[0], c[1]};
    const T& s = c[0];
    const T& t = c[1];
    return {(one - s) * (one - t), s * (one - t), s * t, (one - s) * t};
}

} // namespace oos::elements::mixedHybrid
