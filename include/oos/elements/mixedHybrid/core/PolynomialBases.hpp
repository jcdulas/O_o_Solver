#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// Orthogonal polynomial bases of the reference elements of the mixed hybrid formulation:
// - Legendre polynomials on [−1, 1] and the total-degree Legendre basis P_i(u) P_j(v), i + j ≤ d, of the square;
// - Jacobi polynomials P_n^(α,0);
// - the orthogonal basis of the reference triangle (−1, −1), (1, −1), (−1, 1) and of the reference tetrahedron
//   (−1, −1, −1), (1, −1, −1), (−1, 1, −1), (−1, −1, 1), built on the collapsed coordinates of M. Dubiner (Spectral
//   methods on triangles and other domains, Journal of Scientific Computing 6 (1991) 345-390) and of S. J. Sherwin,
//   G. E. Karniadakis (A new triangular and tetrahedral basis for high-order (hp) finite element methods,
//   International Journal for Numerical Methods in Engineering 38 (1995) 3775-3802).
// The recurrences of the Legendre and Jacobi polynomials are those of M. Abramowitz, I. A. Stegun, Handbook of
// Mathematical Functions, 1964, section 22.7. The collapsed bases are written as polynomials of the reference
// coordinates without any division (homogeneous forms q^k P_k(p / q)), so that they are regular at the collapsed
// vertex and can be differentiated through the Dual scalar.
//
// Orderings: the bases of total degree d list their functions by increasing total degree, so that the basis of degree
// d − 1 comes first. The generic functions (templates on the scalar type) are evaluated with double or Dual.
//
// Thread-safety: pure functions.

// Values and derivatives of the polynomials of degrees 0 to d at one point.
struct PolynomialTable
{
    std::vector<double> values;
    std::vector<double> derivatives;
};

// The Legendre polynomials P_0 … P_d and their derivatives at x:
//   (k) P_k = (2k − 1) x P_{k−1} − (k − 1) P_{k−2},   P'_k = P'_{k−2} + (2k − 1) P_{k−1}.
[[nodiscard]] PolynomialTable legendreTable(std::size_t degree, double x);

// The Legendre polynomial P_n at x, by the same recurrence.
[[nodiscard]] double legendreValue(std::size_t n, double x);

// The number of polynomials of total degree ≤ d in two variables, (d + 1)(d + 2) / 2.
[[nodiscard]] constexpr std::size_t totalDegreeCount2D(std::size_t degree)
{
    return (degree + 1) * (degree + 2) / 2;
}

// The number of polynomials of total degree ≤ d in three variables, (d + 1)(d + 2)(d + 3) / 6.
[[nodiscard]] constexpr std::size_t totalDegreeCount3D(std::size_t degree)
{
    return (degree + 1) * (degree + 2) * (degree + 3) / 6;
}

// The total-degree Legendre basis of degree d on the square at (u, v): P_i(u) P_{t−i}(v) for t = 0 … d and i = 0 … t,
// in that order. `out` holds at least totalDegreeCount2D(d) values.
void legendreBasis(std::size_t degree, double u, double v, std::span<double> out);

// The same basis with its derivatives d/du and d/dv.
void legendreBasisWithDerivatives(std::size_t degree, double u, double v, std::span<double> values,
                                  std::span<double> du, std::span<double> dv);

// The Jacobi polynomials P_0^(α,0) … P_d^(α,0) and their derivatives at x (orthogonal on [−1, 1] for the weight
// (1 − x)^α). With c = 2n + α:
//   2 (n + 1)(n + α + 1) c P_{n+1} = (c + 1) [(c + 2) c x + α²] P_n − 2 (n + α) n (c + 2) P_{n−1},
// P_0 = 1, P_1 = (α + (α + 2) x) / 2.
[[nodiscard]] PolynomialTable jacobiTable(std::size_t degree, double alpha, double x);

// The orthogonal basis of total degree d on the reference triangle (Dubiner) and its derivatives d/du, d/dv:
//   ψ_ij = q^i P_i(p / q) P_j^(2i+1,0)(v),   p = u + (1 + v) / 2,   q = (1 − v) / 2,   i + j ≤ d,
// ordered by total degree t = i + j, then i increasing. The factor s_i = q^i P_i(p / q) is computed by the
// homogeneous Legendre recurrence s_{i+1} = ((2i + 1) p s_i − i q² s_{i−1}) / (i + 1), with its derivatives in p and
// q; d/du = d/dp and d/dv = (d/dp − d/dq) / 2. The spans hold at least totalDegreeCount2D(d) values.
void triangleBasisWithDerivatives(std::size_t degree, double u, double v, std::span<double> values,
                                  std::span<double> du, std::span<double> dv);

// The values of the same basis.
void triangleBasis(std::size_t degree, double u, double v, std::span<double> out);

// q^k P_k^(α,0)(p / q) for k = 0 … n: the Jacobi polynomials made homogeneous (Legendre for α = 0), by the recurrence
// of jacobiTable multiplied through by q^(k+1):
//   S_0 = 1,  S_1 = (α + 2) p / 2 + α q / 2,  S_{k+1} = (a1 p + a0 q) S_k − b q² S_{k−1},
// with a1 = (c + 1)(c + 2) c / den, a0 = (c + 1) α² / den, b = 2 (k + α) k (c + 2) / den, c = 2k + α,
// den = 2 (k + 1)(k + α + 1) c. No division by q: a polynomial of (p, q), regular where q = 0.
template <Scalar T>
[[nodiscard]] std::vector<T> scaledJacobi(std::size_t n, double alpha, const T& p, const T& q)
{
    std::vector<T> out;
    out.reserve(n + 1);
    out.push_back(constantOf<T>(1.0));
    if (n >= 1)
        out.push_back(scale(p, 0.5 * (alpha + 2.0)) + scale(q, 0.5 * alpha));
    for (std::size_t k = 1; k < n; ++k)
    {
        const double m = static_cast<double>(k);
        const double c = 2.0 * m + alpha;
        const double den = 2.0 * (m + 1.0) * (m + alpha + 1.0) * c;
        const double a1 = (c + 1.0) * (c + 2.0) * c / den;
        const double a0 = (c + 1.0) * alpha * alpha / den;
        const double b = 2.0 * (m + alpha) * m * (c + 2.0) / den;
        const T next = (scale(p, a1) + scale(q, a0)) * out[k] - scale(q * q, b) * out[k - 1];
        out.push_back(next);
    }
    return out;
}

// The orthogonal basis of total degree d on the reference tetrahedron (not normalised), by increasing total degree
// (the basis of degree d − 1 comes first), then i, then j:
//   S_i(p1, q1) S_j^(2i+1)(p2, q2) P_k^(2i+2j+2,0)(t),   i + j + k ≤ d,
// S the homogeneous polynomials of scaledJacobi, p1 = 1 + r + (s + t) / 2, q1 = −(s + t) / 2, p2 = s + (1 + t) / 2,
// q2 = (1 − t) / 2 (Sherwin and Karniadakis, 1995). Polynomials of (r, s, t).
template <Scalar T>
[[nodiscard]] std::vector<T> tetrahedronBasis(std::size_t degree, const T& r, const T& s, const T& t)
{
    const T one = constantOf<T>(1.0);
    const T half = constantOf<T>(0.5);
    const T p1 = one + r + (s + t) * half;
    const T q1 = -(s + t) * half;
    const T p2 = s + (one + t) * half;
    const T q2 = (one - t) * half;
    const std::vector<T> si = scaledJacobi(degree, 0.0, p1, q1);
    std::vector<std::vector<T>> sj;
    sj.reserve(degree + 1);
    for (std::size_t i = 0; i <= degree; ++i)
        sj.push_back(scaledJacobi(degree - i, static_cast<double>(2 * i + 1), p2, q2));
    // pk[i][j][k] = P_k^(2i+2j+2,0)(t)
    std::vector<std::vector<std::vector<T>>> pk(degree + 1);
    for (std::size_t i = 0; i <= degree; ++i)
        for (std::size_t j = 0; j <= degree - i; ++j)
            pk[i].push_back(scaledJacobi(degree - i - j, static_cast<double>(2 * i + 2 * j + 2), t, one));
    std::vector<T> out;
    out.reserve(totalDegreeCount3D(degree));
    for (std::size_t total = 0; total <= degree; ++total)
        for (std::size_t i = 0; i <= total; ++i)
            for (std::size_t j = 0; j <= total - i; ++j)
                out.push_back(si[i] * sj[i][j] * pk[i][j][total - i - j]);
    return out;
}

} // namespace oos::elements::mixedHybrid
