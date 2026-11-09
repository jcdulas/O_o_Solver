#pragma once

// Small fixed-size vectors and matrices whose scalar is a template parameter, for the shell kernels that are evaluated
// both in double and in std::complex<double> (ShellQuad4, ShellTria3).
//
// Why complex numbers: the design derivatives of these kernels are taken by the complex step (Squire & Trapp, "Using
// complex variables to estimate derivatives of real functions", SIAM Review 40, 1998; Martins, Sturdza & Alonso, "The
// complex-step derivative approximation", ACM TOMS 29, 2003). For a real analytic f, f(x + i·h) = f(x) + i·h·f'(x) +
// O(h²), so f'(x) = Im f(x + i·h) / h with no subtraction: the derivative is exact to round-off for any tiny h. Every
// operation of a kernel must therefore be analytic in the perturbed inputs: the helpers below compare real parts
// (a branch is decided by the value, not by the perturbation), take |x| as ±x by the sign of the real part, and use
// the dot product without conjugation.

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

namespace oos::elements
{

using Complex = std::complex<double>;

// The complex step h: small enough that h² vanishes against any value, large enough to stay a normal double.
inline constexpr double COMPLEX_STEP = 1.0e-30;

// The real part of a scalar (the scalar itself for a double).
template <typename T>
[[nodiscard]] constexpr double realPart(const T& x) noexcept
{
    if constexpr (std::is_same_v<T, double>)
        return x;
    else
        return x.real();
}

// The scalar of value `value` moved along the rate `rate` by the complex step.
[[nodiscard]] inline Complex complexStepped(double value, double rate) noexcept
{
    return {value, COMPLEX_STEP * rate};
}

// The derivative carried by a complex-stepped result.
[[nodiscard]] inline double complexStepDerivative(const Complex& x) noexcept
{
    return x.imag() / COMPLEX_STEP;
}

// |x| continued analytically: x or −x by the sign of the real part.
template <typename T>
[[nodiscard]] T analyticAbs(const T& x)
{
    return realPart(x) < 0.0 ? -x : x;
}

template <typename T>
[[nodiscard]] T squareRoot(const T& x)
{
    using std::sqrt;
    return sqrt(x);
}

// The smaller and larger of two scalars, chosen by their real parts.
template <typename T>
[[nodiscard]] T smallerOf(const T& a, const T& b)
{
    return realPart(b) < realPart(a) ? b : a;
}

template <typename T>
[[nodiscard]] T largerOf(const T& a, const T& b)
{
    return realPart(b) > realPart(a) ? b : a;
}

template <typename T>
using Vector3 = std::array<T, 3>;

template <typename T>
[[nodiscard]] T dotOf(const Vector3<T>& a, const Vector3<T>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
[[nodiscard]] Vector3<T> crossOf(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template <typename T, typename S>
[[nodiscard]] Vector3<T> scaledBy(const Vector3<T>& a, const S& s)
{
    return {a[0] * s, a[1] * s, a[2] * s};
}

template <typename T>
[[nodiscard]] Vector3<T> sumOf(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

// a / |a| (the analytic norm √(a·a)).
template <typename T>
[[nodiscard]] Vector3<T> unitOf(const Vector3<T>& a)
{
    return scaledBy(a, T(1.0) / squareRoot(dotOf(a, a)));
}

// A row-major R × C matrix of scalars T.
template <typename T, std::size_t R, std::size_t C = R>
struct ScalarMatrix
{
    std::array<T, R * C> values{};

    [[nodiscard]] static constexpr std::size_t rows() noexcept { return R; }
    [[nodiscard]] static constexpr std::size_t cols() noexcept { return C; }
    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const { return values[i * C + j]; }
    [[nodiscard]] T& operator()(std::size_t i, std::size_t j) { return values[i * C + j]; }
};

template <typename T>
using Matrix2 = ScalarMatrix<T, 2>;
template <typename T>
using Matrix3 = ScalarMatrix<T, 3>;

template <typename T, std::size_t R, std::size_t K, std::size_t C>
[[nodiscard]] ScalarMatrix<T, R, C> productOf(const ScalarMatrix<T, R, K>& a, const ScalarMatrix<T, K, C>& b)
{
    ScalarMatrix<T, R, C> result;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t k = 0; k < K; ++k)
        {
            const T aik = a(i, k);
            for (std::size_t j = 0; j < C; ++j)
                result(i, j) += aik * b(k, j);
        }
    return result;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] std::array<T, R> productOf(const ScalarMatrix<T, R, C>& a, const std::array<T, C>& v)
{
    std::array<T, R> result{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            result[i] += a(i, j) * v[j];
    return result;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] ScalarMatrix<T, C, R> transposeOf(const ScalarMatrix<T, R, C>& a)
{
    ScalarMatrix<T, C, R> result;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            result(j, i) = a(i, j);
    return result;
}

// aᵀ·m·a: a congruence, used to rotate material matrices (Gₑ = Uᵀ G U).
template <typename T, std::size_t N, typename S>
[[nodiscard]] ScalarMatrix<T, N> congruenceOf(const ScalarMatrix<T, N>& m, const ScalarMatrix<S, N>& a)
{
    ScalarMatrix<T, N> result;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            T sum{};
            for (std::size_t k = 0; k < N; ++k)
                for (std::size_t l = 0; l < N; ++l)
                    sum += a(k, i) * m(k, l) * a(l, j);
            result(i, j) = sum;
        }
    return result;
}

template <typename T, std::size_t R, std::size_t C, typename S>
[[nodiscard]] ScalarMatrix<T, R, C> scaledBy(const ScalarMatrix<T, R, C>& a, const S& s)
{
    ScalarMatrix<T, R, C> result = a;
    for (T& value : result.values)
        value *= s;
    return result;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] ScalarMatrix<T, R, C> sumOf(const ScalarMatrix<T, R, C>& a, const ScalarMatrix<T, R, C>& b)
{
    ScalarMatrix<T, R, C> result = a;
    for (std::size_t k = 0; k < R * C; ++k)
        result.values[k] += b.values[k];
    return result;
}

template <typename T>
[[nodiscard]] T determinantOf(const Matrix2<T>& a)
{
    return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
}

template <typename T>
[[nodiscard]] T determinantOf(const Matrix3<T>& a)
{
    return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
           a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

// The inverse by cofactors; the caller checks the determinant first.
template <typename T>
[[nodiscard]] Matrix2<T> inverseOf(const Matrix2<T>& a)
{
    const T scale = T(1.0) / determinantOf(a);
    Matrix2<T> result;
    result(0, 0) = a(1, 1) * scale;
    result(0, 1) = -a(0, 1) * scale;
    result(1, 0) = -a(1, 0) * scale;
    result(1, 1) = a(0, 0) * scale;
    return result;
}

template <typename T>
[[nodiscard]] Matrix3<T> inverseOf(const Matrix3<T>& a)
{
    const T scale = T(1.0) / determinantOf(a);
    Matrix3<T> result;
    result(0, 0) = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * scale;
    result(0, 1) = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * scale;
    result(0, 2) = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * scale;
    result(1, 0) = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * scale;
    result(1, 1) = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * scale;
    result(1, 2) = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * scale;
    result(2, 0) = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * scale;
    result(2, 1) = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * scale;
    result(2, 2) = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * scale;
    return result;
}

// k += w·Bᵀ·G·B for a strain matrix B (S × D) and a symmetric constitutive matrix G (S × S).
template <typename T, std::size_t S, std::size_t D>
void addCongruence(ScalarMatrix<T, D>& k, const ScalarMatrix<T, S, D>& b, const ScalarMatrix<T, S>& g, const T& w)
{
    ScalarMatrix<T, S, D> gb = productOf(g, b);
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t s = 0; s < S; ++s)
        {
            const T bsi = w * b(s, i);
            if (bsi == T(0.0))
                continue; // the B matrices are sparse: half of each row is zero
            for (std::size_t j = 0; j < D; ++j)
                k(i, j) += bsi * gb(s, j);
        }
}

// The real parts (the values) and the complex-step derivatives of a matrix.
template <std::size_t R, std::size_t C>
[[nodiscard]] ScalarMatrix<double, R, C> realPartOf(const ScalarMatrix<Complex, R, C>& a)
{
    ScalarMatrix<double, R, C> result;
    for (std::size_t k = 0; k < R * C; ++k)
        result.values[k] = a.values[k].real();
    return result;
}

template <std::size_t R, std::size_t C>
[[nodiscard]] ScalarMatrix<double, R, C> complexStepDerivativeOf(const ScalarMatrix<Complex, R, C>& a)
{
    ScalarMatrix<double, R, C> result;
    for (std::size_t k = 0; k < R * C; ++k)
        result.values[k] = complexStepDerivative(a.values[k]);
    return result;
}

} // namespace oos::elements
