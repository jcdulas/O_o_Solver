#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// A value with its gradient and its Hessian over N local variables: forward-mode differentiation to second order,
// used for the second variation of the nonlinear strain measures (the geometric stiffness). The variables are those
// of one point (N = 15 on a shell point: the first and second derivatives of the position; N = 6 on an edge point).
// Each operation applies the chain rule to second order,
//   f(a)_,i = f'(a) a_,i,   f(a)_,ij = f'(a) a_,ij + f''(a) a_,i a_,j,
// and the product rule (ab)_,ij = a_,ij b + a b_,ij + a_,i b_,j + a_,j b_,i. The Hessian is kept full (both triangles),
// so that its symmetry is that of the formulas, not imposed.
//
// Thread-safety: a plain value type; the functions are pure.
template <std::size_t N>
struct HessianScalar
{
    double value = 0.0;
    std::array<double, N> gradient{};
    std::array<std::array<double, N>, N> hessian{};

    // The constant v: gradient and Hessian zero.
    [[nodiscard]] static HessianScalar constant(double v)
    {
        HessianScalar x;
        x.value = v;
        return x;
    }

    // The local variable number `index` at the value v: its gradient is the unit vector e_index.
    [[nodiscard]] static HessianScalar variable(double v, std::size_t index)
    {
        HessianScalar x = constant(v);
        x.gradient[index] = 1.0;
        return x;
    }
};

template <std::size_t N>
[[nodiscard]] HessianScalar<N> operator+(const HessianScalar<N>& a, const HessianScalar<N>& b)
{
    HessianScalar<N> r = a;
    r.value += b.value;
    for (std::size_t i = 0; i < N; ++i)
    {
        r.gradient[i] += b.gradient[i];
        for (std::size_t j = 0; j < N; ++j)
            r.hessian[i][j] += b.hessian[i][j];
    }
    return r;
}

template <std::size_t N>
[[nodiscard]] HessianScalar<N> operator-(const HessianScalar<N>& a, const HessianScalar<N>& b)
{
    HessianScalar<N> r = a;
    r.value -= b.value;
    for (std::size_t i = 0; i < N; ++i)
    {
        r.gradient[i] -= b.gradient[i];
        for (std::size_t j = 0; j < N; ++j)
            r.hessian[i][j] -= b.hessian[i][j];
    }
    return r;
}

// a·s, s a plain number.
template <std::size_t N>
[[nodiscard]] HessianScalar<N> scale(const HessianScalar<N>& a, double s)
{
    HessianScalar<N> r = a;
    r.value *= s;
    for (std::size_t i = 0; i < N; ++i)
    {
        r.gradient[i] *= s;
        for (std::size_t j = 0; j < N; ++j)
            r.hessian[i][j] *= s;
    }
    return r;
}

// (ab)_,i = a_,i b + a b_,i;  (ab)_,ij = a_,ij b + a b_,ij + a_,i b_,j + a_,j b_,i.
template <std::size_t N>
[[nodiscard]] HessianScalar<N> operator*(const HessianScalar<N>& a, const HessianScalar<N>& b)
{
    HessianScalar<N> r = HessianScalar<N>::constant(a.value * b.value);
    for (std::size_t i = 0; i < N; ++i)
    {
        r.gradient[i] = a.gradient[i] * b.value + a.value * b.gradient[i];
        for (std::size_t j = 0; j < N; ++j)
            r.hessian[i][j] = a.hessian[i][j] * b.value + a.value * b.hessian[i][j] + a.gradient[i] * b.gradient[j] +
                              a.gradient[j] * b.gradient[i];
    }
    return r;
}

// f(a) from f(a.value) = f0, f'(a.value) = f1 and f''(a.value) = f2 by the chain rule to second order.
template <std::size_t N>
[[nodiscard]] HessianScalar<N> applyChainRule(const HessianScalar<N>& a, double f0, double f1, double f2)
{
    HessianScalar<N> r = HessianScalar<N>::constant(f0);
    for (std::size_t i = 0; i < N; ++i)
    {
        r.gradient[i] = f1 * a.gradient[i];
        for (std::size_t j = 0; j < N; ++j)
            r.hessian[i][j] = f1 * a.hessian[i][j] + f2 * a.gradient[i] * a.gradient[j];
    }
    return r;
}

// √a: f' = 1 / (2 √a), f'' = −1 / (4 a √a).
template <std::size_t N>
[[nodiscard]] HessianScalar<N> sqrt(const HessianScalar<N>& a)
{
    const double root = std::sqrt(a.value);
    return applyChainRule(a, root, 0.5 / root, -0.25 / (root * a.value));
}

// 1 / a: f' = −1 / a², f'' = 2 / a³.
template <std::size_t N>
[[nodiscard]] HessianScalar<N> reciprocal(const HessianScalar<N>& a)
{
    const double r = 1.0 / a.value;
    return applyChainRule(a, r, -r * r, 2.0 * r * r * r);
}

// The angle atan2(y, x) with its derivatives. With r² = x² + y²: θ_,x = −y / r², θ_,y = x / r²,
// θ_,xx = 2xy / r⁴ = −θ_,yy, θ_,xy = (y² − x²) / r⁴.
template <std::size_t N>
[[nodiscard]] HessianScalar<N> atan2(const HessianScalar<N>& y, const HessianScalar<N>& x)
{
    const double r2 = x.value * x.value + y.value * y.value;
    HessianScalar<N> a = HessianScalar<N>::constant(std::atan2(y.value, x.value));
    const double gx = -y.value / r2;
    const double gy = x.value / r2;
    const double hxx = 2.0 * x.value * y.value / (r2 * r2);
    const double hyy = -hxx;
    const double hxy = (y.value * y.value - x.value * x.value) / (r2 * r2);
    for (std::size_t i = 0; i < N; ++i)
    {
        a.gradient[i] = gx * x.gradient[i] + gy * y.gradient[i];
        for (std::size_t j = 0; j < N; ++j)
            a.hessian[i][j] = gx * x.hessian[i][j] + gy * y.hessian[i][j] + hxx * x.gradient[i] * x.gradient[j] +
                              hyy * y.gradient[i] * y.gradient[j] +
                              hxy * (x.gradient[i] * y.gradient[j] + y.gradient[i] * x.gradient[j]);
    }
    return a;
}

// ---- 3-vectors of HessianScalar ---------------------------------------------------------------------------------

template <std::size_t N>
using HessianVec3 = std::array<HessianScalar<N>, 3>;

template <std::size_t N>
[[nodiscard]] HessianScalar<N> hessianDot(const HessianVec3<N>& a, const HessianVec3<N>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <std::size_t N>
[[nodiscard]] HessianVec3<N> hessianCross(const HessianVec3<N>& a, const HessianVec3<N>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// a / |a|; the caller makes sure a is not zero.
template <std::size_t N>
[[nodiscard]] HessianVec3<N> hessianUnit(const HessianVec3<N>& a)
{
    const HessianScalar<N> inverseNorm = reciprocal(sqrt(hessianDot(a, a)));
    return {a[0] * inverseNorm, a[1] * inverseNorm, a[2] * inverseNorm};
}

} // namespace oos::elements::mixedHybrid
