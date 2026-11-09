#pragma once

#include <array>
#include <cstddef>

namespace oos::base
{

// A small dense matrix of fixed size, row-major, for element kernels (a 6 × 6 node block, a 24 × 24 shell stiffness):
// MatN<6> is square, MatN<3, 24> is a strain-displacement matrix. It lives on the stack; larger or variable-size
// matrices are linalg::DenseMatrix. Every operation sums in a fixed order (index increasing), so that a result has the
// same bits everywhere. A plain value type.
template <std::size_t Rows, std::size_t Cols = Rows>
struct MatN
{
    std::array<double, Rows * Cols> values{}; // row-major: term (i, j) at i·Cols + j

    [[nodiscard]] static constexpr std::size_t rows() { return Rows; }
    [[nodiscard]] static constexpr std::size_t cols() { return Cols; }

    [[nodiscard]] constexpr double operator()(std::size_t i, std::size_t j) const { return values[i * Cols + j]; }
    [[nodiscard]] constexpr double& operator()(std::size_t i, std::size_t j) { return values[i * Cols + j]; }

    [[nodiscard]] static constexpr MatN identity()
        requires(Rows == Cols)
    {
        MatN m;
        for (std::size_t i = 0; i < Rows; ++i)
            m(i, i) = 1.0;
        return m;
    }

    [[nodiscard]] constexpr MatN<Cols, Rows> transposed() const
    {
        MatN<Cols, Rows> t;
        for (std::size_t i = 0; i < Rows; ++i)
            for (std::size_t j = 0; j < Cols; ++j)
                t(j, i) = (*this)(i, j);
        return t;
    }

    constexpr MatN& operator+=(const MatN& other)
    {
        for (std::size_t k = 0; k < Rows * Cols; ++k)
            values[k] += other.values[k];
        return *this;
    }
    constexpr MatN& operator-=(const MatN& other)
    {
        for (std::size_t k = 0; k < Rows * Cols; ++k)
            values[k] -= other.values[k];
        return *this;
    }
    constexpr MatN& operator*=(double scale)
    {
        for (double& value : values)
            value *= scale;
        return *this;
    }

    friend constexpr MatN operator+(MatN x, const MatN& y) { return x += y; }
    friend constexpr MatN operator-(MatN x, const MatN& y) { return x -= y; }
    friend constexpr MatN operator*(double scale, MatN x) { return x *= scale; }
    friend constexpr MatN operator*(MatN x, double scale) { return x *= scale; }
    friend constexpr bool operator==(const MatN&, const MatN&) = default;
};

// Matrix product (x·y)(i, j) = Σ_k x(i, k)·y(k, j), summed for k increasing.
template <std::size_t R, std::size_t K, std::size_t C>
[[nodiscard]] constexpr MatN<R, C> operator*(const MatN<R, K>& x, const MatN<K, C>& y)
{
    MatN<R, C> r;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k)
                sum += x(i, k) * y(k, j);
            r(i, j) = sum;
        }
    return r;
}

// Matrix-vector product (x·v)(i) = Σ_j x(i, j)·v(j), summed for j increasing.
template <std::size_t R, std::size_t C>
[[nodiscard]] constexpr std::array<double, R> operator*(const MatN<R, C>& x, const std::array<double, C>& v)
{
    std::array<double, R> r{};
    for (std::size_t i = 0; i < R; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < C; ++j)
            sum += x(i, j) * v[j];
        r[i] = sum;
    }
    return r;
}

// The congruence tᵀ·k·t that carries a stiffness or mass matrix from one set of coordinates to another (u = t·u'),
// e.g. from the element frame to the displacement frames of its grid points.
template <std::size_t N, std::size_t M>
[[nodiscard]] constexpr MatN<M, M> congruence(const MatN<N, N>& k, const MatN<N, M>& t)
{
    return t.transposed() * (k * t);
}

} // namespace oos::base
