#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace oos::elements
{

// K = Hᵀ·diag(d)·H, the matrix of an element whose R generalised deformations e = H·u (relative motions, axial
// stretch and twist...) carry independent stiffnesses d: K(i, j) = Σ_r d_r·H(r, i)·H(r, j), summed for r increasing.
template <std::size_t R, std::size_t N>
[[nodiscard]] constexpr base::MatN<N, N> diagonalCongruence(const std::array<double, R>& d, const base::MatN<R, N>& h)
{
    base::MatN<N, N> k;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < R; ++r)
                sum += d[r] * h(r, i) * h(r, j);
            k(i, j) = sum;
        }
    return k;
}

// Carries a matrix over 3-component blocks (translations or rotations of the grid points) from the basic system to
// the displacement systems: K_CD = Tᵀ·K_basic·T with T = diag(R_0, R_1, ...), u_basic = T·u_CD, R_b the CD-to-basic
// rotation of block b (ElementBatchView::displacementFrame). Block by block: K_CD(a, b) = R_aᵀ·K(a, b)·R_b.
template <std::size_t N>
[[nodiscard]] constexpr base::MatN<N, N> toDisplacementFrames(const base::MatN<N, N>& kBasic,
                                                              const std::array<base::Mat3, N / 3>& frames)
{
    static_assert(N % 3 == 0, "the matrix must be made of 3 x 3 blocks");
    base::MatN<N, N> result;
    for (std::size_t a = 0; a < N / 3; ++a)
        for (std::size_t b = 0; b < N / 3; ++b)
        {
            base::Mat3 block;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    block(i, j) = kBasic(3 * a + i, 3 * b + j);
            const base::Mat3 rotated = frames[a].transposed() * block * frames[b];
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    result(3 * a + i, 3 * b + j) = rotated(i, j);
        }
    return result;
}

// The cross-product matrix [d]× of a vector: [d]×·v = d × v.
[[nodiscard]] constexpr base::Mat3 crossProductMatrix(base::Vec3 d)
{
    return base::matrixFromRows({0.0, -d[2], d[1]}, {d[2], 0.0, -d[0]}, {-d[1], d[0], 0.0});
}

// The six DOFs (translations 1-3, rotations 4-6) of the grid point of a row, in the order of the matrices above.
[[nodiscard]] inline std::array<ElementDof, 6> gridPointDofs(std::int32_t gridRow)
{
    return {ElementDof::grid(gridRow, 1), ElementDof::grid(gridRow, 2), ElementDof::grid(gridRow, 3),
            ElementDof::grid(gridRow, 4), ElementDof::grid(gridRow, 5), ElementDof::grid(gridRow, 6)};
}

// The inverse of a small non-singular matrix by Gauss-Jordan elimination with partial pivoting (the condensation of the
// incompatible modes of a ShellQuad4, the moment flexibility of its hybrid bending). Precondition: the matrix is far
// from singular, which the element geometry checks ensure (a degenerate element is refused before its matrices are
// formed).
template <std::size_t N>
[[nodiscard]] constexpr base::MatN<N, N> inverted(base::MatN<N, N> a)
{
    base::MatN<N, N> inverse = base::MatN<N, N>::identity();
    for (std::size_t k = 0; k < N; ++k)
    {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < N; ++i)
            if (std::abs(a(i, k)) > std::abs(a(pivot, k)))
                pivot = i;
        if (pivot != k)
            for (std::size_t j = 0; j < N; ++j)
            {
                std::swap(a(k, j), a(pivot, j));
                std::swap(inverse(k, j), inverse(pivot, j));
            }
        const double scale = 1.0 / a(k, k);
        for (std::size_t j = 0; j < N; ++j)
        {
            a(k, j) *= scale;
            inverse(k, j) *= scale;
        }
        for (std::size_t i = 0; i < N; ++i)
        {
            if (i == k)
                continue;
            const double factor = a(i, k);
            for (std::size_t j = 0; j < N; ++j)
            {
                a(i, j) -= factor * a(k, j);
                inverse(i, j) -= factor * inverse(k, j);
            }
        }
    }
    return inverse;
}

// Appends a fixed-size element matrix to a block.
template <std::size_t N>
void appendMatrix(ElementMatrixBlock& block, std::int64_t row, const std::array<ElementDof, N>& dofs,
                  const base::MatN<N, N>& matrix)
{
    block.append(row, std::span<const ElementDof>(dofs), std::span<const double>(matrix.values));
}

} // namespace oos::elements
