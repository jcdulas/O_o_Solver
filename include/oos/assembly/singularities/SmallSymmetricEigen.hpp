#pragma once

#include "oos/base/math/MatN.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace oos::assembly
{

// The eigenvalues (increasing) and the orthonormal eigenvectors (column j of `vectors` for values[j]) of a small
// symmetric matrix.
template <std::size_t N>
struct SmallEigenDecomposition
{
    std::array<double, N> values{};
    base::MatN<N> vectors;
};

// The eigen decomposition of a symmetric matrix of a few rows (the 3 × 3 node blocks of the singularity processor, the
// 6 × 6 normal matrix of an RigidInterpolation) by the cyclic Jacobi method (Golub and Van Loan, Matrix Computations,
// 4th ed., §8.5.3; the rotation of Numerical Recipes §11.1: θ = (a_qq − a_pp)/(2·a_pq), t = sgn(θ)/(|θ| + √(θ² + 1)),
// c = 1/√(t² + 1), s = t·c). Accurate to the round-off of the largest eigenvalue, also for the zero and nearly zero
// eigenvalues the singularity test looks at; the sweeps run in a fixed order, so the result has the same bits
// everywhere.
//
// Thread-safety: stateless.
template <std::size_t N>
class SmallSymmetricEigen
{
public:
    // Sweeps before giving up; the method converges quadratically and needs 4 to 8 for N ≤ 6.
    static constexpr int MAX_SWEEPS = 60;

    // `a` must be symmetric (its upper triangle is read with the lower one).
    [[nodiscard]] static SmallEigenDecomposition<N> decompose(const base::MatN<N>& a)
    {
        base::MatN<N> m = a;
        base::MatN<N> v = base::MatN<N>::identity();
        for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep)
        {
            double offDiagonal = 0.0;
            for (std::size_t p = 0; p < N; ++p)
                for (std::size_t q = p + 1; q < N; ++q)
                    offDiagonal += m(p, q) * m(p, q);
            if (offDiagonal == 0.0)
                break;
            for (std::size_t p = 0; p < N; ++p)
                for (std::size_t q = p + 1; q < N; ++q)
                    rotate(m, v, p, q);
        }
        SmallEigenDecomposition<N> result;
        std::array<std::size_t, N> order{};
        for (std::size_t k = 0; k < N; ++k)
            order[k] = k;
        // Insertion sort by increasing eigenvalue, ties kept in index order: N is at most 6.
        for (std::size_t k = 1; k < N; ++k)
            for (std::size_t j = k; j > 0 && m(order[j], order[j]) < m(order[j - 1], order[j - 1]); --j)
            {
                const std::size_t swapped = order[j];
                order[j] = order[j - 1];
                order[j - 1] = swapped;
            }
        for (std::size_t k = 0; k < N; ++k)
        {
            result.values[k] = m(order[k], order[k]);
            for (std::size_t i = 0; i < N; ++i)
                result.vectors(i, k) = v(i, order[k]);
        }
        return result;
    }

private:
    // One Jacobi rotation zeroing m(p, q): m ← Jᵀ·m·J, v ← v·J.
    static void rotate(base::MatN<N>& m, base::MatN<N>& v, std::size_t p, std::size_t q)
    {
        const double apq = m(p, q);
        if (apq == 0.0)
            return;
        // A term below the round-off of both diagonal terms is set to zero without a rotation (Numerical Recipes
        // §11.1: the test |a_pq| + |a_pp| == |a_pp|), which also ends the sweeps once the matrix is diagonal.
        if (std::abs(m(p, p)) + std::abs(apq) == std::abs(m(p, p)) &&
            std::abs(m(q, q)) + std::abs(apq) == std::abs(m(q, q)))
        {
            m(p, q) = 0.0;
            m(q, p) = 0.0;
            return;
        }
        const double theta = (m(q, q) - m(p, p)) / (2.0 * apq);
        // For a huge θ, t ≈ 1/(2θ) without squaring θ (Numerical Recipes §11.1).
        const double t = std::abs(theta) > HUGE_THETA
                             ? 1.0 / (2.0 * theta)
                             : (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < N; ++k)
        {
            const double mkp = m(k, p);
            const double mkq = m(k, q);
            m(k, p) = c * mkp - s * mkq;
            m(k, q) = s * mkp + c * mkq;
        }
        for (std::size_t k = 0; k < N; ++k)
        {
            const double mpk = m(p, k);
            const double mqk = m(q, k);
            m(p, k) = c * mpk - s * mqk;
            m(q, k) = s * mpk + c * mqk;
        }
        m(p, q) = 0.0;
        m(q, p) = 0.0;
        for (std::size_t k = 0; k < N; ++k)
        {
            const double vkp = v(k, p);
            const double vkq = v(k, q);
            v(k, p) = c * vkp - s * vkq;
            v(k, q) = s * vkp + c * vkq;
        }
    }

    // Above this |θ|, θ² would overflow or lose t entirely.
    static constexpr double HUGE_THETA = 1.0e150;
};

} // namespace oos::assembly
