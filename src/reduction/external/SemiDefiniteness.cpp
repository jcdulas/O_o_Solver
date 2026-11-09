#include "oos/reduction/external/SemiDefiniteness.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::reduction
{

namespace
{

// Swaps rows and columns k and p (k < p) of a symmetric matrix held in its lower triangle (a(i, j), i ≥ j): the
// symmetric permutation of the pivot (LAPACK's xSYTF2 swaps the same pieces).
void swapSymmetric(linalg::DenseMatrix& a, std::size_t k, std::size_t p)
{
    const std::size_t n = a.rows();
    std::swap(a(k, k), a(p, p));
    for (std::size_t j = 0; j < k; ++j)
        std::swap(a(k, j), a(p, j));
    for (std::size_t i = p + 1; i < n; ++i)
        std::swap(a(i, k), a(i, p));
    for (std::size_t m = k + 1; m < p; ++m)
        std::swap(a(m, k), a(p, m));
}

} // namespace

SemiDefiniteResult testSemiDefinite(linalg::ConstDenseMatrixView symmetric, double tolerance,
                                    base::MemoryBudget& budget)
{
    const std::size_t n = symmetric.rows;
    if (symmetric.columns != n)
        throw std::invalid_argument(base::concatenate({"testSemiDefinite: expected a square matrix, found ",
                                                       std::to_string(n), " x ", std::to_string(symmetric.columns)}));
    SemiDefiniteResult result;
    if (n == 0)
        return result;
    linalg::DenseMatrix a(budget, n, n, "semi-definiteness test");
    double scale = 0.0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
        {
            a(i, j) = symmetric(i, j);
            scale = std::max(scale, std::abs(a(i, j)));
        }
    if (scale == 0.0)
        return result;                 // the zero matrix: semi-definite, rank 0
    std::vector<std::size_t> rowOf(n); // the original row at each position
    std::iota(rowOf.begin(), rowOf.end(), std::size_t{0});
    const double threshold = tolerance * scale;
    result.smallestPivot = 1.0;
    for (std::size_t k = 0; k < n; ++k)
    {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (a(i, i) > a(p, p))
                p = i;
        if (a(p, p) <= threshold)
        {
            // The remainder S_k must be negligible (the class comment).
            std::size_t worst = k;
            double offDiagonal = 0.0;
            for (std::size_t j = k; j < n; ++j)
            {
                if (a(j, j) < a(worst, worst))
                    worst = j;
                for (std::size_t i = j + 1; i < n; ++i)
                    offDiagonal = std::max(offDiagonal, std::abs(a(i, j)));
            }
            // −|b| of a remaining coupling (a +0 when there is none, so that a zero remainder reports 0).
            const double coupling = offDiagonal > 0.0 ? -offDiagonal : 0.0;
            const double pivot = std::min(a(worst, worst), coupling);
            if (pivot < result.smallestPivot * scale)
            {
                result.smallestPivot = pivot / scale;
                result.row = rowOf[worst];
            }
            result.isSemiDefinite = a(worst, worst) >= -threshold && offDiagonal <= threshold;
            return result;
        }
        if (p != k)
        {
            swapSymmetric(a, k, p);
            std::swap(rowOf[k], rowOf[p]);
        }
        const double pivot = a(k, k);
        ++result.rank;
        if (pivot / scale < result.smallestPivot)
        {
            result.smallestPivot = pivot / scale;
            result.row = rowOf[k];
        }
        // S_{k+1} = S_k(k+1:, k+1:) − l·d·lᵀ, l = S_k(k+1:, k)/d: the lower triangle, column by column.
        for (std::size_t j = k + 1; j < n; ++j)
        {
            const double factor = a(j, k) / pivot;
            if (factor == 0.0)
                continue;
            for (std::size_t i = j; i < n; ++i)
                a(i, j) -= a(i, k) * factor;
        }
    }
    return result;
}

} // namespace oos::reduction
