#include "oos/linalg/dense/DenseLu.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{

DenseLu::DenseLu(DenseMatrix a, base::MemoryBudget& budget)
    : m_factor(std::move(a)), m_pivots(budget, m_factor.rows(), "LU pivots")
{
    const std::size_t n = m_factor.rows();
    if (m_factor.columns() != n)
        throw std::invalid_argument(base::concatenate({"DenseLu: expected a square matrix, found ", std::to_string(n),
                                                       " x ", std::to_string(m_factor.columns())}));
    DenseMatrix& f = m_factor;
    double smallest = 0.0;
    double largest = 0.0;
    for (std::size_t k = 0; k < n; ++k)
    {
        // The pivot: the largest magnitude in column k at or below the diagonal (the first one on a tie).
        std::size_t pivotRow = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(f(i, k)) > std::abs(f(pivotRow, k)))
                pivotRow = i;
        m_pivots[k] = static_cast<std::int64_t>(pivotRow);
        if (pivotRow != k)
            for (std::size_t j = 0; j < n; ++j)
                std::swap(f(k, j), f(pivotRow, j));
        const double pivot = f(k, k);
        const double magnitude = std::abs(pivot);
        if (!(magnitude > 0.0) || !std::isfinite(magnitude))
        {
            if (m_singularColumn < 0)
                m_singularColumn = static_cast<std::int64_t>(k);
            continue;
        }
        smallest = k == 0 ? magnitude : std::min(smallest, magnitude);
        largest = std::max(largest, magnitude);
        for (std::size_t i = k + 1; i < n; ++i)
            f(i, k) /= pivot;
        // a_ij ← a_ij − l_ik·u_kj, column by column (the columns are contiguous).
        for (std::size_t j = k + 1; j < n; ++j)
        {
            const double ukj = f(k, j);
            if (ukj == 0.0)
                continue;
            for (std::size_t i = k + 1; i < n; ++i)
                f(i, j) -= f(i, k) * ukj;
        }
    }
    m_pivotRatio = largest > 0.0 && m_singularColumn < 0 ? smallest / largest : 0.0;
}

void DenseLu::solve(DenseMatrixView rhs) const
{
    const std::size_t n = order();
    if (rhs.rows != n)
        throw std::invalid_argument(base::concatenate({"DenseLu::solve: expected right-hand sides of ",
                                                       std::to_string(n), " rows, found ", std::to_string(rhs.rows)}));
    if (isSingular())
        throw std::invalid_argument(
            base::concatenate({"DenseLu::solve: the matrix is singular at column ", std::to_string(m_singularColumn),
                               "; check isSingular() before solving"}));
    const DenseMatrix& f = m_factor;
    for (std::size_t c = 0; c < rhs.columns; ++c)
    {
        // P·b, then L·y = P·b (forward), then U·x = y (backward).
        for (std::size_t k = 0; k < n; ++k)
        {
            const auto pivotRow = static_cast<std::size_t>(m_pivots[k]);
            if (pivotRow != k)
                std::swap(rhs(k, c), rhs(pivotRow, c));
        }
        for (std::size_t k = 0; k < n; ++k)
        {
            const double yk = rhs(k, c);
            if (yk == 0.0)
                continue;
            for (std::size_t i = k + 1; i < n; ++i)
                rhs(i, c) -= f(i, k) * yk;
        }
        for (std::size_t k = n; k-- > 0;)
        {
            rhs(k, c) /= f(k, k);
            const double xk = rhs(k, c);
            if (xk == 0.0)
                continue;
            for (std::size_t i = 0; i < k; ++i)
                rhs(i, c) -= f(i, k) * xk;
        }
    }
}

} // namespace oos::linalg
