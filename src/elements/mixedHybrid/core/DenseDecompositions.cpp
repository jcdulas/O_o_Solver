#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"

#include "oos/base/errors/InternalError.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The floor of the largest term of a row or column in the equilibration: a zero row stays zero (and the matrix is
// then singular) instead of being divided by zero.
constexpr double EQUILIBRATION_FLOOR = 1e-300;

// x ← x − factor · y over a row.
void subtractScaledRow(std::span<double> x, double factor, std::span<const double> y)
{
    if (factor == 0.0)
        return;
    for (std::size_t j = 0; j < x.size(); ++j)
        x[j] -= factor * y[j];
}

void divideRow(std::span<double> x, double divisor)
{
    for (double& value : x)
        value /= divisor;
}

// b ← S b, the rows scaled.
Matrix scaleRows(Matrix b, const Vector& scales)
{
    for (std::size_t i = 0; i < b.rows(); ++i)
        b.scaleRow(i, scales[i]);
    return b;
}

} // namespace

// ---- Cholesky -------------------------------------------------------------------------------------------------

std::optional<Matrix> choleskyLower(const Matrix& a)
{
    OOS_ASSERT(a.rows() == a.columns());
    const std::size_t n = a.rows();
    Matrix lower(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::span<const double> rowI = lower.rowSpan(i);
        for (std::size_t j = 0; j <= i; ++j)
        {
            const std::span<const double> rowJ = lower.rowSpan(j);
            double s = a(i, j);
            for (std::size_t k = 0; k < j; ++k)
                s -= rowI[k] * rowJ[k];
            if (i == j)
            {
                // The negated test also refuses a NaN pivot.
                if (!(s > 0.0))
                    return std::nullopt;
                lower(i, i) = std::sqrt(s);
            }
            else
            {
                lower(i, j) = s / lower(j, j);
            }
        }
    }
    return lower;
}

// Row i of X = L⁻¹ B: x_i = (b_i − Σ_{k<i} l_ik x_k) / l_ii, the rows of B updated whole.
void solveLower(const Matrix& lower, Matrix& b)
{
    const std::size_t n = lower.rows();
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::span<double> target = b.rowSpan(i);
        for (std::size_t k = 0; k < i; ++k)
            subtractScaledRow(target, lower(i, k), b.rowSpan(k));
        divideRow(target, lower(i, i));
    }
}

void solveLower(const Matrix& lower, Vector& b)
{
    const std::size_t n = lower.rows();
    for (std::size_t i = 0; i < n; ++i)
    {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k)
            s -= lower(i, k) * b[k];
        b[i] = s / lower(i, i);
    }
}

// Row i of X = L⁻ᵀ B, from the last: x_i = (b_i − Σ_{k>i} l_ki x_k) / l_ii.
void solveLowerTransposed(const Matrix& lower, Matrix& b)
{
    const std::size_t n = lower.rows();
    for (std::size_t i = n; i-- > 0;)
    {
        const std::span<double> target = b.rowSpan(i);
        for (std::size_t k = i + 1; k < n; ++k)
            subtractScaledRow(target, lower(k, i), b.rowSpan(k));
        divideRow(target, lower(i, i));
    }
}

void solveLowerTransposed(const Matrix& lower, Vector& b)
{
    const std::size_t n = lower.rows();
    for (std::size_t i = n; i-- > 0;)
    {
        double s = b[i];
        for (std::size_t k = i + 1; k < n; ++k)
            s -= lower(k, i) * b[k];
        b[i] = s / lower(i, i);
    }
}

std::optional<Cholesky> Cholesky::factor(const Matrix& a)
{
    std::optional<Matrix> lower = choleskyLower(a);
    if (!lower)
        return std::nullopt;
    return Cholesky(std::move(*lower));
}

Vector Cholesky::solve(Vector b) const
{
    solveLower(m_lower, b);
    solveLowerTransposed(m_lower, b);
    return b;
}

Matrix Cholesky::solve(Matrix b) const
{
    solveLower(m_lower, b);
    solveLowerTransposed(m_lower, b);
    return b;
}

Matrix Cholesky::inverse() const
{
    Matrix x = Matrix::identity(size());
    solveLower(m_lower, x);
    return transposeTimes(x, x);
}

// ---- Cholesky after equilibration -----------------------------------------------------------------------------

std::optional<EquilibratedCholesky> EquilibratedCholesky::factor(const Matrix& a)
{
    OOS_ASSERT(a.rows() == a.columns());
    const std::size_t n = a.rows();
    Vector scales(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!(a(i, i) > 0.0))
            return std::nullopt;
        scales[i] = 1.0 / std::sqrt(a(i, i));
    }
    Matrix scaled(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            scaled(i, j) = a(i, j) * scales[i] * scales[j];
    std::optional<Matrix> lower = choleskyLower(scaled);
    if (!lower)
        return std::nullopt;
    return EquilibratedCholesky(std::move(scales), std::move(*lower));
}

Matrix EquilibratedCholesky::halfSolve(Matrix b) const
{
    b = scaleRows(std::move(b), m_scales);
    solveLower(m_lower, b);
    return b;
}

Matrix EquilibratedCholesky::solve(Matrix b) const
{
    b = halfSolve(std::move(b));
    solveLowerTransposed(m_lower, b);
    return scaleRows(std::move(b), m_scales);
}

Vector EquilibratedCholesky::solve(Vector b) const
{
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] *= m_scales[i];
    solveLower(m_lower, b);
    solveLowerTransposed(m_lower, b);
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] *= m_scales[i];
    return b;
}

// A⁻¹ = S XᵀX S with X = L⁻¹: the halfSolve of the identity is X S.
Matrix EquilibratedCholesky::inverse() const
{
    const Matrix x = halfSolve(Matrix::identity(size()));
    return transposeTimes(x, x);
}

// ---- LU with partial pivoting ---------------------------------------------------------------------------------

LuDecomposition::LuDecomposition(const Matrix& a) : m_factors(a), m_rowOf(a.rows())
{
    OOS_ASSERT(a.rows() == a.columns());
    const std::size_t n = a.rows();
    for (std::size_t k = 0; k < n; ++k)
        m_rowOf[k] = k;
    for (std::size_t k = 0; k < n; ++k)
    {
        // The pivot: the largest remaining term of column k (the first of equal ones).
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::fabs(m_factors(i, k)) > std::fabs(m_factors(pivot, k)))
                pivot = i;
        if (m_factors(pivot, k) == 0.0)
        {
            m_singular = true;
            continue;
        }
        if (pivot != k)
        {
            for (std::size_t j = 0; j < n; ++j)
                std::swap(m_factors(k, j), m_factors(pivot, j));
            std::swap(m_rowOf[k], m_rowOf[pivot]);
            m_permutationSign = -m_permutationSign;
        }
        const double diagonal = m_factors(k, k);
        for (std::size_t i = k + 1; i < n; ++i)
        {
            const double multiplier = m_factors(i, k) / diagonal;
            m_factors(i, k) = multiplier;
            if (multiplier == 0.0)
                continue;
            for (std::size_t j = k + 1; j < n; ++j)
                m_factors(i, j) -= multiplier * m_factors(k, j);
        }
    }
}

double LuDecomposition::determinant() const
{
    double product = m_permutationSign;
    for (std::size_t k = 0; k < size(); ++k)
        product *= m_factors(k, k);
    return product;
}

// L y = P b (unit diagonal), then U x = y.
void LuDecomposition::substitute(std::span<double> x) const
{
    const std::size_t n = size();
    for (std::size_t i = 0; i < n; ++i)
    {
        double s = x[i];
        for (std::size_t k = 0; k < i; ++k)
            s -= m_factors(i, k) * x[k];
        x[i] = s;
    }
    for (std::size_t i = n; i-- > 0;)
    {
        double s = x[i];
        for (std::size_t k = i + 1; k < n; ++k)
            s -= m_factors(i, k) * x[k];
        x[i] = s / m_factors(i, i);
    }
}

std::optional<Vector> LuDecomposition::solve(const Vector& b) const
{
    if (m_singular)
        return std::nullopt;
    Vector x(size());
    for (std::size_t k = 0; k < size(); ++k)
        x[k] = b[m_rowOf[k]];
    substitute(x);
    return x;
}

std::optional<Matrix> LuDecomposition::solve(const Matrix& b) const
{
    if (m_singular)
        return std::nullopt;
    Matrix x(b.rows(), b.columns());
    Vector column(size());
    for (std::size_t j = 0; j < b.columns(); ++j)
    {
        for (std::size_t k = 0; k < size(); ++k)
            column[k] = b(m_rowOf[k], j);
        substitute(column);
        x.setColumn(j, column);
    }
    return x;
}

std::optional<Matrix> LuDecomposition::inverse() const
{
    return solve(Matrix::identity(size()));
}

std::optional<Matrix> inverse(const Matrix& a)
{
    return LuDecomposition(a).inverse();
}

double determinant(const Matrix& a)
{
    return LuDecomposition(a).determinant();
}

std::optional<Matrix> equilibratedInverse(const Matrix& m)
{
    OOS_ASSERT(m.rows() == m.columns());
    const std::size_t n = m.rows();
    Vector rowScales(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        double largest = 0.0;
        for (double value : m.rowSpan(i))
            largest = std::max(largest, std::fabs(value));
        rowScales[i] = 1.0 / std::max(largest, EQUILIBRATION_FLOOR);
    }
    Matrix scaled = scaleRows(m, rowScales);
    Vector columnScales(n);
    for (std::size_t j = 0; j < n; ++j)
    {
        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            largest = std::max(largest, std::fabs(scaled(i, j)));
        columnScales[j] = 1.0 / std::max(largest, EQUILIBRATION_FLOOR);
    }
    for (std::size_t j = 0; j < n; ++j)
        scaled.scaleColumn(j, columnScales[j]);
    std::optional<Matrix> scaledInverse = LuDecomposition(scaled).inverse();
    if (!scaledInverse)
        return std::nullopt;
    // m = R⁻¹ M_s C⁻¹, so m⁻¹ = C M_s⁻¹ R.
    Matrix out(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            out(i, j) = columnScales[i] * (*scaledInverse)(i, j) * rowScales[j];
    return out;
}

} // namespace oos::elements::mixedHybrid
