#include "oos/elements/mixedHybrid/core/Matrix.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements::mixedHybrid
{

Matrix::Matrix(std::size_t rows, std::size_t columns, double value)
    : m_rows(rows), m_columns(columns), m_values(rows * columns, value)
{
}

Matrix Matrix::identity(std::size_t size)
{
    Matrix m(size, size);
    for (std::size_t i = 0; i < size; ++i)
        m(i, i) = 1.0;
    return m;
}

Vector Matrix::row(std::size_t i) const
{
    const std::span<const double> values = rowSpan(i);
    return Vector(values.begin(), values.end());
}

Vector Matrix::column(std::size_t j) const
{
    Vector out(m_rows);
    for (std::size_t i = 0; i < m_rows; ++i)
        out[i] = (*this)(i, j);
    return out;
}

void Matrix::setRow(std::size_t i, std::span<const double> values)
{
    const std::span<double> target = rowSpan(i);
    for (std::size_t j = 0; j < m_columns; ++j)
        target[j] = values[j];
}

void Matrix::setColumn(std::size_t j, std::span<const double> values)
{
    for (std::size_t i = 0; i < m_rows; ++i)
        (*this)(i, j) = values[i];
}

Matrix Matrix::block(std::size_t row0, std::size_t column0, std::size_t rows, std::size_t columns) const
{
    Matrix out(rows, columns);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
            out(i, j) = (*this)(row0 + i, column0 + j);
    return out;
}

void Matrix::setBlock(std::size_t row0, std::size_t column0, const Matrix& values)
{
    for (std::size_t i = 0; i < values.rows(); ++i)
        for (std::size_t j = 0; j < values.columns(); ++j)
            (*this)(row0 + i, column0 + j) = values(i, j);
}

Matrix Matrix::transposed() const
{
    Matrix out(m_columns, m_rows);
    for (std::size_t i = 0; i < m_rows; ++i)
        for (std::size_t j = 0; j < m_columns; ++j)
            out(j, i) = (*this)(i, j);
    return out;
}

void Matrix::scaleRow(std::size_t i, double factor)
{
    for (double& value : rowSpan(i))
        value *= factor;
}

void Matrix::scaleColumn(std::size_t j, double factor)
{
    for (std::size_t i = 0; i < m_rows; ++i)
        (*this)(i, j) *= factor;
}

Matrix& Matrix::operator+=(const Matrix& other)
{
    for (std::size_t k = 0; k < m_values.size(); ++k)
        m_values[k] += other.m_values[k];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other)
{
    for (std::size_t k = 0; k < m_values.size(); ++k)
        m_values[k] -= other.m_values[k];
    return *this;
}

Matrix& Matrix::operator*=(double factor)
{
    for (double& value : m_values)
        value *= factor;
    return *this;
}

// (A·B)(i, :) = Σ_k a_ik B(k, :): the rows of B are read contiguously, and each term sums for k increasing.
Matrix operator*(const Matrix& a, const Matrix& b)
{
    return timesLeadingColumns(a, b, b.columns());
}

Matrix timesLeadingColumns(const Matrix& a, const Matrix& b, std::size_t columnCount)
{
    Matrix out(a.rows(), b.columns());
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        const std::span<double> target = out.rowSpan(i);
        for (std::size_t k = 0; k < a.columns(); ++k)
        {
            const double factor = a(i, k);
            if (factor == 0.0)
                continue;
            const std::span<const double> source = b.rowSpan(k);
            for (std::size_t j = 0; j < columnCount; ++j)
                target[j] += factor * source[j];
        }
    }
    return out;
}

Vector operator*(const Matrix& a, std::span<const double> x)
{
    Vector out(a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
        out[i] = dot(a.rowSpan(i), x);
    return out;
}

// (AᵀB)(i, :) = Σ_k a_ki B(k, :), summed for k increasing.
Matrix transposeTimes(const Matrix& a, const Matrix& b)
{
    return transposeTimesLeadingColumns(a, b, b.columns());
}

Matrix transposeTimesLeadingColumns(const Matrix& a, const Matrix& b, std::size_t columnCount)
{
    Matrix out(a.columns(), b.columns());
    for (std::size_t k = 0; k < a.rows(); ++k)
    {
        const std::span<const double> source = b.rowSpan(k);
        for (std::size_t i = 0; i < a.columns(); ++i)
        {
            const double factor = a(k, i);
            if (factor == 0.0)
                continue;
            const std::span<double> target = out.rowSpan(i);
            for (std::size_t j = 0; j < columnCount; ++j)
                target[j] += factor * source[j];
        }
    }
    return out;
}

// (AᵀA)(i, j) = Σ_k a_ki a_kj for j >= i, summed for k increasing as transposeTimes does; then (j, i) = (i, j).
Matrix transposeTimesSelf(const Matrix& a)
{
    const std::size_t n = a.columns();
    Matrix out(n, n);
    for (std::size_t k = 0; k < a.rows(); ++k)
    {
        const std::span<const double> source = a.rowSpan(k);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double factor = source[i];
            if (factor == 0.0)
                continue;
            const std::span<double> target = out.rowSpan(i);
            for (std::size_t j = i; j < n; ++j)
                target[j] += factor * source[j];
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            out(i, j) = out(j, i);
    return out;
}

Vector transposeTimes(const Matrix& a, std::span<const double> x)
{
    Vector out(a.columns(), 0.0);
    for (std::size_t k = 0; k < a.rows(); ++k)
    {
        const std::span<const double> source = a.rowSpan(k);
        for (std::size_t i = 0; i < a.columns(); ++i)
            out[i] += source[i] * x[k];
    }
    return out;
}

// (ABᵀ)(i, j) = row i of A · row j of B.
Matrix timesTranspose(const Matrix& a, const Matrix& b)
{
    Matrix out(a.rows(), b.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < b.rows(); ++j)
            out(i, j) = dot(a.rowSpan(i), b.rowSpan(j));
    return out;
}

Matrix gramLower(const Matrix& y)
{
    const std::size_t n = y.columns();
    Matrix out(n, n);
    for (std::size_t k = 0; k < y.rows(); ++k)
    {
        const std::span<const double> source = y.rowSpan(k);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double factor = source[i];
            if (factor == 0.0)
                continue;
            const std::span<double> target = out.rowSpan(i);
            for (std::size_t j = 0; j <= i; ++j)
                target[j] += factor * source[j];
        }
    }
    return out;
}

Matrix weightedRows(const Matrix& m, std::span<const double> weights)
{
    Matrix out = m;
    for (std::size_t i = 0; i < m.rows(); ++i)
        out.scaleRow(i, weights[i]);
    return out;
}

Matrix symmetricPart(const Matrix& a)
{
    Matrix out(a.rows(), a.columns());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.columns(); ++j)
            out(i, j) = 0.5 * (a(i, j) + a(j, i));
    return out;
}

double maxAbs(const Matrix& a)
{
    double largest = 0.0;
    for (double value : a.values())
        largest = std::max(largest, std::fabs(value));
    return largest;
}

double frobeniusNorm(const Matrix& a)
{
    return euclideanNorm(a.values());
}

double trace(const Matrix& a)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < std::min(a.rows(), a.columns()); ++i)
        sum += a(i, i);
    return sum;
}

double dot(std::span<const double> a, std::span<const double> b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

double euclideanNorm(std::span<const double> a)
{
    return std::sqrt(dot(a, a));
}

} // namespace oos::elements::mixedHybrid
