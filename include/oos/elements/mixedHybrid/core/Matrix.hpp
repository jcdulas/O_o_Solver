#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The small dense matrices of one element of the mixed hybrid formulation (a few hundred rows at most): Gram
// matrices, the compliance of the moments, the trace matrices of the solids, element stiffness before assembly.
// These are working matrices of one element on one thread, so they are plain values, not budgeted (the large
// matrices of the solver are linalg::DenseMatrix).
//
// Storage is row-major, term (i, j) at i·columns + j. Every product sums its terms in a fixed order (the summation
// index increasing), so that a result has the same bits on every run and every thread count.
//
// Thread-safety: a plain value type; the free functions are pure.

using Vector = std::vector<double>;

class Matrix
{
public:
    Matrix() = default;
    // rows × columns, every term equal to `value`.
    Matrix(std::size_t rows, std::size_t columns, double value = 0.0);

    [[nodiscard]] static Matrix zeros(std::size_t rows, std::size_t columns) { return Matrix(rows, columns); }
    [[nodiscard]] static Matrix identity(std::size_t size);

    [[nodiscard]] std::size_t rows() const { return m_rows; }
    [[nodiscard]] std::size_t columns() const { return m_columns; }
    [[nodiscard]] bool empty() const { return m_values.empty(); }

    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const { return m_values[i * m_columns + j]; }
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) { return m_values[i * m_columns + j]; }

    // Row i as a contiguous range (row-major storage).
    [[nodiscard]] std::span<const double> rowSpan(std::size_t i) const
    {
        return {m_values.data() + i * m_columns, m_columns};
    }
    [[nodiscard]] std::span<double> rowSpan(std::size_t i) { return {m_values.data() + i * m_columns, m_columns}; }
    // Every term, row after row.
    [[nodiscard]] std::span<const double> values() const { return m_values; }

    [[nodiscard]] Vector row(std::size_t i) const;
    [[nodiscard]] Vector column(std::size_t j) const;
    void setRow(std::size_t i, std::span<const double> values);
    void setColumn(std::size_t j, std::span<const double> values);

    // The rows × columns block whose first term is (row0, column0).
    [[nodiscard]] Matrix block(std::size_t row0, std::size_t column0, std::size_t rows, std::size_t columns) const;
    // Writes `values` at (row0, column0) (the block must fit).
    void setBlock(std::size_t row0, std::size_t column0, const Matrix& values);

    [[nodiscard]] Matrix transposed() const;

    void scaleRow(std::size_t i, double factor);
    void scaleColumn(std::size_t j, double factor);

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double factor);

    friend Matrix operator+(Matrix a, const Matrix& b) { return a += b; }
    friend Matrix operator-(Matrix a, const Matrix& b) { return a -= b; }
    friend Matrix operator*(Matrix a, double factor) { return a *= factor; }
    friend Matrix operator*(double factor, Matrix a) { return a *= factor; }
    friend Matrix operator-(Matrix a) { return a *= -1.0; }
    friend bool operator==(const Matrix&, const Matrix&) = default;

private:
    std::size_t m_rows = 0;
    std::size_t m_columns = 0;
    std::vector<double> m_values;
};

// A·B.
[[nodiscard]] Matrix operator*(const Matrix& a, const Matrix& b);
// A·x.
[[nodiscard]] Vector operator*(const Matrix& a, std::span<const double> x);
// Aᵀ·B, without forming Aᵀ.
[[nodiscard]] Matrix transposeTimes(const Matrix& a, const Matrix& b);
// Aᵀ·x.
[[nodiscard]] Vector transposeTimes(const Matrix& a, std::span<const double> x);
// A·Bᵀ, without forming Bᵀ.
[[nodiscard]] Matrix timesTranspose(const Matrix& a, const Matrix& b);
// The lower triangle (diagonal included) of the Gram matrix YᵀY; the strict upper triangle is left at zero. Half the
// work of the full product, for a matrix that a Cholesky factorisation reads by its lower triangle.
[[nodiscard]] Matrix gramLower(const Matrix& y);
// The rows of m scaled by the weights: diag(w)·m (the weights of the quadrature points, one per row).
[[nodiscard]] Matrix weightedRows(const Matrix& m, std::span<const double> weights);
// (A + Aᵀ) / 2: removes the round-off asymmetry of a matrix that is symmetric in exact arithmetic.
[[nodiscard]] Matrix symmetricPart(const Matrix& a);
// max |a_ij| (zero for an empty matrix).
[[nodiscard]] double maxAbs(const Matrix& a);
// (Σ a_ij²)^½.
[[nodiscard]] double frobeniusNorm(const Matrix& a);
// Σ a_ii.
[[nodiscard]] double trace(const Matrix& a);

// Σ a_i b_i.
[[nodiscard]] double dot(std::span<const double> a, std::span<const double> b);
// (Σ a_i²)^½.
[[nodiscard]] double euclideanNorm(std::span<const double> a);

} // namespace oos::elements::mixedHybrid
