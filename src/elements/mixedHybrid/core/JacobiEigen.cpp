#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// A sweep visits every pair (p, q) once; the methods converge quadratically once the off-diagonal part is small, so
// that a few sweeps suffice in practice. The bound only stops a matrix of non-finite terms.
constexpr int MAX_SWEEPS = 100;
// Convergence of jacobiEigen: Σ_{i<j} a_ij² ≤ 1e-32 Σ_i a_ii², the off-diagonal part at the round-off of the diagonal.
constexpr double EIGEN_CONVERGENCE = 1e-32;
// Keeps the convergence test meaningful for a zero matrix.
constexpr double TINY_DIAGONAL = 1e-300;
// Convergence of the one-sided method: |a_p · a_q| ≤ 1e-15 |a_p| |a_q| for every pair of columns.
constexpr double ORTHOGONALITY_TOLERANCE = 1e-15;
// minimumNormInverse: the eigenvalues of AᵀA below 1e-12 of the largest are zero (the exact rank deficiency).
constexpr double MINIMUM_NORM_RELATIVE_THRESHOLD = 1e-12;

// The tangent t of the rotation angle that zeroes the coupling of a 2 × 2 symmetric block, from
// ratio = (d_q − d_p) / (2 coupling): the smaller root of t² + 2 ratio t − 1 = 0 (the rotation of at most π/4), t = 1
// when the two diagonal terms are equal.
double rotationTangent(double ratio)
{
    if (ratio == 0.0)
        return 1.0;
    return std::copysign(1.0, ratio) / (std::fabs(ratio) + std::sqrt(ratio * ratio + 1.0));
}

// Columns p and q of m ← (c m_p − s m_q, s m_p + c m_q): m J with J the plane rotation.
void rotateColumns(Matrix& m, std::size_t p, std::size_t q, double c, double s)
{
    for (std::size_t k = 0; k < m.rows(); ++k)
    {
        const double mkp = m(k, p);
        const double mkq = m(k, q);
        m(k, p) = c * mkp - s * mkq;
        m(k, q) = s * mkp + c * mkq;
    }
}

// Rows p and q of m ← (c m_p − s m_q, s m_p + c m_q): Jᵀ m.
void rotateRows(Matrix& m, std::size_t p, std::size_t q, double c, double s)
{
    for (std::size_t k = 0; k < m.columns(); ++k)
    {
        const double mpk = m(p, k);
        const double mqk = m(q, k);
        m(p, k) = c * mpk - s * mqk;
        m(q, k) = s * mpk + c * mqk;
    }
}

// Σ_{i<j} a_ij² ≤ 1e-32 Σ_i a_ii².
bool isNearlyDiagonal(const Matrix& a)
{
    double offDiagonal = 0.0;
    double diagonal = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        diagonal += a(i, i) * a(i, i);
        for (std::size_t j = i + 1; j < a.columns(); ++j)
            offDiagonal += a(i, j) * a(i, j);
    }
    return offDiagonal <= EIGEN_CONVERGENCE * std::max(diagonal, TINY_DIAGONAL);
}

// The products of columns p and q of m: |m_p|², |m_q|², m_p · m_q.
struct ColumnProducts
{
    double pp = 0.0;
    double qq = 0.0;
    double pq = 0.0;
};

ColumnProducts columnProducts(const Matrix& m, std::size_t p, std::size_t q)
{
    ColumnProducts products;
    for (std::size_t k = 0; k < m.rows(); ++k)
    {
        products.pp += m(k, p) * m(k, p);
        products.qq += m(k, q) * m(k, q);
        products.pq += m(k, p) * m(k, q);
    }
    return products;
}

// One sweep of one-sided rotations over the column pairs; false when every pair was already orthogonal.
bool orthogonalisationSweep(Matrix& u, Matrix& v)
{
    bool rotated = false;
    for (std::size_t p = 0; p < u.columns(); ++p)
    {
        for (std::size_t q = p + 1; q < u.columns(); ++q)
        {
            const ColumnProducts g = columnProducts(u, p, q);
            if (g.pq == 0.0 || std::fabs(g.pq) <= ORTHOGONALITY_TOLERANCE * std::sqrt(g.pp * g.qq))
                continue;
            rotated = true;
            const double t = rotationTangent((g.qq - g.pp) / (2.0 * g.pq));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;
            rotateColumns(u, p, q, c, s);
            rotateColumns(v, p, q, c, s);
        }
    }
    return rotated;
}

} // namespace

SymmetricEigen jacobiEigen(const Matrix& m)
{
    const std::size_t n = m.rows();
    Matrix a = m;
    Matrix v = Matrix::identity(n);
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep)
    {
        if (isNearlyDiagonal(a))
            break;
        for (std::size_t p = 0; p < n; ++p)
        {
            for (std::size_t q = p + 1; q < n; ++q)
            {
                const double apq = a(p, q);
                if (apq == 0.0)
                    continue;
                const double t = rotationTangent((a(q, q) - a(p, p)) / (2.0 * apq));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                rotateColumns(a, p, q, c, s);
                rotateRows(a, p, q, c, s);
                rotateColumns(v, p, q, c, s);
            }
        }
    }
    SymmetricEigen eigen;
    eigen.values.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        eigen.values[i] = a(i, i);
    eigen.vectors = std::move(v);
    return eigen;
}

Matrix minimumNormInverse(const Matrix& a)
{
    const SymmetricEigen eigen = jacobiEigen(symmetricPart(transposeTimes(a, a)));
    double largest = 0.0;
    for (double value : eigen.values)
        largest = std::max(largest, value);
    const std::size_t n = a.columns();
    Matrix gramInverse(n, n);
    for (std::size_t k = 0; k < n; ++k)
    {
        const double lambda = eigen.values[k];
        if (!(lambda > MINIMUM_NORM_RELATIVE_THRESHOLD * largest))
            continue;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                gramInverse(i, j) += eigen.vectors(i, k) * eigen.vectors(j, k) / lambda;
    }
    return timesTranspose(gramInverse, a);
}

SingularValueDecomposition jacobiSingularValueDecomposition(const Matrix& a)
{
    // A wide matrix: Aᵀ = U Σ Vᵀ gives A = V Σ Uᵀ.
    if (a.rows() < a.columns())
    {
        SingularValueDecomposition transposed = jacobiSingularValueDecomposition(a.transposed());
        return {std::move(transposed.right), std::move(transposed.values), std::move(transposed.left)};
    }
    Matrix u = a;
    Matrix v = Matrix::identity(a.columns());
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep)
        if (!orthogonalisationSweep(u, v))
            break;
    // The columns of A V are now orthogonal: their norms are the singular values, their directions the left vectors.
    Vector values(a.columns());
    for (std::size_t j = 0; j < a.columns(); ++j)
    {
        values[j] = euclideanNorm(u.column(j));
        if (values[j] > 0.0)
            u.scaleColumn(j, 1.0 / values[j]);
    }
    return {std::move(u), std::move(values), std::move(v)};
}

Matrix pseudoInverse(const Matrix& a, double singularValueTolerance)
{
    const SingularValueDecomposition svd = jacobiSingularValueDecomposition(a);
    Matrix out(a.columns(), a.rows());
    for (std::size_t k = 0; k < svd.values.size(); ++k)
    {
        const double sigma = svd.values[k];
        if (!(sigma > singularValueTolerance))
            continue;
        for (std::size_t i = 0; i < a.columns(); ++i)
            for (std::size_t j = 0; j < a.rows(); ++j)
                out(i, j) += svd.right(i, k) * svd.left(j, k) / sigma;
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
