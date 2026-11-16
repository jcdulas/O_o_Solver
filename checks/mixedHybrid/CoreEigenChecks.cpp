#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;

namespace
{

// The plane rotation of angle `angle` in the plane (p, q), as an n × n matrix.
mh::Matrix rotation(std::size_t n, std::size_t p, std::size_t q, double angle)
{
    mh::Matrix r = mh::Matrix::identity(n);
    r(p, p) = std::cos(angle);
    r(q, q) = std::cos(angle);
    r(p, q) = std::sin(angle);
    r(q, p) = -std::sin(angle);
    return r;
}

// V diag(values) Vᵀ.
mh::Matrix reconstruct(const mh::SymmetricEigen& eigen)
{
    mh::Matrix scaled = eigen.vectors;
    for (std::size_t k = 0; k < eigen.values.size(); ++k)
        scaled.scaleColumn(k, eigen.values[k]);
    return mh::timesTranspose(scaled, eigen.vectors);
}

double orthogonalityError(const mh::Matrix& v)
{
    return mh::maxAbs(mh::transposeTimes(v, v) - mh::Matrix::identity(v.columns()));
}

// The four conditions of Penrose that define the pseudo-inverse X of A: AXA = A, XAX = X, AX and XA symmetric.
double penroseError(const mh::Matrix& a, const mh::Matrix& x)
{
    const mh::Matrix ax = a * x;
    const mh::Matrix xa = x * a;
    double worst = mh::maxAbs(ax * a - a) / mh::maxAbs(a);
    worst = std::max(worst, mh::maxAbs(xa * x - x) / mh::maxAbs(x));
    worst = std::max(worst, mh::maxAbs(ax - ax.transposed()));
    worst = std::max(worst, mh::maxAbs(xa - xa.transposed()));
    return worst;
}

// A rank-3 matrix of size rows × columns (the product of a rows × 3 and a 3 × columns matrix).
mh::Matrix rankThree(std::size_t rows, std::size_t columns)
{
    mh::Matrix left(rows, 3);
    mh::Matrix right(3, columns);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            left(i, k) = std::sin(0.3 + 1.1 * static_cast<double>(i * (k + 1)) + 2.3 * static_cast<double>(k));
    for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t j = 0; j < columns; ++j)
        {
            const double frequency = 0.7 * static_cast<double>((k + 1) * (k + 1));
            right(k, j) = std::cos(0.2 + frequency * static_cast<double>(j) + 1.9 * static_cast<double>(k));
        }
    return left * right;
}

} // namespace

CHECK_CASE("core: Jacobi eigen on a nearly diagonal matrix with a double eigenvalue")
{
    // The case where the reference found a general-purpose routine 4e-2 off: diag(2, 1, 2, 3, 1 + 1e-9) plus a
    // symmetric coupling of 1e-3.
    const std::size_t n = 5;
    const double diagonal[5] = {2.0, 1.0, 2.0, 3.0, 1.0 + 1e-9};
    mh::Matrix a(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a(i, j) = i == j ? diagonal[i] : 1e-3 * std::sin(static_cast<double>(i + j));
    const mh::SymmetricEigen eigen = mh::jacobiEigen(a);
    runner.expect(orthogonalityError(eigen.vectors) < 1e-14, "orthonormal eigenvectors");
    runner.expect(mh::maxAbs(reconstruct(eigen) - a) < 1e-13 * mh::maxAbs(a), "V diag V^T = A");
    for (std::size_t k = 0; k < n; ++k)
    {
        const mh::Vector v = eigen.vectors.column(k);
        const mh::Vector av = a * v;
        double residual = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            residual = std::max(residual, std::fabs(av[i] - eigen.values[k] * v[i]));
        runner.expect(residual < 1e-14, "A v = lambda v, pair " + std::to_string(k));
    }

    // An exactly double eigenvalue hidden by a small rotation: Q diag(1, 1, 2, 3) Q^T.
    mh::Matrix q =
        rotation(4, 0, 1, 1e-3) * rotation(4, 1, 2, 2e-3) * rotation(4, 0, 3, -1.5e-3) * rotation(4, 2, 3, 0.7);
    mh::Matrix lambda(4, 4);
    lambda(0, 0) = 1.0;
    lambda(1, 1) = 1.0;
    lambda(2, 2) = 2.0;
    lambda(3, 3) = 3.0;
    const mh::Matrix b = mh::symmetricPart(mh::timesTranspose(q * lambda, q));
    const mh::SymmetricEigen pairs = mh::jacobiEigen(b);
    std::vector<double> sorted = pairs.values;
    std::sort(sorted.begin(), sorted.end());
    const double expected[4] = {1.0, 1.0, 2.0, 3.0};
    for (std::size_t k = 0; k < 4; ++k)
        runner.expectNear(sorted[k], expected[k], 1e-14, "eigenvalue " + std::to_string(k));
    runner.expect(orthogonalityError(pairs.vectors) < 1e-14, "orthonormal eigenvectors (double eigenvalue)");
    runner.expect(mh::maxAbs(reconstruct(pairs) - b) < 1e-13, "reconstruction (double eigenvalue)");

    // A diagonal matrix needs no rotation; a zero matrix gives zero eigenvalues.
    const mh::SymmetricEigen diagonalPairs = mh::jacobiEigen(lambda);
    runner.expect(diagonalPairs.vectors == mh::Matrix::identity(4), "diagonal: identity eigenvectors");
    const mh::SymmetricEigen zero = mh::jacobiEigen(mh::Matrix(3, 3));
    runner.expect(zero.values == mh::Vector(3, 0.0), "zero matrix");
}

CHECK_CASE("core: Jacobi eigen on a larger symmetric matrix")
{
    const std::size_t n = 40;
    mh::Matrix a(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value =
                std::sin(1.0 + static_cast<double>(i * 7 + j * 3)) / (1.0 + static_cast<double>(i + j));
            a(i, j) = value;
            a(j, i) = value;
        }
    const mh::SymmetricEigen eigen = mh::jacobiEigen(a);
    runner.expect(orthogonalityError(eigen.vectors) < 1e-13, "orthonormal eigenvectors");
    runner.expect(mh::maxAbs(reconstruct(eigen) - a) < 1e-13 * mh::maxAbs(a), "V diag V^T = A");
}

CHECK_CASE("core: minimum-norm inverse")
{
    // Full column rank: the least-squares inverse (A^T A)^-1 A^T, a left inverse.
    mh::Matrix tall(9, 4);
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            tall(i, j) =
                std::cos(0.5 + 1.7 * static_cast<double>(i) * static_cast<double>(j + 1)) + (i == j ? 1.0 : 0.0);
    const mh::Matrix left = mh::minimumNormInverse(tall);
    runner.expect(left.rows() == 4 && left.columns() == 9, "size of the inverse");
    runner.expect(mh::maxAbs(left * tall - mh::Matrix::identity(4)) < 1e-12, "left inverse");
    runner.expect(penroseError(tall, left) < 1e-12, "Penrose conditions (full rank)");

    // Exact rank deficiency: rank 3 in 8 × 6, the columns of unit scale. The null space of A^T A is dropped.
    const mh::Matrix deficient = rankThree(8, 6);
    const mh::Matrix x = mh::minimumNormInverse(deficient);
    runner.expect(penroseError(deficient, x) < 1e-9, "Penrose conditions (rank 3)");
    // The minimum-norm solution of A x = b, b in the range of A, solves it and lies in the range of A^T.
    mh::Vector solution(6);
    for (std::size_t j = 0; j < 6; ++j)
        solution[j] = 0.3 * static_cast<double>(j) - 0.7;
    const mh::Vector b = deficient * solution;
    const mh::Vector least = x * b;
    const mh::Vector reproduced = deficient * least;
    for (std::size_t i = 0; i < 8; ++i)
        runner.expectNear(reproduced[i], b[i], 1e-10, "A x = b");
    runner.expect(mh::euclideanNorm(least) <= mh::euclideanNorm(solution) + 1e-12, "least norm");
}

CHECK_CASE("core: pseudo-inverse by one-sided Jacobi rotations")
{
    for (const bool wide : {false, true})
    {
        const mh::Matrix a = wide ? rankThree(5, 9) : rankThree(9, 5);
        const mh::SingularValueDecomposition svd = mh::jacobiSingularValueDecomposition(a);
        // A = U diag(sigma) V^T.
        mh::Matrix us = svd.left;
        for (std::size_t k = 0; k < svd.values.size(); ++k)
            us.scaleColumn(k, svd.values[k]);
        runner.expect(mh::maxAbs(mh::timesTranspose(us, svd.right) - a) < 1e-14 * mh::maxAbs(a), "U S V^T = A");
        std::size_t rank = 0;
        for (const double sigma : svd.values)
            rank += sigma > 1e-12 ? 1 : 0;
        runner.expect(rank == 3, "rank 3 found");
        const mh::Matrix x = mh::pseudoInverse(a, 1e-12);
        runner.expect(x.rows() == a.columns() && x.columns() == a.rows(), "size of the pseudo-inverse");
        runner.expect(penroseError(a, x) < 1e-12, std::string("Penrose conditions, ") + (wide ? "wide" : "tall"));
    }
    // An invertible matrix: the pseudo-inverse is the inverse.
    mh::Matrix square(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            square(i, j) = std::sin(static_cast<double>(3 * i + j)) + (i == j ? 2.0 : 0.0);
    runner.expect(mh::maxAbs(mh::pseudoInverse(square, 1e-12) * square - mh::Matrix::identity(4)) < 1e-13,
                  "inverse of a square matrix");
}
