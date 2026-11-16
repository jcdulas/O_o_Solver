#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

namespace mh = oos::elements::mixedHybrid;

namespace
{

// The symmetric positive definite matrix BᵀB of the reference test, B with a dominant diagonal.
mh::Matrix referenceSpd(std::size_t n)
{
    mh::Matrix b(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            b(i, j) = static_cast<double>((i * 31 + j * 17) % 23) / 23.0 - 0.4 + (i == j ? 2.0 : 0.0);
    return mh::transposeTimes(b, b);
}

// A general square matrix of size n, well conditioned.
mh::Matrix generalMatrix(std::size_t n)
{
    mh::Matrix a(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a(i, j) =
                std::sin(1.0 + 0.7 * static_cast<double>(i) + 1.3 * static_cast<double>(j * j)) + (i == j ? 3.0 : 0.0);
    return a;
}

double distanceToIdentity(const mh::Matrix& m)
{
    return mh::maxAbs(m - mh::Matrix::identity(m.rows()));
}

} // namespace

// Port of the reference test factor_and_solves.
CHECK_CASE("core: Cholesky factor, triangular solves and Gram matrix")
{
    for (const std::size_t n : {5U, 48U, 49U, 130U, 301U})
    {
        const mh::Matrix a = referenceSpd(n);
        const std::optional<mh::Matrix> lower = mh::choleskyLower(a);
        runner.expect(lower.has_value(), "factor exists");
        if (!lower)
            continue;
        const double error = mh::maxAbs(mh::timesTranspose(*lower, *lower) - a) / mh::maxAbs(a);
        runner.expect(error < 1e-13, "factor " + std::to_string(n) + ": " + std::to_string(error));
        const std::size_t columns = n % 2 == 0 ? 7 : 40;
        mh::Matrix x0(n, columns);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < columns; ++j)
                x0(i, j) = static_cast<double>((i + 3 * j) % 11) - 5.0;
        mh::Matrix x = *lower * x0;
        mh::solveLower(*lower, x);
        runner.expect(mh::maxAbs(x - x0) < 1e-9, "L solve " + std::to_string(n));
        x = mh::transposeTimes(*lower, x0);
        mh::solveLowerTransposed(*lower, x);
        runner.expect(mh::maxAbs(x - x0) < 1e-9, "L^T solve " + std::to_string(n));
    }
    mh::Matrix y(211, 150);
    for (std::size_t i = 0; i < 211; ++i)
        for (std::size_t j = 0; j < 150; ++j)
            y(i, j) = static_cast<double>((i * 7 + j * 13) % 29) / 29.0 - 0.5;
    const mh::Matrix gram = mh::gramLower(y);
    const mh::Matrix full = mh::transposeTimes(y, y);
    double worst = 0.0;
    for (std::size_t j = 0; j < 150; ++j)
    {
        for (std::size_t i = j; i < 150; ++i)
            worst = std::fmax(worst, std::fabs(gram(i, j) - full(i, j)));
        for (std::size_t i = 0; i < j; ++i)
            worst = std::fmax(worst, std::fabs(gram(i, j)));
    }
    runner.expect(worst < 1e-12, "gram");
    runner.expect(!mh::choleskyLower(-1.0 * mh::Matrix::identity(60)).has_value(), "negative definite refused");
}

CHECK_CASE("core: Cholesky solve and inverse")
{
    const mh::Matrix a = referenceSpd(9);
    const std::optional<mh::Cholesky> cholesky = mh::Cholesky::factor(a);
    runner.expect(cholesky.has_value(), "factor exists");
    if (!cholesky)
        return;
    const mh::Matrix inverse = cholesky->inverse();
    runner.expect(distanceToIdentity(a * inverse) < 1e-12, "A A^-1 = I");
    runner.expect(inverse == inverse.transposed(), "inverse symmetric");
    mh::Vector b(9);
    for (std::size_t i = 0; i < 9; ++i)
        b[i] = std::cos(static_cast<double>(i));
    const mh::Vector x = cholesky->solve(b);
    const mh::Vector ax = a * x;
    for (std::size_t i = 0; i < 9; ++i)
        runner.expectNear(ax[i], b[i], 1e-12, "A x = b");
    const mh::Matrix xm = cholesky->solve(mh::Matrix::identity(9));
    runner.expect(mh::maxAbs(xm - inverse) < 1e-12, "solve of I is the inverse");
}

CHECK_CASE("core: equilibrated Cholesky on a diagonal of many decades")
{
    // A = D C D with C well conditioned and D spanning 16 decades: the scaled matrix is C up to its own diagonal.
    const std::size_t n = 8;
    const mh::Matrix c = referenceSpd(n);
    const auto decade = [](std::size_t i) { return std::pow(10.0, static_cast<double>(i) - 4.0); };
    mh::Matrix a(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a(i, j) = c(i, j) * decade(i) * decade(j);
    const std::optional<mh::EquilibratedCholesky> factor = mh::EquilibratedCholesky::factor(a);
    runner.expect(factor.has_value(), "factor exists");
    if (!factor)
        return;
    for (std::size_t i = 0; i < n; ++i)
        runner.expectNear(factor->scales()[i] * std::sqrt(a(i, i)), 1.0, 1e-15, "scale");
    // The scaled matrix has a unit diagonal: L L^T has it too.
    const mh::Matrix llt = mh::timesTranspose(factor->lower(), factor->lower());
    for (std::size_t i = 0; i < n; ++i)
        runner.expectNear(llt(i, i), 1.0, 1e-14, "unit diagonal");
    // A^-1 A = I in the relative sense of each term: (A^-1 A)_ij is scaled by 10^(j - i).
    const mh::Matrix inverse = factor->inverse();
    const mh::Matrix product = inverse * a;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            const double relative = product(i, j) * std::pow(10.0, static_cast<double>(i) - static_cast<double>(j));
            worst = std::fmax(worst, std::fabs(relative - (i == j ? 1.0 : 0.0)));
        }
    runner.expect(worst < 1e-12, "A^-1 A = I: " + std::to_string(worst));
    runner.expect(inverse == inverse.transposed(), "inverse symmetric");
    // B^T A^-1 B = Y^T Y with Y = L^-1 S B, and A X = B for X = A^-1 B.
    mh::Matrix b(n, 3);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            b(i, j) = std::pow(10.0, static_cast<double>(i) - 4.0) * (1.0 + static_cast<double>(i + j) * 0.1);
    const mh::Matrix y = factor->halfSolve(b);
    const mh::Matrix direct = mh::transposeTimes(b, inverse * b);
    const mh::Matrix gram = mh::transposeTimes(y, y);
    runner.expect(mh::maxAbs(gram - direct) < 1e-12 * mh::maxAbs(direct), "B^T A^-1 B = Y^T Y");
    const mh::Matrix x = factor->solve(b);
    runner.expect(mh::maxAbs(a * x - b) < 1e-12 * mh::maxAbs(b), "A X = B");
    const mh::Vector xv = factor->solve(b.column(1));
    for (std::size_t i = 0; i < n; ++i)
        runner.expect(std::fabs(xv[i] - x(i, 1)) <= 1e-13 * std::fabs(x(i, 1)), "vector solve");
    // A negative diagonal term is refused.
    mh::Matrix bad = mh::Matrix::identity(3);
    bad(1, 1) = -1.0;
    runner.expect(!mh::EquilibratedCholesky::factor(bad).has_value(), "negative diagonal refused");
}

CHECK_CASE("core: LU with partial pivoting: determinant, inverse, solve")
{
    // The second difference matrix: det = 4, inverse = [[3, 2, 1], [2, 4, 2], [1, 2, 3]] / 4.
    mh::Matrix t(3, 3);
    t(0, 0) = 2.0;
    t(0, 1) = -1.0;
    t(1, 0) = -1.0;
    t(1, 1) = 2.0;
    t(1, 2) = -1.0;
    t(2, 1) = -1.0;
    t(2, 2) = 2.0;
    runner.expectNear(mh::determinant(t), 4.0, 1e-15, "det of the second difference matrix");
    const std::optional<mh::Matrix> inverse = mh::inverse(t);
    runner.expect(inverse.has_value(), "inverse exists");
    if (inverse)
    {
        const double expected[3][3] = {{0.75, 0.5, 0.25}, {0.5, 1.0, 0.5}, {0.25, 0.5, 0.75}};
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                runner.expectNear((*inverse)(i, j), expected[i][j], 1e-15, "inverse term");
    }
    // A zero first pivot needs the row exchange: det [[0, 1], [1, 0]] = −1.
    mh::Matrix swap(2, 2);
    swap(0, 1) = 1.0;
    swap(1, 0) = 1.0;
    runner.expectNear(mh::determinant(swap), -1.0, 1e-15, "det of the exchange");
    runner.expect(mh::inverse(swap).has_value() && *mh::inverse(swap) == swap, "the exchange is its own inverse");
    // A singular matrix: no inverse, no solution, zero determinant.
    mh::Matrix singular(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            singular(i, j) = static_cast<double>(i + 1) * static_cast<double>(j + 1);
    const mh::LuDecomposition lu(singular);
    runner.expect(lu.isSingular(), "singular detected");
    runner.expect(!lu.inverse().has_value(), "no inverse");
    runner.expect(!lu.solve(mh::Vector{1.0, 2.0, 3.0}).has_value(), "no solution");
    runner.expect(lu.determinant() == 0.0, "zero determinant");
    // A general 7 × 7 matrix.
    const mh::Matrix a = generalMatrix(7);
    const std::optional<mh::Matrix> ai = mh::inverse(a);
    runner.expect(ai.has_value(), "general inverse exists");
    if (ai)
    {
        runner.expect(distanceToIdentity(a * *ai) < 1e-13, "A A^-1 = I");
        runner.expect(distanceToIdentity(*ai * a) < 1e-13, "A^-1 A = I");
    }
    // det(A) = det(L) det(U) is the product of the eigenvalues; checked through det(2A) = 2^7 det(A) and
    // det(A^T) = det(A).
    const double det = mh::determinant(a);
    runner.expectNear(mh::determinant(2.0 * a), 128.0 * det, 1e-12, "det(2A) = 2^7 det(A)");
    runner.expectNear(mh::determinant(a.transposed()), det, 1e-12, "det(A^T) = det(A)");
    mh::Vector b(7);
    for (std::size_t i = 0; i < 7; ++i)
        b[i] = 1.0 + static_cast<double>(i);
    const std::optional<mh::Vector> x = mh::LuDecomposition(a).solve(b);
    runner.expect(x.has_value(), "solution exists");
    if (x)
    {
        const mh::Vector ax = a * *x;
        for (std::size_t i = 0; i < 7; ++i)
            runner.expectNear(ax[i], b[i], 1e-13, "A x = b");
    }
}

CHECK_CASE("core: equilibrated inverse of rows of different scales")
{
    // Rows scaled from 1e-8 to 1e8, columns from 1e6 to 1e-6.
    const std::size_t n = 6;
    const mh::Matrix base = generalMatrix(n);
    mh::Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m(i, j) = base(i, j) * std::pow(10.0, 3.0 * static_cast<double>(i) - 8.0) *
                      std::pow(10.0, 6.0 - 2.4 * static_cast<double>(j));
    const std::optional<mh::Matrix> inverse = mh::equilibratedInverse(m);
    runner.expect(inverse.has_value(), "inverse exists");
    if (!inverse)
        return;
    // (m⁻¹ m)_ij scales as 10^(2.4 (i − j)): compare in the relative sense.
    const mh::Matrix product = *inverse * m;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            const double shift = static_cast<double>(j) - static_cast<double>(i);
            const double relative = product(i, j) * std::pow(10.0, 2.4 * shift);
            worst = std::fmax(worst, std::fabs(relative - (i == j ? 1.0 : 0.0)));
        }
    runner.expect(worst < 1e-12, "m^-1 m = I: " + std::to_string(worst));
    mh::Matrix zeroRow = mh::Matrix::identity(3);
    zeroRow(1, 1) = 0.0;
    runner.expect(!mh::equilibratedInverse(zeroRow).has_value(), "a zero row is singular");
}

CHECK_CASE("core: matrix products and blocks")
{
    const mh::Matrix a = generalMatrix(5).block(0, 1, 5, 3); // 5 × 3
    const mh::Matrix b = generalMatrix(5).block(1, 0, 4, 3); // 4 × 3
    const mh::Matrix ab = mh::timesTranspose(a, b);          // 5 × 4
    runner.expect(mh::maxAbs(ab - a * b.transposed()) == 0.0, "A B^T");
    runner.expect(mh::maxAbs(mh::transposeTimes(a, ab) - a.transposed() * ab) == 0.0, "A^T B");
    mh::Matrix big(6, 6);
    big.setBlock(2, 1, a.block(0, 0, 4, 3));
    runner.expect(big.block(2, 1, 4, 3) == a.block(0, 0, 4, 3), "set and get a block");
    const mh::Vector weights{1.0, 2.0, 3.0, 4.0, 5.0};
    const mh::Matrix weighted = mh::weightedRows(a, weights);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            runner.expect(weighted(i, j) == a(i, j) * weights[i], "weighted rows");
    const mh::Vector x{1.0, -2.0, 0.5};
    const mh::Vector ax = a * x;
    const mh::Vector atax = mh::transposeTimes(a, ax);
    const mh::Vector expected = mh::transposeTimes(a, a) * x;
    for (std::size_t j = 0; j < 3; ++j)
        runner.expectNear(atax[j], expected[j], 1e-14, "A^T (A x)");
    const mh::Matrix s = mh::symmetricPart(generalMatrix(4));
    runner.expect(s == s.transposed(), "symmetric part");
    runner.expectNear(mh::frobeniusNorm(mh::Matrix::identity(4)), 2.0, 1e-15, "Frobenius norm");
    mh::Matrix columns(3, 2);
    columns.setColumn(1, mh::Vector{1.0, 2.0, 3.0});
    columns.setRow(0, mh::Vector{7.0, 8.0});
    runner.expect(columns.column(1) == mh::Vector({8.0, 2.0, 3.0}), "set a column and a row");
}
