#include "oos/linalg/dense/SmallEigenSolver.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{
namespace
{

// The convergence test of tql2: a subdiagonal term is negligible below ε·max(|d_l| + |e_l|) (ε = 2⁻⁵²).
constexpr double MACHINE_EPSILON = 0x1p-52;

// √(a² + b²) without overflow or destructive underflow, from √ and divisions only (correctly rounded operations), so
// that its bits do not depend on the math library (std::hypot does).
double hypotenuse(double a, double b)
{
    const double absA = std::abs(a);
    const double absB = std::abs(b);
    if (absA > absB)
    {
        const double ratio = b / a;
        return absA * std::sqrt(1.0 + ratio * ratio);
    }
    if (b != 0.0)
    {
        const double ratio = a / b;
        return absB * std::sqrt(1.0 + ratio * ratio);
    }
    return 0.0;
}

// EISPACK tred2: reduces the symmetric matrix held in v to tridiagonal form by Householder similarity transformations
// and accumulates them in v. On return d holds the diagonal and e the subdiagonal (e[i] = T(i, i−1), e[0] = 0).
void reduceToTridiagonal(DenseMatrix& v, std::vector<double>& d, std::vector<double>& e)
{
    const std::size_t n = v.rows();
    for (std::size_t j = 0; j < n; ++j)
        d[j] = v(n - 1, j);

    for (std::size_t i = n - 1; i > 0; --i)
    {
        // Scale the row to avoid under- and overflow in the Householder vector.
        double scale = 0.0;
        double h = 0.0;
        for (std::size_t k = 0; k < i; ++k)
            scale += std::abs(d[k]);
        if (scale == 0.0)
        {
            e[i] = d[i - 1];
            for (std::size_t j = 0; j < i; ++j)
            {
                d[j] = v(i - 1, j);
                v(i, j) = 0.0;
                v(j, i) = 0.0;
            }
        }
        else
        {
            for (std::size_t k = 0; k < i; ++k)
            {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            double f = d[i - 1];
            double g = std::sqrt(h);
            if (f > 0.0)
                g = -g;
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for (std::size_t j = 0; j < i; ++j)
                e[j] = 0.0;

            // The similarity transformation of the remaining columns.
            for (std::size_t j = 0; j < i; ++j)
            {
                f = d[j];
                v(j, i) = f;
                g = e[j] + v(j, j) * f;
                for (std::size_t k = j + 1; k < i; ++k)
                {
                    g += v(k, j) * d[k];
                    e[k] += v(k, j) * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (std::size_t j = 0; j < i; ++j)
            {
                e[j] /= h;
                f += e[j] * d[j];
            }
            const double hh = f / (h + h);
            for (std::size_t j = 0; j < i; ++j)
                e[j] -= hh * d[j];
            for (std::size_t j = 0; j < i; ++j)
            {
                f = d[j];
                g = e[j];
                for (std::size_t k = j; k < i; ++k)
                    v(k, j) -= (f * e[k] + g * d[k]);
                d[j] = v(i - 1, j);
                v(i, j) = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate the transformations.
    for (std::size_t i = 0; i + 1 < n; ++i)
    {
        v(n - 1, i) = v(i, i);
        v(i, i) = 1.0;
        const double h = d[i + 1];
        if (h != 0.0)
        {
            for (std::size_t k = 0; k <= i; ++k)
                d[k] = v(k, i + 1) / h;
            for (std::size_t j = 0; j <= i; ++j)
            {
                double g = 0.0;
                for (std::size_t k = 0; k <= i; ++k)
                    g += v(k, i + 1) * v(k, j);
                for (std::size_t k = 0; k <= i; ++k)
                    v(k, j) -= g * d[k];
            }
        }
        for (std::size_t k = 0; k <= i; ++k)
            v(k, i + 1) = 0.0;
    }
    for (std::size_t j = 0; j < n; ++j)
    {
        d[j] = v(n - 1, j);
        v(n - 1, j) = 0.0;
    }
    v(n - 1, n - 1) = 1.0;
    e[0] = 0.0;
}

// EISPACK tql2: the eigenvalues and eigenvectors of the tridiagonal matrix (d, e) by the implicit QL algorithm, the
// rotations applied to the columns of v. Returns false when an eigenvalue needs more than MAX_SWEEPS_PER_EIGENVALUE
// sweeps.
bool diagonaliseTridiagonal(DenseMatrix& v, std::vector<double>& d, std::vector<double>& e)
{
    const std::size_t n = v.rows();
    for (std::size_t i = 1; i < n; ++i)
        e[i - 1] = e[i];
    e[n - 1] = 0.0;

    double shiftSum = 0.0;
    double scale = 0.0;
    for (std::size_t l = 0; l < n; ++l)
    {
        // The first negligible subdiagonal term at or after l splits the matrix.
        scale = std::max(scale, std::abs(d[l]) + std::abs(e[l]));
        std::size_t m = l;
        while (m < n && std::abs(e[m]) > MACHINE_EPSILON * scale)
            ++m;
        if (m == n)
            m = n - 1; // e[n − 1] = 0 always stops the search; kept for safety

        if (m > l)
        {
            std::size_t sweeps = 0;
            do
            {
                if (++sweeps > SmallEigenSolver::MAX_SWEEPS_PER_EIGENVALUE)
                    return false;
                // The implicit shift from the leading 2 × 2 block (Wilkinson).
                double g = d[l];
                double p = (d[l + 1] - g) / (2.0 * e[l]);
                double r = hypotenuse(p, 1.0);
                if (p < 0.0)
                    r = -r;
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                const double dl1 = d[l + 1];
                double h = g - d[l];
                for (std::size_t i = l + 2; i < n; ++i)
                    d[i] -= h;
                shiftSum += h;

                // The implicit QL transformation, rotations from row m − 1 up to row l.
                p = d[m];
                double c = 1.0;
                double c2 = c;
                double c3 = c;
                const double el1 = e[l + 1];
                double s = 0.0;
                double s2 = 0.0;
                for (std::size_t i = m; i-- > l;)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypotenuse(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for (std::size_t k = 0; k < n; ++k)
                    {
                        h = v(k, i + 1);
                        v(k, i + 1) = s * v(k, i) + c * h;
                        v(k, i) = c * v(k, i) - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
            } while (std::abs(e[l]) > MACHINE_EPSILON * scale);
        }
        d[l] += shiftSum;
        e[l] = 0.0;
    }
    return true;
}

// Sorts the eigenvalues in increasing order (selection sort: stable choice of the first minimum, n ≤ a few hundred)
// and signs each eigenvector so that its largest component is positive.
void sortAndSign(DenseMatrix& v, std::vector<double>& d)
{
    const std::size_t n = v.rows();
    for (std::size_t i = 0; i + 1 < n; ++i)
    {
        std::size_t smallest = i;
        for (std::size_t j = i + 1; j < n; ++j)
            if (d[j] < d[smallest])
                smallest = j;
        if (smallest != i)
        {
            std::swap(d[i], d[smallest]);
            for (std::size_t k = 0; k < n; ++k)
                std::swap(v(k, i), v(k, smallest));
        }
    }
    for (std::size_t j = 0; j < n; ++j)
    {
        std::size_t largest = 0;
        for (std::size_t k = 1; k < n; ++k)
            if (std::abs(v(k, j)) > std::abs(v(largest, j)))
                largest = k;
        if (v(largest, j) < 0.0)
            for (std::size_t k = 0; k < n; ++k)
                v(k, j) = -v(k, j);
    }
}

} // namespace

SymmetricEigenDecomposition SmallEigenSolver::decompose(ConstDenseMatrixView a, base::MemoryBudget& budget)
{
    if (a.rows != a.columns || a.leadingDimension < std::max<std::size_t>(a.rows, 1))
        throw std::invalid_argument(base::concatenate(
            {"SmallEigenSolver::decompose: expected a square matrix with a leading dimension of at least its order, "
             "found ",
             std::to_string(a.rows), " × ", std::to_string(a.columns), " with a leading dimension of ",
             std::to_string(a.leadingDimension)}));
    const std::size_t n = a.rows;
    SymmetricEigenDecomposition result{std::vector<double>(n),
                                       DenseMatrix(budget, n, n, "eigenvectors of a small symmetric problem")};
    if (n == 0)
        return result;
    DenseMatrix& v = result.eigenvectors;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
        {
            if (!std::isfinite(a(i, j)))
                throw base::SolverError(base::concatenate(
                    {"SmallEigenSolver::decompose: the term (", std::to_string(i), ", ", std::to_string(j),
                     ") of the matrix is not finite; the problem passed to the eigensolver is wrong: check the "
                     "model for values that are not finite (a zero thickness, a zero mass)"}));
            v(i, j) = a(i, j);
            v(j, i) = a(i, j);
        }

    std::vector<double> e(n);
    reduceToTridiagonal(v, result.eigenvalues, e);
    if (!diagonaliseTridiagonal(v, result.eigenvalues, e))
        throw base::SolverError(base::concatenate({"SmallEigenSolver::decompose: the QL iteration did not converge in ",
                                                   std::to_string(MAX_SWEEPS_PER_EIGENVALUE),
                                                   " sweeps for an eigenvalue of a matrix of order ", std::to_string(n),
                                                   "; check the matrix for NaN or infinite terms"}));
    sortAndSign(v, result.eigenvalues);
    return result;
}

} // namespace oos::linalg
