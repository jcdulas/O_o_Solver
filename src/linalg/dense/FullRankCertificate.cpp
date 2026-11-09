#include "oos/linalg/dense/FullRankCertificate.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace oos::linalg
{

namespace
{

// Householder reflector of x (Golub and Van Loan, Algorithm 5.1.1): v with v[0] = 1 and β such that
// (I − β v vᵀ) x = ∓‖x‖ e₁. Overwrites x[1:] with v[1:] and x[0] with the new diagonal; returns β (0 when x[1:] = 0).
double makeReflector(double* x, std::size_t length)
{
    double tail = 0.0;
    for (std::size_t i = 1; i < length; ++i)
        tail += x[i] * x[i];
    if (tail == 0.0)
        return 0.0;
    const double alpha = x[0];
    const double norm = std::sqrt(alpha * alpha + tail);
    const double diagonal = alpha <= 0.0 ? norm : -norm;
    const double v0 = alpha - diagonal;
    for (std::size_t i = 1; i < length; ++i)
        x[i] /= v0;
    x[0] = diagonal;
    return -v0 / diagonal;
}

// y ← (I − β v vᵀ) y for the column y of length `length`, v[0] = 1 implied and v[1:] in `v`.
void applyReflector(const double* v, double beta, double* y, std::size_t length)
{
    double dot = y[0];
    for (std::size_t i = 1; i < length; ++i)
        dot += v[i] * y[i];
    const double scale = beta * dot;
    y[0] -= scale;
    for (std::size_t i = 1; i < length; ++i)
        y[i] -= scale * v[i];
}

// T of the compact WY form Q = H_1···H_b = I − V·T·Vᵀ of the b reflectors in the columns of `v` (unit diagonal stored,
// zeros above it), T upper triangular (LAPACK xLARFT, forward, columnwise): T(k, k) = β_k and
// T(0:k, k) = −β_k·T(0:k, 0:k)·(V(:, 0:k)ᵀ·v_k).
void formT(ConstDenseMatrixView v, std::span<const double> betas, std::size_t b, DenseMatrixView t)
{
    std::array<double, FullRankCertificate::PANEL_COLUMNS> z{};
    for (std::size_t k = 0; k < b; ++k)
    {
        for (std::size_t i = 0; i < k; ++i)
        {
            double dot = 0.0;
            for (std::size_t r = k; r < v.rows; ++r)
                dot += v(r, i) * v(r, k);
            z[i] = -betas[k] * dot;
        }
        for (std::size_t i = 0; i < k; ++i)
        {
            double sum = 0.0;
            for (std::size_t l = i; l < k; ++l)
                sum += t(i, l) * z[l];
            t(i, k) = sum;
        }
        t(k, k) = betas[k];
        for (std::size_t i = k + 1; i < b; ++i)
            t(i, k) = 0.0;
    }
}

} // namespace

FullRankCertificate::Outcome FullRankCertificate::prove(DenseMatrixView a, double bound, const DenseKernels& kernels,
                                                        base::MemoryBudget& budget)
{
    const std::size_t m = a.rows;
    const std::size_t n = a.columns;
    if (m < n)
        throw std::invalid_argument(
            base::concatenate({"FullRankCertificate::prove: expected at least as many rows as columns, found ",
                               std::to_string(m), " rows for ", std::to_string(n), " columns"}));
    Outcome outcome;
    if (n == 0)
    {
        outcome.isProved = true;
        outcome.singularValueBound = std::numeric_limits<double>::infinity();
        return outcome;
    }
    base::BudgetedBuffer<double> betas(budget, PANEL_COLUMNS, "rank certificate reflectors");
    DenseMatrix v(budget, m, PANEL_COLUMNS, "rank certificate panel");
    DenseMatrix t(budget, PANEL_COLUMNS, PANEL_COLUMNS, "rank certificate T");
    DenseMatrix w(budget, PANEL_COLUMNS, n, "rank certificate V'A");
    DenseMatrix tw(budget, PANEL_COLUMNS, n, "rank certificate T'V'A");

    for (std::size_t p0 = 0; p0 < n; p0 += PANEL_COLUMNS)
    {
        const std::size_t b = std::min(PANEL_COLUMNS, n - p0);
        // The panel, column by column (Householder QR of A(p0:m, p0:p0 + b)).
        for (std::size_t k = 0; k < b; ++k)
        {
            const std::size_t j = p0 + k;
            double* x = &a(j, j);
            betas[k] = makeReflector(x, m - j);
            // σ_min(R) ≤ |r_jj|: below the bound no proof is possible (a NaN fails too).
            if (!(std::abs(a(j, j)) >= bound))
            {
                outcome.factoredColumns = j;
                return outcome;
            }
            for (std::size_t c = j + 1; c < p0 + b; ++c)
                applyReflector(x, betas[k], &a(j, c), m - j);
        }
        outcome.factoredColumns = p0 + b;
        const std::size_t trailing = n - (p0 + b);
        if (trailing == 0)
            continue;
        // A2 ← Qᵀ·A2 = A2 − V·(Tᵀ·(Vᵀ·A2)), V explicit (unit diagonal, zeros above it).
        const std::size_t rows = m - p0;
        const DenseMatrixView panel = v.view().block(0, 0, rows, b);
        for (std::size_t k = 0; k < b; ++k)
            for (std::size_t i = 0; i < rows; ++i)
                panel(i, k) = i < k ? 0.0 : (i == k ? 1.0 : a(p0 + i, p0 + k));
        const DenseMatrixView tb = t.view().block(0, 0, b, b);
        formT(panel, betas.span(), b, tb);
        const DenseMatrixView a2 = a.block(p0, p0 + b, rows, trailing);
        const DenseMatrixView minusVtA = w.view().block(0, 0, b, trailing);
        for (std::size_t c = 0; c < trailing; ++c)
            for (std::size_t i = 0; i < b; ++i)
                minusVtA(i, c) = 0.0;
        kernels.subtractTransposedProduct(minusVtA, panel, a2); // −Vᵀ·A2
        const DenseMatrixView product = tw.view().block(0, 0, b, trailing);
        for (std::size_t c = 0; c < trailing; ++c)
            for (std::size_t i = 0; i < b; ++i)
            {
                // (Tᵀ·Vᵀ·A2)(i, c) = Σ_{l ≤ i} T(l, i)·(Vᵀ·A2)(l, c).
                double sum = 0.0;
                for (std::size_t l = 0; l <= i; ++l)
                    sum += tb(l, i) * minusVtA(l, c);
                product(i, c) = -sum;
            }
        kernels.subtractProduct(a2, panel, product);
    }

    // ‖R⁻¹‖_F = ‖R⁻ᵀ‖_F = ‖D⁻¹·L⁻¹‖_F, L(i, j) = r_ji / r_jj below the diagonal (the reflectors are no longer needed).
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
            a(i, j) = a(j, i) / a(j, j);
    DenseKernelOptions serialOptions = kernels.options();
    serialOptions.pool = nullptr;
    const DenseKernels serial(kernels.kind(), serialOptions);
    const ConstDenseMatrixView l(a.data, n, n, a.leadingDimension);
    const double squares = base::parallelReduce(
        kernels.options().pool, n, INVERSE_BLOCK_COLUMNS, 0.0,
        [&](std::size_t begin, std::size_t end)
        {
            // Columns [begin, end) of L⁻¹: zero above row `begin`, L(begin:n, begin:n)⁻¹ times the identity below.
            const std::size_t width = end - begin;
            const std::size_t below = n - begin;
            DenseMatrix x(budget, below, width, "rank certificate inverse block");
            for (std::size_t c = 0; c < width; ++c)
                x(c, c) = 1.0;
            serial.solveUnitLower(l.block(begin, begin, below, below), x.view());
            double sum = 0.0;
            for (std::size_t c = 0; c < width; ++c)
                for (std::size_t i = 0; i < below; ++i)
                {
                    const double value = x(i, c) / a(begin + i, begin + i);
                    sum += value * value;
                }
            return sum;
        },
        [](double total, double piece) { return total + piece; });
    outcome.singularValueBound = squares > 0.0 ? 1.0 / std::sqrt(squares) : std::numeric_limits<double>::infinity();
    outcome.isProved = outcome.singularValueBound >= bound;
    return outcome;
}

} // namespace oos::linalg
