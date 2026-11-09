#include "oos/reduction/absorption/SpiderChangeOfBasis.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/dense/SmallEigenSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

double scaleOf(SpiderDofKind kind, double characteristicLength) noexcept
{
    return kind == SpiderDofKind::Rotation ? characteristicLength : 1.0;
}

} // namespace

SpiderChangeOfBasis::SpiderChangeOfBasis(const SpiderWeightMatrix& weights, base::MemoryBudget& budget)
    : m_referenceDofCount(weights.referenceDofCount), m_legDofCount(weights.legDofCount)
{
    const std::size_t m = m_referenceDofCount;
    const std::size_t n = m_legDofCount;
    if (m == 0 || weights.values.size() != m * n || weights.referenceKinds.size() != m || weights.legKinds.size() != n)
        throw std::invalid_argument(base::concatenate(
            {"SpiderChangeOfBasis: expected at least one reference DOF and ", std::to_string(m), " x ",
             std::to_string(n), " weights with one kind per DOF, found ", std::to_string(weights.values.size()),
             " weights, ", std::to_string(weights.referenceKinds.size()), " reference kinds and ",
             std::to_string(weights.legKinds.size()), " leg kinds"}));
    const double length = weights.characteristicLength;
    if (!std::isfinite(length) || !(length > 0.0))
        throw std::invalid_argument(base::concatenate(
            {"SpiderChangeOfBasis: expected a positive characteristic length, found ", std::to_string(length)}));
    for (const double value : weights.values)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "SpiderChangeOfBasis: a weight is not finite; the spider's relation is broken: give its "
                "RigidInterpolation finite "
                "weights");

    m_legScales.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        m_legScales[i] = scaleOf(weights.legKinds[i], length);
    m_referenceScales.resize(m);
    for (std::size_t j = 0; j < m; ++j)
        m_referenceScales[j] = scaleOf(weights.referenceKinds[j], length);

    if (n < m)
    {
        // Fewer leg DOFs than reference DOFs: W cannot have full row rank.
        m_isRankDeficient = true;
        m_conditionNumber = std::numeric_limits<double>::infinity();
        return;
    }

    // A = W̃ᵀ (n × m, column-major), W̃ = S_r·W·S_c⁻¹.
    m_reflectors = base::BudgetedBuffer<double>(budget, n * m, "spider change of basis: reflectors");
    const auto a = [this, n](std::size_t i, std::size_t j) -> double& { return m_reflectors[i + j * n]; };
    for (std::size_t j = 0; j < m; ++j)
        for (std::size_t i = 0; i < n; ++i)
            a(i, j) = weights.at(j, i) * m_referenceScales[j] / m_legScales[i];

    // Householder QR, column by column (Golub & Van Loan §5.2.2): H_j = I − β_j·v_j·v_jᵀ maps rows j..n−1 of column j
    // to α_j·e_j, α_j = −sign(x_j)·‖x‖ (the sign that avoids cancellation in v_j = x − α_j·e_j); v_j is stored scaled
    // so that v_j[j] = 1, which gives β_j = −(x_j − α_j)/α_j (vᵀv = −2·α_j·(x_j − α_j)).
    m_betas.assign(m, 0.0);
    m_r.assign(m * m, 0.0);
    for (std::size_t j = 0; j < m; ++j)
    {
        double squaredNorm = 0.0;
        for (std::size_t i = j; i < n; ++i)
            squaredNorm += a(i, j) * a(i, j);
        const double norm = std::sqrt(squaredNorm);
        const double head = a(j, j);
        if (norm == 0.0)
        {
            // A zero column: no reflection (β = 0, v = e_j); R(j, j) = 0 makes W̃ rank-deficient below.
            a(j, j) = 1.0;
        }
        else
        {
            const double alpha = head >= 0.0 ? -norm : norm;
            const double pivot = head - alpha;
            for (std::size_t i = j + 1; i < n; ++i)
                a(i, j) /= pivot;
            a(j, j) = 1.0;
            m_betas[j] = -pivot / alpha;
            m_r[j + j * m] = alpha;
            const linalg::DenseMatrixView rest{m_reflectors.data() + (j + 1) * n, n, m - j - 1, n};
            applyReflector(j, rest);
        }
        for (std::size_t k = j + 1; k < m; ++k)
            m_r[j + k * m] = a(j, k);
    }

    // The singular values of W̃ from the eigenvalues of Rᵀ·R = W̃·W̃ᵀ (m × m, symmetric).
    std::vector<double> normal(m * m, 0.0);
    for (std::size_t p = 0; p < m; ++p)
        for (std::size_t q = 0; q < m; ++q)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k <= std::min(p, q); ++k)
                sum += m_r[k + p * m] * m_r[k + q * m];
            normal[p + q * m] = sum;
        }
    const linalg::SymmetricEigenDecomposition eigen =
        linalg::SmallEigenSolver::decompose(linalg::ConstDenseMatrixView(normal.data(), m, m, m), budget);
    m_singularValues.resize(m);
    for (std::size_t k = 0; k < m; ++k)
        m_singularValues[k] = std::sqrt(std::max(eigen.eigenvalues[k], 0.0));
    const double largest = m_singularValues.back();
    const double smallest = m_singularValues.front();
    m_isRankDeficient = !(largest > 0.0) || smallest <= RANK_TOLERANCE * largest;
    m_conditionNumber = m_isRankDeficient ? std::numeric_limits<double>::infinity() : largest / smallest;
}

void SpiderChangeOfBasis::applyReflector(std::size_t j, linalg::DenseMatrixView rows) const noexcept
{
    const double beta = m_betas[j];
    if (beta == 0.0)
        return;
    const std::size_t n = m_legDofCount;
    const double* v = m_reflectors.data() + j * n;
    for (std::size_t c = 0; c < rows.columns; ++c)
    {
        double dot = 0.0;
        for (std::size_t i = j; i < n; ++i)
            dot += v[i] * rows(i, c);
        dot *= beta;
        for (std::size_t i = j; i < n; ++i)
            rows(i, c) -= dot * v[i];
    }
}

void SpiderChangeOfBasis::requireFullRank(const char* operation) const
{
    if (m_isRankDeficient)
        throw std::logic_error(base::concatenate(
            {"SpiderChangeOfBasis::", operation,
             ": the spider's weighting matrix is rank-deficient, so it has no change of basis; the safeguard "
             "(AbsorptionSafeguard) keeps such a spider on its legs and nothing may transform it"}));
}

void SpiderChangeOfBasis::transformRows(linalg::DenseMatrixView rows) const
{
    requireFullRank("transformRows");
    const std::size_t m = m_referenceDofCount;
    const std::size_t n = m_legDofCount;
    if (rows.rows != n)
        throw std::invalid_argument(
            base::concatenate({"SpiderChangeOfBasis::transformRows: expected ", std::to_string(n),
                               " rows (the leg DOFs), found ", std::to_string(rows.rows)}));
    // Tᵀ = blockdiag(S_r·R⁻¹, I)·Qᵀ·S_c⁻¹, Qᵀ = H_{m−1}···H_0.
    for (std::size_t c = 0; c < rows.columns; ++c)
        for (std::size_t i = 0; i < n; ++i)
            rows(i, c) /= m_legScales[i];
    for (std::size_t j = 0; j < m; ++j)
        applyReflector(j, rows);
    for (std::size_t c = 0; c < rows.columns; ++c)
    {
        // Back substitution R·x = b on the first m rows (R upper triangular, non-singular when of full rank).
        for (std::size_t i = m; i-- > 0;)
        {
            double sum = rows(i, c);
            for (std::size_t k = i + 1; k < m; ++k)
                sum -= m_r[i + k * m] * rows(k, c);
            rows(i, c) = sum / m_r[i + i * m];
        }
        for (std::size_t i = 0; i < m; ++i)
            rows(i, c) *= m_referenceScales[i];
    }
}

void SpiderChangeOfBasis::transformSymmetric(linalg::DenseMatrixView matrix) const
{
    requireFullRank("transformSymmetric");
    const std::size_t n = m_legDofCount;
    if (matrix.rows != n || matrix.columns != n)
        throw std::invalid_argument(base::concatenate(
            {"SpiderChangeOfBasis::transformSymmetric: expected a ", std::to_string(n), " x ", std::to_string(n),
             " matrix (the leg DOFs), found ", std::to_string(matrix.rows), " x ", std::to_string(matrix.columns)}));
    // Tᵀ·K·T = Tᵀ·(Tᵀ·K)ᵀ for a symmetric K.
    transformRows(matrix);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
            std::swap(matrix(i, j), matrix(j, i));
    transformRows(matrix);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
        {
            const double mean = 0.5 * (matrix(i, j) + matrix(j, i));
            matrix(i, j) = mean;
            matrix(j, i) = mean;
        }
}

void SpiderChangeOfBasis::expand(std::span<const double> reference, std::span<const double> deformation,
                                 std::span<double> legs) const
{
    requireFullRank("expand");
    const std::size_t m = m_referenceDofCount;
    const std::size_t n = m_legDofCount;
    if (reference.size() != m || deformation.size() != n - m || legs.size() != n)
        throw std::invalid_argument(base::concatenate(
            {"SpiderChangeOfBasis::expand: expected ", std::to_string(m), " reference, ", std::to_string(n - m),
             " deformation and ", std::to_string(n), " leg values, found ", std::to_string(reference.size()), ", ",
             std::to_string(deformation.size()), " and ", std::to_string(legs.size())}));
    // T = S_c⁻¹·Q·blockdiag(R⁻ᵀ·S_r, I), Q = H_0···H_{m−1}.
    for (std::size_t i = 0; i < m; ++i)
    {
        // Forward substitution Rᵀ·y = S_r·u_ref (Rᵀ lower triangular).
        double sum = m_referenceScales[i] * reference[i];
        for (std::size_t k = 0; k < i; ++k)
            sum -= m_r[k + i * m] * legs[k];
        legs[i] = sum / m_r[i + i * m];
    }
    for (std::size_t k = m; k < n; ++k)
        legs[k] = deformation[k - m];
    const linalg::DenseMatrixView column{legs.data(), n, 1, n};
    for (std::size_t j = m; j-- > 0;)
        applyReflector(j, column);
    for (std::size_t i = 0; i < n; ++i)
        legs[i] /= m_legScales[i];
}

} // namespace oos::reduction
