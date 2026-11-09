#include "oos/linalg/eigen/DensePencil.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SingularityError.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{

DensePencil::DensePencil(std::span<const double> stiffnessDiagonal, ConstDenseMatrixView rightHand,
                         SpectralTransform transform, const DenseKernels& kernels, base::MemoryBudget& budget)
    : m_kernels(kernels), m_transform(transform)
{
    const std::size_t n = stiffnessDiagonal.size();
    if (rightHand.rows != n || rightHand.columns != n)
        throw std::invalid_argument(
            base::concatenate({"DensePencil: the right-hand matrix is ", std::to_string(rightHand.rows), " x ",
                               std::to_string(rightHand.columns), " for a stiffness diagonal of ", std::to_string(n),
                               " values; give a square matrix of the diagonal's order"}));
    m_rightHand = DenseMatrix(budget, n, n, "dense pencil: right-hand matrix");
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
        {
            m_rightHand(i, j) = rightHand(i, j);
            m_rightHand(j, i) = rightHand(i, j);
        }
    initialise(stiffnessDiagonal, budget);
}

DensePencil::DensePencil(std::span<const double> stiffnessDiagonal, DenseMatrix&& rightHand,
                         SpectralTransform transform, const DenseKernels& kernels, base::MemoryBudget& budget)
    : m_kernels(kernels), m_transform(transform)
{
    const std::size_t n = stiffnessDiagonal.size();
    if (rightHand.rows() != n || rightHand.columns() != n)
        throw std::invalid_argument(
            base::concatenate({"DensePencil: the right-hand matrix is ", std::to_string(rightHand.rows()), " x ",
                               std::to_string(rightHand.columns()), " for a stiffness diagonal of ", std::to_string(n),
                               " values; give a square matrix of the diagonal's order"}));
    m_rightHand = std::move(rightHand);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
            m_rightHand(j, i) = m_rightHand(i, j);
    initialise(stiffnessDiagonal, budget);
}

void DensePencil::initialise(std::span<const double> stiffnessDiagonal, base::MemoryBudget& budget)
{
    const std::size_t n = stiffnessDiagonal.size();
    const SpectralTransform transform = m_transform;
    m_stiffness = base::BudgetedBuffer<double>(budget, n, "dense pencil: stiffness diagonal");
    std::copy(stiffnessDiagonal.begin(), stiffnessDiagonal.end(), m_stiffness.data());
    m_factor = DenseMatrix(budget, n, n, "dense pencil: factor");
    m_diagonalRayleighBound = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double k = m_stiffness[i];
        const double b = m_rightHand(i, i);
        const bool isCounted = transform == SpectralTransform::Buckling ? b != 0.0 && k > 0.0 : b > 0.0 && k > 0.0;
        if (isCounted)
            m_diagonalRayleighBound = std::min(m_diagonalRayleighBound, k / std::abs(b));
    }
}

std::uint64_t DensePencil::estimateMemoryBytes(std::size_t dimension) noexcept
{
    const auto leading = static_cast<std::uint64_t>(DenseMatrix::alignedLeadingDimension(dimension));
    return 2 * leading * dimension * sizeof(double) + dimension * (2 * sizeof(double) + sizeof(std::size_t));
}

ShiftFactorisation DensePencil::factorise(double shift)
{
    const std::size_t n = m_stiffness.size();
    // K − σB, lower triangle (factorWithPivoting reads it alone).
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = j; i < n; ++i)
            m_factor(i, j) = -shift * m_rightHand(i, j);
        m_factor(j, j) += m_stiffness[j];
    }
    PivotedLdlt pivoted = m_kernels.factorWithPivoting(m_factor.view());
    m_permutation = std::move(pivoted.permutation);
    m_subdiagonal = std::move(pivoted.subdiagonal);
    m_hasFactor = true;
    m_isSingular = pivoted.inertia.zero > 0;
    return {shift, pivoted.inertia};
}

void DensePencil::solve(DenseMatrixView block) const
{
    if (!m_hasFactor)
        throw std::logic_error("DensePencil::solve: no factor yet; call factorise(shift) first");
    if (m_isSingular)
        throw base::SingularityError(
            "the reduced pencil is singular at the shift: a reduced eigenvalue sits on it (no remedy: the Lanczos "
            "solver moves its shifts off the roots; report it if this is raised)",
            {});
    const std::size_t n = m_stiffness.size();
    if (block.rows != n)
        throw std::invalid_argument(base::concatenate({"DensePencil::solve: expected a block of ", std::to_string(n),
                                                       " rows, found ", std::to_string(block.rows)}));
    // (K − σB)⁻¹ = Pᵀ·L⁻ᵀ·D⁻¹·L⁻¹·P with P·A·Pᵀ = L·D·Lᵀ (permutation[k]: the original index now at k).
    std::vector<double> permuted(n * block.columns); // bounded by the block
    const DenseMatrixView y{permuted.data(), n, block.columns, std::max<std::size_t>(n, 1)};
    for (std::size_t j = 0; j < block.columns; ++j)
        for (std::size_t k = 0; k < n; ++k)
            y(k, j) = block(m_permutation[k], j);
    const ConstDenseMatrixView factored = m_factor.view();
    m_kernels.solveUnitLower(factored, y);
    m_kernels.solveBlockDiagonal(blockDiagonalOf(factored, m_subdiagonal.data()), y);
    m_kernels.solveUnitLowerTransposed(factored, y);
    for (std::size_t j = 0; j < block.columns; ++j)
        for (std::size_t k = 0; k < n; ++k)
            block(m_permutation[k], j) = y(k, j);
}

void DensePencil::multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const
{
    const std::size_t n = m_stiffness.size();
    if (x.rows != n || y.rows != n || x.columns != y.columns)
        throw std::invalid_argument(base::concatenate(
            {"DensePencil::multiplyMass: expected blocks of ", std::to_string(n), " rows with the same columns, found ",
             std::to_string(x.rows), " x ", std::to_string(x.columns), " and ", std::to_string(y.rows), " x ",
             std::to_string(y.columns)}));
    // y = B·x as 0 − (−B)·x would need −B: y = −(0 − B·x), each element one chain over B's columns.
    for (std::size_t j = 0; j < y.columns; ++j)
        std::fill_n(&y(0, j), n, 0.0);
    m_kernels.subtractProduct(y, m_rightHand.view(), x);
    for (std::size_t j = 0; j < y.columns; ++j)
        for (std::size_t i = 0; i < n; ++i)
            y(i, j) = -y(i, j);
}

void DensePencil::multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const
{
    if (m_transform == SpectralTransform::Vibration)
    {
        multiplyMass(x, y);
        return;
    }
    const std::size_t n = m_stiffness.size();
    for (std::size_t j = 0; j < y.columns; ++j)
        for (std::size_t i = 0; i < n; ++i)
            y(i, j) = m_stiffness[i] * x(i, j);
}

} // namespace oos::linalg
