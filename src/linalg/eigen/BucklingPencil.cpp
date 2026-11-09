#include "oos/linalg/eigen/BucklingPencil.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

// The diagonal term of column j (0 when it is not stored).
double diagonalOf(const SymmetricCsc& matrix, std::size_t j)
{
    const auto offsets = matrix.columnOffsets();
    for (auto k = static_cast<std::size_t>(offsets[j]); k < static_cast<std::size_t>(offsets[j + 1]); ++k)
    {
        if (static_cast<std::size_t>(matrix.rowIndices()[k]) == j)
            return matrix.values()[k];
    }
    return 0.0;
}

// min K_ii / |K_d,ii| over K_d,ii ≠ 0 and K_ii > 0: the Rayleigh quotient of a unit vector e_i of K φ = λ B φ is
// K_ii / B_ii, and for K positive definite the lowest root on the side of the sign of B_ii is at most its magnitude.
double computeDiagonalRayleighBound(const SymmetricCsc& stiffness, const SymmetricCsc& differential)
{
    double bound = std::numeric_limits<double>::infinity();
    const auto dimension = static_cast<std::size_t>(stiffness.dimension());
    for (std::size_t j = 0; j < dimension; ++j)
    {
        const double d = diagonalOf(differential, j);
        const double k = diagonalOf(stiffness, j);
        if (d != 0.0 && std::isfinite(d) && k > 0.0)
            bound = std::min(bound, k / std::abs(d));
    }
    return bound;
}

} // namespace

BucklingPencil::BucklingPencil(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness,
                               const SymmetricCsc& differential, const DenseKernels& kernels,
                               const FactorOptions& options, base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_stiffness(stiffness),
      m_differential(differential),
      m_kernels(kernels),
      m_options(options),
      m_budget(budget)
{
    if (!stiffness.hasSameStructure(differential))
        throw std::invalid_argument(base::concatenate(
            {"BucklingPencil: the stiffness and differential stiffness matrices must have the same structure, found ",
             std::to_string(stiffness.nonZeroCount()), " and ", std::to_string(differential.nonZeroCount()),
             " terms (or other rows); assemble both on one sparse pattern and convert them with one DOF selection"}));
    if (static_cast<std::int64_t>(stiffness.dimension()) != symbolic.dofCount())
        throw std::invalid_argument(
            base::concatenate({"BucklingPencil: the matrices have ", std::to_string(stiffness.dimension()),
                               " equations, the symbolic factor ", std::to_string(symbolic.dofCount()),
                               "; analyse the pattern the matrices were converted on"}));
    m_options.pivoting = PivotingMode::Indefinite;
    m_diagonalRayleighBound = computeDiagonalRayleighBound(stiffness, differential);
}

ShiftFactorisation BucklingPencil::factorise(double shift)
{
    // The previous factor and shifted matrix are given back first: one factor alive at a time.
    m_hasFactor = false;
    m_factor = SupernodalFactor();
    m_shifted = SymmetricCsc();
    // K − σB = K + σK_d.
    m_shifted = SymmetricCsc::combine(1.0, m_stiffness, shift, m_differential, m_budget);
    m_factor = SupernodalFactor::factorise(m_symbolic, m_shifted, m_kernels, m_options, m_budget);
    m_hasFactor = true;
    ++m_factorisationCount;
    return {shift, m_factor.inertia()};
}

void BucklingPencil::solve(DenseMatrixView block) const
{
    if (!m_hasFactor)
        throw std::logic_error("BucklingPencil::solve: no factor yet; call factorise(shift) first");
    FactorSolve(m_factor, m_kernels).solve(block, m_options.cancellation);
}

void BucklingPencil::multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const
{
    m_differential.multiply(x, y, m_kernels.options().pool, m_options.cancellation);
    for (std::size_t j = 0; j < y.columns; ++j)
    {
        for (std::size_t i = 0; i < y.rows; ++i)
            y(i, j) = -y(i, j);
    }
}

void BucklingPencil::multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const
{
    m_stiffness.multiply(x, y, m_kernels.options().pool, m_options.cancellation);
}

std::uint64_t BucklingPencil::estimateMemoryBytes(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness,
                                                  std::size_t threadCount)
{
    const auto dimension = static_cast<std::uint64_t>(stiffness.dimension());
    const auto nonZeros = static_cast<std::uint64_t>(stiffness.nonZeroCount());
    const std::uint64_t shifted =
        (dimension + 1) * sizeof(std::int64_t) + nonZeros * (sizeof(std::int32_t) + sizeof(double));
    return shifted + SupernodalFactor::estimateMemoryBytes(symbolic, threadCount, PivotingMode::Indefinite);
}

} // namespace oos::linalg
