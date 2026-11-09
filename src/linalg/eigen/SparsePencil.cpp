#include "oos/linalg/eigen/SparsePencil.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
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

// min K_ii / M_ii over M_ii > 0 and K_ii > 0, after checking that the mass diagonal is finite and not negative.
double computeDiagonalRayleighBound(const SymmetricCsc& stiffness, const SymmetricCsc& mass)
{
    double bound = std::numeric_limits<double>::infinity();
    const auto dimension = static_cast<std::size_t>(mass.dimension());
    for (std::size_t j = 0; j < dimension; ++j)
    {
        const double m = diagonalOf(mass, j);
        if (!std::isfinite(m) || m < 0.0)
            throw base::SolverError(base::concatenate(
                {"eigenvalue analysis: the mass matrix has the diagonal term ", std::to_string(m), " at equation ",
                 std::to_string(j),
                 "; the Lanczos method needs a positive semi-definite mass matrix (no negative or infinite mass): "
                 "check the masses and densities of the model"}));
        const double k = diagonalOf(stiffness, j);
        if (m > 0.0 && k > 0.0)
            bound = std::min(bound, k / m);
    }
    return bound;
}

} // namespace

SparsePencil::SparsePencil(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness, const SymmetricCsc& mass,
                           const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_stiffness(stiffness),
      m_mass(mass),
      m_kernels(kernels),
      m_options(options),
      m_budget(budget)
{
    if (!stiffness.hasSameStructure(mass))
        throw std::invalid_argument(base::concatenate(
            {"SparsePencil: the stiffness and mass matrices must have the same structure, found ",
             std::to_string(stiffness.nonZeroCount()), " and ", std::to_string(mass.nonZeroCount()),
             " terms (or other rows); assemble both on one sparse pattern and convert them with one DOF selection"}));
    if (static_cast<std::int64_t>(stiffness.dimension()) != symbolic.dofCount())
        throw std::invalid_argument(
            base::concatenate({"SparsePencil: the matrices have ", std::to_string(stiffness.dimension()),
                               " equations, the symbolic factor ", std::to_string(symbolic.dofCount()),
                               "; analyse the pattern the matrices were converted on"}));
    m_options.pivoting = PivotingMode::Indefinite;
    m_diagonalRayleighBound = computeDiagonalRayleighBound(stiffness, mass);
    // PERFORMANCE: a lumped mass is stored on the pattern of K (one structure for both, so that K − σM is formed term
    // by term), but only its diagonal is non-zero: its products use the diagonal, n multiplications per column in
    // place of a pass over the whole pattern. The bits are those of the general product (the zero terms add ±0).
    // Measured: 3 000-DOF space truss, 20 modes: the mass products from 99 ms to 0.7 ms, the solve from 231 to 128 ms.
    const auto values = mass.values();
    const auto rows = mass.rowIndices();
    const auto offsets = mass.columnOffsets();
    bool isDiagonal = true;
    for (std::size_t j = 0; isDiagonal && j < static_cast<std::size_t>(mass.dimension()); ++j)
    {
        for (auto k = static_cast<std::size_t>(offsets[j]); k < static_cast<std::size_t>(offsets[j + 1]); ++k)
        {
            if (static_cast<std::size_t>(rows[k]) != j && values[k] != 0.0)
            {
                isDiagonal = false;
                break;
            }
        }
    }
    if (isDiagonal)
    {
        const auto dimension = static_cast<std::size_t>(mass.dimension());
        m_massDiagonal = base::BudgetedBuffer<double>(budget, dimension, "eigenvalue pencil: mass diagonal");
        for (std::size_t j = 0; j < dimension; ++j)
            m_massDiagonal[j] = diagonalOf(mass, j);
    }
}

ShiftFactorisation SparsePencil::factorise(double shift)
{
    // The previous factor and shifted matrix are given back first: one factor alive at a time.
    m_hasFactor = false;
    m_factor = SupernodalFactor();
    m_shifted = SymmetricCsc();
    m_shifted = SymmetricCsc::combine(1.0, m_stiffness, -shift, m_mass, m_budget);
    m_factor = SupernodalFactor::factorise(m_symbolic, m_shifted, m_kernels, m_options, m_budget);
    m_hasFactor = true;
    ++m_factorisationCount;
    return {shift, m_factor.inertia()};
}

void SparsePencil::solve(DenseMatrixView block) const
{
    if (!m_hasFactor)
        throw std::logic_error("SparsePencil::solve: no factor yet; call factorise(shift) first");
    FactorSolve(m_factor, m_kernels).solve(block, m_options.cancellation);
}

void SparsePencil::multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const
{
    if (m_massDiagonal.empty())
    {
        m_mass.multiply(x, y, m_kernels.options().pool, m_options.cancellation);
        return;
    }
    const auto dimension = static_cast<std::size_t>(m_mass.dimension());
    if (x.rows != dimension || y.rows != dimension || x.columns != y.columns)
        throw std::invalid_argument(base::concatenate(
            {"SparsePencil::multiplyMass: expected blocks of ", std::to_string(dimension),
             " rows with the same columns, found ", std::to_string(x.rows), " x ", std::to_string(x.columns), " and ",
             std::to_string(y.rows), " x ", std::to_string(y.columns)}));
    for (std::size_t j = 0; j < x.columns; ++j)
    {
        for (std::size_t i = 0; i < dimension; ++i)
            y(i, j) = m_massDiagonal[i] * x(i, j);
    }
}

std::uint64_t SparsePencil::estimateMemoryBytes(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness,
                                                std::size_t threadCount)
{
    const auto dimension = static_cast<std::uint64_t>(stiffness.dimension());
    const auto nonZeros = static_cast<std::uint64_t>(stiffness.nonZeroCount());
    // The shifted matrix and the diagonal of a lumped mass.
    const std::uint64_t shifted = (dimension + 1) * sizeof(std::int64_t) +
                                  nonZeros * (sizeof(std::int32_t) + sizeof(double)) + dimension * sizeof(double);
    return shifted + SupernodalFactor::estimateMemoryBytes(symbolic, threadCount, PivotingMode::Indefinite);
}

} // namespace oos::linalg
