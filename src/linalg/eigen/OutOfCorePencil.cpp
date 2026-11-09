#include "oos/linalg/eigen/OutOfCorePencil.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::linalg
{

OutOfCorePencil::OutOfCorePencil(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness,
                                 const SymmetricCsc& rightHand, SpectralTransform transform,
                                 const DenseKernels& kernels, const FactorOptions& options,
                                 const ScratchSpool::Options& spool, base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_stiffness(&stiffness),
      m_rightHand(&rightHand),
      m_transform(transform),
      m_kernels(kernels),
      m_options(options),
      m_spool(spool),
      m_budget(budget)
{
    m_options.pivoting = PivotingMode::Indefinite;
    if (transform == SpectralTransform::Buckling)
        m_buckling = std::make_unique<BucklingPencil>(symbolic, stiffness, rightHand, kernels, m_options, budget);
    else
        m_vibration = std::make_unique<SparsePencil>(symbolic, stiffness, rightHand, kernels, m_options, budget);
}

OutOfCorePencil::OutOfCorePencil(const SymbolicFactor& symbolic, PencilSource& source, SpectralTransform transform,
                                 const DenseKernels& kernels, const FactorOptions& options,
                                 const ScratchSpool::Options& spool, base::MemoryBudget& budget)
    : m_symbolic(symbolic),
      m_source(&source),
      m_transform(transform),
      m_kernels(kernels),
      m_options(options),
      m_spool(spool),
      m_budget(budget)
{
    m_options.pivoting = PivotingMode::Indefinite;
    // SparsePencil's and BucklingPencil's bounds from the source's diagonals: min K_ii / M_ii over M_ii > 0 and
    // K_ii > 0 (the mass checked positive semi-definite), or min K_ii / |B_ii| over B_ii ≠ 0 and K_ii > 0.
    const auto n = static_cast<std::size_t>(symbolic.dofCount());
    base::BudgetedBuffer<double> stiffness(budget, n, "out-of-core pencil: stiffness diagonal");
    base::BudgetedBuffer<double> rightHand(budget, n, "out-of-core pencil: right-hand diagonal");
    source.diagonals(stiffness.span(), rightHand.span());
    double bound = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n; ++j)
    {
        const double k = stiffness[j];
        const double b = rightHand[j];
        if (transform == SpectralTransform::Vibration)
        {
            if (!std::isfinite(b) || b < 0.0)
                throw base::SolverError(base::concatenate(
                    {"eigenvalue analysis: the mass matrix has the diagonal term ", std::to_string(b), " at equation ",
                     std::to_string(j),
                     "; the Lanczos method needs a positive semi-definite mass matrix (no negative or infinite mass): "
                     "check the masses and densities of the model"}));
            if (b > 0.0 && k > 0.0)
                bound = std::min(bound, k / b);
        }
        else if (b != 0.0 && std::isfinite(b) && k > 0.0)
            bound = std::min(bound, k / std::abs(b));
    }
    m_sourceRayleighBound = bound;
}

ShiftFactorisation OutOfCorePencil::factorise(double shift)
{
    // The previous factor, its spool and the shifted matrix given back first: one factor alive at a time.
    if (m_hasFactor)
    {
        const OutOfCoreTraffic traffic = m_factor.traffic();
        m_earlier.writtenBytes += traffic.writtenBytes;
        m_earlier.readBytes += traffic.readBytes;
        m_earlier.readRequests += traffic.readRequests;
        m_earlier.readSeconds += traffic.readSeconds;
    }
    m_hasFactor = false;
    m_factor = OutOfCoreFactor();
    m_shifted = SymmetricCsc();
    if (m_source != nullptr)
        m_factor = OutOfCoreFactor::factorise(m_symbolic, m_source->shiftedSource(shift), m_kernels, m_options, m_spool,
                                              m_budget);
    else
    {
        // SparsePencil's K − σM, BucklingPencil's K + σK_d: the same matrix to the bit.
        m_shifted = m_transform == SpectralTransform::Buckling
                        ? SymmetricCsc::combine(1.0, *m_stiffness, shift, *m_rightHand, m_budget)
                        : SymmetricCsc::combine(1.0, *m_stiffness, -shift, *m_rightHand, m_budget);
        m_factor = OutOfCoreFactor::factorise(m_symbolic, m_shifted, m_kernels, m_options, m_spool, m_budget);
    }
    m_largestSpoolBytes = std::max(m_largestSpoolBytes, m_factor.spooledBytes());
    m_hasFactor = true;
    ++m_factorisationCount;
    sample();
    return {shift, m_factor.inertia()};
}

void OutOfCorePencil::sample() const
{
    constexpr std::size_t SAMPLED_CONSUMERS = 12;
    const std::uint64_t used = m_budget.used();
    if (used <= m_sampledPeakBytes)
        return;
    m_sampledPeakBytes = used;
    m_sampledConsumers = m_budget.largestConsumers(SAMPLED_CONSUMERS);
}

void OutOfCorePencil::solve(DenseMatrixView block) const
{
    if (!m_hasFactor)
        throw std::logic_error("OutOfCorePencil::solve: no factor yet; call factorise(shift) first");
    m_factor.solve(block, m_kernels, m_options.cancellation);
    sample();
}

void OutOfCorePencil::multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const
{
    if (m_source != nullptr)
        m_source->multiplyRightHand(x, y);
    else if (m_buckling)
        m_buckling->multiplyMass(x, y);
    else
        m_vibration->multiplyMass(x, y);
}

void OutOfCorePencil::multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const
{
    if (m_source != nullptr && m_transform == SpectralTransform::Buckling)
        m_source->multiplyStiffness(x, y);
    else if (m_source != nullptr)
        m_source->multiplyRightHand(x, y);
    else if (m_buckling)
        m_buckling->multiplyInnerProduct(x, y);
    else
        m_vibration->multiplyInnerProduct(x, y);
}

double OutOfCorePencil::diagonalRayleighBound() const
{
    if (m_source != nullptr)
        return m_sourceRayleighBound;
    return m_buckling ? m_buckling->diagonalRayleighBound() : m_vibration->diagonalRayleighBound();
}

OutOfCoreTraffic OutOfCorePencil::traffic() const noexcept
{
    OutOfCoreTraffic traffic = m_earlier;
    if (m_hasFactor)
    {
        const OutOfCoreTraffic current = m_factor.traffic();
        traffic.writtenBytes += current.writtenBytes;
        traffic.readBytes += current.readBytes;
        traffic.readRequests += current.readRequests;
        traffic.readSeconds += current.readSeconds;
    }
    return traffic;
}

} // namespace oos::linalg
