#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include "FactorSchedule.hpp"
#include "MultifrontalDriver.hpp"
#include "SolvePlan.hpp"

#include "oos/linalg/dense/DenseMatrix.hpp"

namespace oos::linalg
{

SupernodalFactor SupernodalFactor::factorise(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                             const DenseKernels& kernels, const FactorOptions& options,
                                             base::MemoryBudget& budget)
{
    return factoriseSupernodes(symbolic, matrix, kernels, options, false, nullptr, budget);
}

SupernodalFactor SupernodalFactor::factoriseSupernodes(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                       const DenseKernels& kernels, const FactorOptions& options,
                                                       bool excludesTrailing, DenseMatrix* schurComplement,
                                                       base::MemoryBudget& budget)
{
    SupernodalFactor factor;
    MultifrontalDriver driver(symbolic, matrix, kernels, options, excludesTrailing, budget);
    driver.run(factor, schurComplement);
    return factor;
}

std::uint64_t SupernodalFactor::estimateMemoryBytes(const SymbolicFactor& symbolic, std::size_t threadCount,
                                                    PivotingMode pivoting)
{
    base::MemoryBudget scratch(base::MemoryBudget::UNLIMITED);
    const FactorSchedule schedule(symbolic, false, threadCount, pivoting, scratch);
    const auto count = static_cast<std::uint64_t>(symbolic.supernodeCount());
    const auto dofs = static_cast<std::uint64_t>(symbolic.dofCount());
    std::uint64_t rows = 0;
    for (std::int32_t s = 0; s < symbolic.supernodeCount(); ++s)
        rows += static_cast<std::uint64_t>(symbolic.columnCount(s) + symbolic.rowCount(s));
    const std::uint64_t panels = static_cast<std::uint64_t>(symbolic.panelValueCount()) * sizeof(double);
    const std::uint64_t workspace = static_cast<std::uint64_t>(schedule.workspaceValues()) * sizeof(double);
    // Per supernode, at most PER_SUPERNODE_BYTES: two 64-bit offsets and three 32-bit counts in the factor (28), the
    // status, zero-pivot and delay arrays of the driver (9), the plan (children, starts, slots and its temporaries:
    // 32), the table of grown panels (8). Per equation, the subdiagonal of the 2 × 2 pivots (indefinite mode).
    constexpr std::uint64_t PER_SUPERNODE_BYTES = 80;
    constexpr std::uint64_t FIXED_BYTES = 64;
    // The plan of the solves, built with the factor (SolvePlan).
    const std::uint64_t solvePlan =
        SolvePlan::estimateMemoryBytes(symbolic.supernodeCount(), static_cast<std::int64_t>(rows), symbolic.dofCount());
    return panels + rows * sizeof(std::int32_t) + workspace + count * PER_SUPERNODE_BYTES + dofs * sizeof(double) +
           solvePlan + FIXED_BYTES;
}

FactorStorage SupernodalFactor::releaseStorage() noexcept
{
    FactorStorage storage;
    storage.m_values = m_panelValues.takeValues();
    storage.m_rows = m_panelRows.takeValues();
    *this = SupernodalFactor();
    return storage;
}

const SolvePlan& SupernodalFactor::solvePlan() const noexcept
{
    static const SolvePlan EMPTY_PLAN;
    return m_solvePlan ? *m_solvePlan : EMPTY_PLAN;
}

FactorPanel SupernodalFactor::panel(std::int32_t supernode) const noexcept
{
    const auto at = static_cast<std::size_t>(supernode);
    FactorPanel result;
    const auto rowCount = static_cast<std::size_t>(m_rowCount[at]);
    const auto eliminated = static_cast<std::size_t>(m_eliminated[at]);
    if (!m_grownPanels.empty() && m_grownPanels[at])
    {
        const GrownPanel& grown = *m_grownPanels[at];
        const double* values = m_panelPointer.empty() ? grown.values.data() : m_panelPointer[at];
        result.values = {values, rowCount, eliminated, std::max<std::size_t>(rowCount, 1)};
        result.rows = grown.rows;
        result.subdiagonal = grown.subdiagonal.data();
        return result;
    }
    const double* values = m_panelPointer.empty() ? m_panelValues.data() + m_panelOffset[at] : m_panelPointer[at];
    result.values = {values, rowCount, eliminated, std::max<std::size_t>(rowCount, 1)};
    result.rows = m_panelRows.span().subspan(static_cast<std::size_t>(m_rowOffset[at]), rowCount);
    if (m_pivoting == PivotingMode::Indefinite)
        result.subdiagonal = m_subdiagonal.data() + m_firstColumn[at];
    return result;
}

} // namespace oos::linalg
