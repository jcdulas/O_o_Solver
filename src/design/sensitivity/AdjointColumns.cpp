#include "oos/design/sensitivity/AdjointColumns.hpp"

#include <algorithm>
#include <numeric>

namespace oos::design
{

AdjointColumns::AdjointColumns(const SensitivitySystem& system, const PseudoLoads& loads,
                               std::span<const std::size_t> subcases, bool readsEvery, base::MemoryBudget& budget,
                               base::ThreadPool* pool)
    : m_system(&system), m_readsEvery(readsEvery)
{
    const auto g = static_cast<std::size_t>(system.dofMap().gSetSize());
    base::BudgetedBuffer<std::uint8_t> isRead(budget, g, "adjoint column reads");
    for (const std::size_t s : subcases)
        if (!m_readsEvery && !loads.markReads(s, isRead.span()))
            m_readsEvery = true;
    const auto equations = static_cast<std::size_t>(system.equationCount());
    m_equationOfDof = base::BudgetedBuffer<std::int32_t>(budget, g, "adjoint column equations of the DOFs");
    std::fill(m_equationOfDof.data(), m_equationOfDof.data() + g, -1);
    for (std::size_t e = 0; e < equations; ++e)
        m_equationOfDof[static_cast<std::size_t>(system.dofOfEquation(static_cast<std::int32_t>(e)))] =
            static_cast<std::int32_t>(e);
    const assembly::RigidElimination& gm = system.gm();
    std::vector<std::uint8_t> isTerm(g, 0); // one per g-set DOF, for the setup
    std::size_t termCount = 0;
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
        for (const assembly::ConstraintTerm& term : gm.row(i))
            if (isTerm[static_cast<std::size_t>(term.dof)] == 0)
            {
                isTerm[static_cast<std::size_t>(term.dof)] = 1;
                ++termCount;
            }
    m_gmTermDofs = base::BudgetedBuffer<base::DofIndex>(budget, termCount, "adjoint column Gm term DOFs");
    std::size_t next = 0;
    for (std::size_t dof = 0; dof < g; ++dof)
        if (isTerm[dof] != 0)
            m_gmTermDofs[next++] = static_cast<base::DofIndex>(dof);
    if (m_readsEvery)
        return;
    std::vector<std::int32_t> rowOfDof(g, -1); // one per g-set DOF, for the setup
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
        rowOfDof[static_cast<std::size_t>(gm.mDofs()[i])] = static_cast<std::int32_t>(i);
    const auto readCount = static_cast<std::size_t>(
        std::count_if(isRead.data(), isRead.data() + g, [](std::uint8_t flag) { return flag != 0; }));
    m_readDofs = base::BudgetedBuffer<base::DofIndex>(budget, readCount, "adjoint column read DOFs");
    m_readSource = base::BudgetedBuffer<std::int32_t>(budget, readCount, "adjoint column read sources");
    next = 0;
    for (std::size_t dof = 0; dof < g; ++dof)
    {
        if (isRead[dof] == 0)
            continue;
        m_readDofs[next] = static_cast<base::DofIndex>(dof);
        m_readSource[next] = m_equationOfDof[dof] >= 0 ? m_equationOfDof[dof]
                             : rowOfDof[dof] >= 0      ? -1 - rowOfDof[dof]
                                                       : NONE;
        ++next;
    }
    m_dots = std::make_unique<PseudoLoadDots>(loads, m_readDofs.span(), g, budget, pool);
}

void AdjointColumns::reduceToEquations(std::vector<double>& scratch, std::vector<base::DofIndex>& touched,
                                       std::span<double> equations) const
{
    m_system->gm().reduceToN(scratch);
    std::fill(equations.begin(), equations.end(), 0.0);
    const auto gather = [&](base::DofIndex dof)
    {
        const std::int32_t e = m_equationOfDof[static_cast<std::size_t>(dof)];
        if (e >= 0)
            equations[static_cast<std::size_t>(e)] = scratch[static_cast<std::size_t>(dof)];
    };
    for (const base::DofIndex dof : touched)
        gather(dof);
    for (const base::DofIndex dof : m_gmTermDofs.span())
        gather(dof);
    for (const base::DofIndex dof : touched)
        scratch[static_cast<std::size_t>(dof)] = 0.0;
    for (const base::DofIndex dof : m_gmTermDofs.span())
        scratch[static_cast<std::size_t>(dof)] = 0.0;
    touched.clear();
}

double AdjointColumns::valueAtRead(std::span<const double> equations, std::size_t k) const
{
    const std::int32_t source = m_readSource[k];
    double value = 0.0;
    if (source >= 0)
        value = equations[static_cast<std::size_t>(source)];
    else if (source != NONE)
    {
        // RigidElimination::expandToG: the sum of the row over the n-set values (zero off the equations).
        for (const assembly::ConstraintTerm& term : m_system->gm().row(static_cast<std::size_t>(-1 - source)))
        {
            const std::int32_t e = m_equationOfDof[static_cast<std::size_t>(term.dof)];
            value += term.coefficient * (e >= 0 ? equations[static_cast<std::size_t>(e)] : 0.0);
        }
    }
    return value;
}

std::int32_t AdjointColumns::firstEquation(const SparseVector& weights) const noexcept
{
    auto first = static_cast<std::int32_t>(m_system->equationCount());
    for (const auto& [dof, value] : weights)
        if (const std::int32_t e = m_equationOfDof[static_cast<std::size_t>(dof)]; e >= 0)
            first = std::min(first, e);
    return first;
}

std::vector<std::size_t> AdjointColumns::solveSlots(std::span<const std::int32_t> firstEquations)
{
    std::vector<std::size_t> order(firstEquations.size()); // bounded by a solve's columns
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return firstEquations[a] < firstEquations[b]; });
    std::vector<std::size_t> slot(order.size()); // the same
    for (std::size_t position = 0; position < order.size(); ++position)
        slot[order[position]] = position;
    return slot;
}

void AdjointColumns::expand(std::span<const double> equations, std::span<double> vectorG) const
{
    if (m_readsEvery)
    {
        m_system->fromEquations(equations, vectorG);
        return;
    }
    for (std::size_t k = 0; k < m_readDofs.size(); ++k)
        vectorG[static_cast<std::size_t>(m_readDofs[k])] = valueAtRead(equations, k);
}

void AdjointColumns::addProducts(std::span<const std::size_t> columnSubcases, std::span<double* const> destinations,
                                 const TileFill& fill)
{
    m_dots->addProducts(columnSubcases, destinations, fill);
}

} // namespace oos::design
