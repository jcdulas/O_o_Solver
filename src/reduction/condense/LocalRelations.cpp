#include "oos/reduction/condense/LocalRelations.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/condense/SymmetricBlockAccess.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::reduction
{

LocalRelations::LocalRelations(std::size_t rowCount, std::size_t termCount, base::MemoryBudget& budget,
                               const std::string& label)
    : m_dependents(budget, rowCount, label), m_offsets(budget, rowCount, label), m_terms(budget, termCount, label)
{
}

void LocalRelations::appendRow(std::int64_t dependent)
{
    if (m_rowCount == m_dependents.size())
        throw std::logic_error(base::concatenate({"LocalRelations: more than the ", std::to_string(m_dependents.size()),
                                                  " rows given (an internal error: report it)"}));
    if (m_rowCount > 0 && dependent <= m_dependents[m_rowCount - 1])
        throw std::logic_error(
            base::concatenate({"LocalRelations: dependent DOF ", std::to_string(dependent), " after ",
                               std::to_string(m_dependents[m_rowCount - 1]),
                               "; the rows come by increasing dependent DOF (an internal error: report it)"}));
    m_dependents[m_rowCount] = dependent;
    m_offsets[m_rowCount] = static_cast<std::int64_t>(m_termCount);
    ++m_rowCount;
}

void LocalRelations::appendTerm(LocalTerm term)
{
    if (m_rowCount == 0 || m_termCount == m_terms.size())
        throw std::logic_error(
            base::concatenate({"LocalRelations: a term before the first row or past the ",
                               std::to_string(m_terms.size()), " terms given (an internal error: report it)"}));
    m_terms[m_termCount++] = term;
}

void LocalRelations::eliminate(const linalg::SparsePattern& pattern, std::span<const std::int64_t> firstDof,
                               const linalg::NodeGraph& graph, std::span<double> values,
                               base::MemoryBudget& budget) const
{
    const SymmetricBlockAccess access(pattern, firstDof, values);
    const auto dofCount = static_cast<std::size_t>(firstDof[firstDof.size() - 1]);
    base::BudgetedBuffer<std::int64_t> rowOfDof(budget, dofCount, "relations row of DOF");
    std::fill(rowOfDof.data(), rowOfDof.data() + dofCount, -1);
    for (std::size_t r = 0; r < m_rowCount; ++r)
        rowOfDof[static_cast<std::size_t>(m_dependents[r])] = static_cast<std::int64_t>(r);
    // The DOFs of a node and of its neighbours, by increasing node.
    std::vector<std::int64_t> coupled; // bounded by a node's degree
    const auto gatherCoupled = [&](std::int32_t node)
    {
        coupled.clear();
        const auto addNode = [&](std::int32_t n)
        {
            for (std::int64_t dof = firstDof[static_cast<std::size_t>(n)];
                 dof < firstDof[static_cast<std::size_t>(n) + 1]; ++dof)
                coupled.push_back(dof);
        };
        bool isOwnAdded = false;
        for (const std::int32_t neighbour : graph.neighbours(node))
        {
            if (!isOwnAdded && neighbour > node)
            {
                addNode(node);
                isOwnAdded = true;
            }
            addNode(neighbour);
        }
        if (!isOwnAdded)
            addNode(node);
    };
    for (std::size_t r = 0; r < m_rowCount; ++r)
    {
        const std::int64_t d = m_dependents[r];
        gatherCoupled(access.nodeOf(d));
        for (const std::int64_t i : coupled)
        {
            const double kid = access.read(i, d);
            if (kid == 0.0)
                continue;
            if (const std::int64_t rowE = rowOfDof[static_cast<std::size_t>(i)]; rowE >= 0)
            {
                // Gmᵀ·K̄mm·Gm: d and e = i both dependent.
                for (const LocalTerm& termJ : terms(r))
                    for (const LocalTerm& termK : terms(static_cast<std::size_t>(rowE)))
                        access.add(termJ.dof, termK.dof, termJ.coefficient * kid * termK.coefficient);
                continue;
            }
            // K̄nm·Gm and its transpose: i independent.
            for (const LocalTerm& term : terms(r))
            {
                const double value = kid * term.coefficient;
                access.add(i, term.dof, value);
                access.add(term.dof, i, value);
            }
        }
    }
}

void LocalRelations::reduce(std::span<double> loads) const noexcept
{
    for (std::size_t r = 0; r < m_rowCount; ++r)
    {
        double& dependentLoad = loads[static_cast<std::size_t>(m_dependents[r])];
        if (dependentLoad == 0.0)
            continue;
        for (const LocalTerm& term : terms(r))
            loads[static_cast<std::size_t>(term.dof)] += term.coefficient * dependentLoad;
        dependentLoad = 0.0;
    }
}

void LocalRelations::expand(std::span<double> values, std::span<const std::uint8_t> isSkipped) const noexcept
{
    for (std::size_t r = 0; r < m_rowCount; ++r)
    {
        if (!isSkipped.empty() && isSkipped[r] != 0)
            continue;
        double sum = 0.0;
        for (const LocalTerm& term : terms(r))
            sum += term.coefficient * values[static_cast<std::size_t>(term.dof)];
        values[static_cast<std::size_t>(m_dependents[r])] = sum;
    }
}

} // namespace oos::reduction
