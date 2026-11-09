#include "oos/output/sections/SparseRows.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::output
{

namespace
{

// The rows of one task of apply(): enough to amortise the task, few enough to share a few thousand rows between the
// threads.
constexpr std::size_t ROWS_PER_TASK = 256;

} // namespace

SparseRows::SparseRows(const std::vector<std::vector<std::pair<base::DofIndex, double>>>& rows,
                       base::MemoryBudget& budget)
{
    std::size_t entries = 0;
    for (const std::vector<std::pair<base::DofIndex, double>>& row : rows)
        entries += row.size();
    m_starts = base::BudgetedBuffer<std::size_t>(budget, rows.size() + 1, "sparse rows starts");
    m_dofs = base::BudgetedBuffer<base::DofIndex>(budget, entries, "sparse rows DOFs");
    m_coefficients = base::BudgetedBuffer<double>(budget, entries, "sparse rows coefficients");
    std::size_t next = 0;
    for (std::size_t r = 0; r < rows.size(); ++r)
    {
        m_starts[r] = next;
        for (const auto& [dof, coefficient] : rows[r])
        {
            m_dofs[next] = dof;
            m_coefficients[next] = coefficient;
            ++next;
        }
    }
    m_starts[rows.size()] = next;
}

std::span<const base::DofIndex> SparseRows::dofsOf(std::size_t row) const
{
    return m_dofs.span().subspan(m_starts[row], m_starts[row + 1] - m_starts[row]);
}

std::span<const double> SparseRows::coefficientsOf(std::size_t row) const
{
    return m_coefficients.span().subspan(m_starts[row], m_starts[row + 1] - m_starts[row]);
}

void SparseRows::apply(linalg::ConstDenseMatrixView displacements, linalg::DenseMatrixView out,
                       base::ThreadPool* pool) const
{
    if (out.rows != rowCount() || out.columns != displacements.columns)
        throw std::invalid_argument(
            base::concatenate({"SparseRows::apply: expected an output of ", std::to_string(rowCount()), " rows and ",
                               std::to_string(displacements.columns), " columns, found ", std::to_string(out.rows),
                               " × ", std::to_string(out.columns)}));
    const auto rowsOfTask = [&](std::size_t task)
    {
        const std::size_t first = task * ROWS_PER_TASK;
        const std::size_t last = std::min(rowCount(), first + ROWS_PER_TASK);
        for (std::size_t j = 0; j < displacements.columns; ++j)
            for (std::size_t r = first; r < last; ++r)
            {
                double sum = 0.0;
                for (std::size_t k = m_starts[r]; k < m_starts[r + 1]; ++k)
                    sum += m_coefficients[k] * displacements(static_cast<std::size_t>(m_dofs[k]), j);
                out(r, j) = sum;
            }
    };
    const std::size_t tasks = (rowCount() + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    if (pool == nullptr || tasks < 2)
    {
        for (std::size_t task = 0; task < tasks; ++task)
            rowsOfTask(task);
        return;
    }
    pool->runBatch(tasks, rowsOfTask);
}

void SparseRows::apply(std::span<const double> displacements, std::span<double> values) const
{
    if (values.size() != rowCount())
        throw std::invalid_argument(base::concatenate({"SparseRows::apply: expected ", std::to_string(rowCount()),
                                                       " values, found ", std::to_string(values.size())}));
    for (std::size_t r = 0; r < rowCount(); ++r)
    {
        double sum = 0.0;
        for (std::size_t k = m_starts[r]; k < m_starts[r + 1]; ++k)
            sum += m_coefficients[k] * displacements[static_cast<std::size_t>(m_dofs[k])];
        values[r] = sum;
    }
}

} // namespace oos::output
