#include "oos/output/sections/SectionLoadAccumulator.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::output
{

SectionLoadAccumulator::SectionLoadAccumulator(SectionLoadTable labels, std::size_t caseCount, bool keepsValues,
                                               std::function<void(const SectionLoadBatch&)> sink,
                                               base::MemoryBudget& budget)
    : m_table(std::move(labels)), m_caseCount(caseCount), m_sink(std::move(sink))
{
    const std::size_t rows = m_table.rowCount();
    m_table.cases.clear();
    m_table.cases.reserve(caseCount);
    if (keepsValues)
    {
        m_table.values = base::BudgetedBuffer<double>(
            budget, rows * caseCount, m_table.kind == "cutLoads" ? "cut load values" : "section load values");
        m_table.hasValues = true;
    }
    m_table.maxima.assign(rows, -std::numeric_limits<double>::infinity());
    m_table.minima.assign(rows, std::numeric_limits<double>::infinity());
    m_table.maximumCases.assign(rows, 0);
    m_table.minimumCases.assign(rows, 0);
}

void SectionLoadAccumulator::offer(std::span<const std::int32_t> cases, linalg::ConstDenseMatrixView values)
{
    const std::size_t rows = m_table.rowCount();
    if (values.rows != rows || values.columns < cases.size() || m_table.cases.size() + cases.size() > m_caseCount)
        throw std::invalid_argument(
            base::concatenate({"SectionLoadAccumulator::offer: expected ", std::to_string(rows), " rows and at most ",
                               std::to_string(m_caseCount - m_table.cases.size()), " more cases, found ",
                               std::to_string(values.rows), " rows and ", std::to_string(cases.size()), " cases"}));
    if (cases.empty())
        return;
    const std::size_t first = m_table.cases.size();
    for (std::size_t j = 0; j < cases.size(); ++j)
        for (std::size_t r = 0; r < rows; ++r)
        {
            const double value = values(r, j);
            if (value > m_table.maxima[r])
            {
                m_table.maxima[r] = value;
                m_table.maximumCases[r] = cases[j];
            }
            if (value < m_table.minima[r])
            {
                m_table.minima[r] = value;
                m_table.minimumCases[r] = cases[j];
            }
            if (m_table.hasValues)
                m_table.values[(first + j) * rows + r] = value;
        }
    m_table.cases.insert(m_table.cases.end(), cases.begin(), cases.end());
    ++m_table.batchCount;
    if (m_sink)
        m_sink(SectionLoadBatch{&m_table, cases, {values.data, values.rows, cases.size(), values.leadingDimension}});
}

SectionLoadTable SectionLoadAccumulator::finish() &&
{
    return std::move(m_table);
}

} // namespace oos::output
