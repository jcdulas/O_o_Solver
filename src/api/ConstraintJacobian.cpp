#include "oos/api/ConstraintJacobian.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace oos::api
{
ConstraintJacobian::ConstraintJacobian(const OptimizationEvaluation& point,
                                       const std::optional<std::vector<std::vector<std::size_t>>>& rowColumns)
    : m_problem(point.owner().shared_from_this()), m_rows(point.evaluation().constraints), m_variables(point.x().size())
{
    if (rowColumns && rowColumns->size() != m_rows.size())
        throw std::invalid_argument("expected one column list per constraint");
    if (!rowColumns)
        point.requireOutput(m_rows.size(), m_variables);
    m_offsets.push_back(0);
    for (std::size_t row = 0; row < m_rows.size(); ++row)
    {
        if (rowColumns)
        {
            const auto& columns = (*rowColumns)[row];
            if ((!columns.empty() && columns.back() >= m_variables) ||
                std::adjacent_find(columns.begin(), columns.end(), std::greater_equal<>()) != columns.end())
                throw std::invalid_argument("CSR columns must be in range and strictly increasing");
            point.requireOutput(1, m_columns.size() + columns.size());
            m_columns.insert(m_columns.end(), columns.begin(), columns.end());
        }
        else
        {
            m_columns.resize(m_columns.size() + m_variables);
            std::iota(m_columns.begin() + m_offsets.back(), m_columns.end(), std::size_t{0});
        }
        m_offsets.push_back(m_columns.size());
    }
}

void ConstraintJacobian::checkPoint(const OptimizationEvaluation& point) const
{
    const auto& rows = point.evaluation().constraints;
    const auto sameRow = [](const auto& a, const auto& b)
    {
        return a.setId == b.setId && a.responseId == b.responseId && a.item == b.item && a.label == b.label &&
               a.side == b.side && a.bound == b.bound;
    };
    if (m_problem.get() != &point.owner() ||
        !std::equal(rows.begin(), rows.end(), m_rows.begin(), m_rows.end(), sameRow))
        throw std::invalid_argument("the problem or constraint layout changed");
}

void ConstraintJacobian::constraints(const OptimizationEvaluation& point, std::span<double> output) const
{
    checkPoint(point);
    if (output.size() != m_rows.size())
        throw std::invalid_argument("invalid constraint output size");
    std::transform(point.evaluation().constraints.begin(), point.evaluation().constraints.end(), output.begin(),
                   [](const auto& row) { return row.normalised; });
}

void ConstraintJacobian::values(OptimizationEvaluation& point, std::span<double> output) const
{
    checkPoint(point);
    if (output.size() != m_columns.size())
        throw std::invalid_argument("invalid Jacobian output size");
    if (m_rows.empty() || !m_variables)
        return;

    class Sink final : public design::GradientSink
    {
    public:
        Sink(const ConstraintJacobian& layout, std::span<double> out, const design::DesignEvaluation& evaluation)
            : pattern(layout), result(out)
        {
            std::map<std::int32_t, std::size_t> baseRow;
            std::size_t offset = 0;
            for (const auto& response : evaluation.responses)
            {
                baseRow.emplace(response.id, offset);
                offset += response.itemCount();
            }
            for (std::size_t i = 0; i < evaluation.constraints.size(); ++i)
            {
                const auto& row = evaluation.constraints[i];
                constraintsAt[baseRow.at(row.responseId) + row.item].emplace_back(
                    i, design::ConstraintNormalisation{}.derivative(row.side, row.bound));
            }
        }

        void consume(const design::GradientBlock& block) override
        {
            for (auto it = constraintsAt.lower_bound(block.rowBegin);
                 it != constraintsAt.end() && it->first < block.rowBegin + block.rowCount; ++it)
                for (const auto& [row, scale] : it->second)
                {
                    const auto& columns = pattern.columnIndices();
                    const auto end = columns.begin() + pattern.rowOffsets()[row + 1];
                    auto entry = std::lower_bound(columns.begin() + pattern.rowOffsets()[row], end, block.columnBegin);
                    for (std::size_t c = 0; c < block.columnCount; ++c)
                    {
                        const double value = block.values[(it->first - block.rowBegin) * block.columnCount + c];
                        if (entry != end && *entry == block.columnBegin + c)
                            result[(entry++) - columns.begin()] = scale * value;
                        else if (value != 0.0)
                            throw std::invalid_argument("nonzero derivative outside the declared constraint pattern");
                    }
                }
        }

    private:
        const ConstraintJacobian& pattern;
        std::span<double> result;
        std::map<std::size_t, std::vector<std::pair<std::size_t, double>>> constraintsAt;
    } sink(*this, output, point.evaluation());
    (void)point.streamGradients(sink, design::GRADIENT_STREAM_BLOCK);
}
} // namespace oos::api
