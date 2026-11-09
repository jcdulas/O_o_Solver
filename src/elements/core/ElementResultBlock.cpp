#include "oos/elements/core/ElementResultBlock.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace oos::elements
{

void ElementResultBlock::reset(std::vector<std::string> columns)
{
    m_columns = std::move(columns);
    m_elementRows.clear();
    m_elementIds.clear();
    m_places.clear();
    m_values.clear();
}

void ElementResultBlock::append(std::int64_t elementRow, std::int32_t elementId, ResultPlace place,
                                std::span<const double> values)
{
    if (values.size() != m_columns.size())
        throw std::invalid_argument(base::concatenate(
            {"element results of element ", std::to_string(elementId), ": expected ", std::to_string(m_columns.size()),
             " values (one per column), found ", std::to_string(values.size()), "; the recovery facet is wrong"}));
    m_elementRows.push_back(elementRow);
    m_elementIds.push_back(elementId);
    m_places.push_back(place);
    m_values.insert(m_values.end(), values.begin(), values.end());
}

std::span<const double> ElementResultBlock::values(std::size_t i) const
{
    return std::span<const double>(m_values).subspan(i * m_columns.size(), m_columns.size());
}

std::span<double> ElementResultBlock::mutableValues(std::size_t i)
{
    return std::span<double>(m_values).subspan(i * m_columns.size(), m_columns.size());
}

void ElementResultBlock::reserve(std::size_t rows)
{
    m_elementRows.reserve(rows);
    m_elementIds.reserve(rows);
    m_places.reserve(rows);
    m_values.reserve(rows * m_columns.size());
}

double ElementResultBlock::value(std::size_t i, const std::string& column) const
{
    const auto found = std::find(m_columns.begin(), m_columns.end(), column);
    if (found == m_columns.end())
    {
        std::string known;
        for (const std::string& name : m_columns)
            known.append(known.empty() ? "" : ", ").append(name);
        throw std::invalid_argument(
            base::concatenate({"element results: no column '", column, "'; name one of the columns ", known}));
    }
    return values(i)[static_cast<std::size_t>(found - m_columns.begin())];
}

} // namespace oos::elements
