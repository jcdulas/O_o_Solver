#include "oos/elements/core/ElementVectorBlock.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::elements
{

void ElementVectorBlock::clear() noexcept
{
    m_elementRows.clear();
    m_offsets.resize(1);
    m_dofs.clear();
    m_values.clear();
}

void ElementVectorBlock::append(std::int64_t elementRow, std::span<const ElementDof> dofs,
                                std::span<const double> values)
{
    if (values.size() != dofs.size())
        throw std::invalid_argument(base::concatenate(
            {"element load of batch row ", std::to_string(elementRow), ": expected ", std::to_string(dofs.size()),
             " values (one per DOF), found ", std::to_string(values.size()), "; the loads facet is wrong"}));
    m_elementRows.push_back(elementRow);
    m_dofs.insert(m_dofs.end(), dofs.begin(), dofs.end());
    m_values.insert(m_values.end(), values.begin(), values.end());
    m_offsets.push_back(m_dofs.size());
}

std::span<const ElementDof> ElementVectorBlock::dofs(std::size_t i) const
{
    return std::span<const ElementDof>(m_dofs).subspan(m_offsets[i], m_offsets[i + 1] - m_offsets[i]);
}

std::span<const double> ElementVectorBlock::values(std::size_t i) const
{
    return std::span<const double>(m_values).subspan(m_offsets[i], m_offsets[i + 1] - m_offsets[i]);
}

} // namespace oos::elements
