#include "oos/elements/core/ElementMatrixBlock.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::elements
{

void ElementMatrixBlock::clear() noexcept
{
    m_elementRows.clear();
    m_dofOffsets.resize(1);
    m_dofs.clear();
    m_valueOffsets.resize(1);
    m_values.clear();
}

void ElementMatrixBlock::append(std::int64_t elementRow, std::span<const ElementDof> dofs,
                                std::span<const double> values)
{
    if (values.size() != dofs.size() * dofs.size())
        throw std::invalid_argument(base::concatenate(
            {"element matrix of batch row ", std::to_string(elementRow), ": expected ",
             std::to_string(dofs.size() * dofs.size()), " values (", std::to_string(dofs.size()),
             " DOFs squared), found ", std::to_string(values.size()), "; the element facet is wrong"}));
    m_elementRows.push_back(elementRow);
    m_dofs.insert(m_dofs.end(), dofs.begin(), dofs.end());
    m_dofOffsets.push_back(m_dofs.size());
    m_values.insert(m_values.end(), values.begin(), values.end());
    m_valueOffsets.push_back(m_values.size());
}

std::span<const ElementDof> ElementMatrixBlock::dofs(std::size_t i) const
{
    return std::span<const ElementDof>(m_dofs).subspan(m_dofOffsets[i], m_dofOffsets[i + 1] - m_dofOffsets[i]);
}

std::span<const double> ElementMatrixBlock::matrix(std::size_t i) const
{
    return std::span<const double>(m_values).subspan(m_valueOffsets[i], m_valueOffsets[i + 1] - m_valueOffsets[i]);
}

} // namespace oos::elements
