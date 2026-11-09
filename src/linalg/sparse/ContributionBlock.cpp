#include "oos/linalg/sparse/ContributionBlock.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::linalg
{

void ContributionBlock::clear() noexcept
{
    m_dofOffsets.resize(1);
    m_dofs.clear();
    m_valueOffsets.resize(1);
    m_values.clear();
}

void ContributionBlock::append(std::span<const NodeDof> dofs, std::span<const double> matrix)
{
    if (matrix.size() != dofs.size() * dofs.size())
        throw std::invalid_argument(
            base::concatenate({"ContributionBlock::append: expected a ", std::to_string(dofs.size()), " x ",
                               std::to_string(dofs.size()), " matrix (", std::to_string(dofs.size() * dofs.size()),
                               " values), found ", std::to_string(matrix.size()), " values"}));
    m_dofs.insert(m_dofs.end(), dofs.begin(), dofs.end());
    m_dofOffsets.push_back(m_dofs.size());
    m_values.insert(m_values.end(), matrix.begin(), matrix.end());
    m_valueOffsets.push_back(m_values.size());
}

} // namespace oos::linalg
