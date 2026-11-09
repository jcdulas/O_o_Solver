#include "oos/linalg/supernodal/MatrixFrontSource.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"

#include <stdexcept>
#include <string>

namespace oos::linalg
{

MatrixFrontSource::MatrixFrontSource(const SymmetricCsc& matrix, const SymbolicFactor& symbolic, double scale)
    : m_matrix(matrix), m_symbolic(symbolic), m_scale(scale)
{
    if (matrix.dimension() != symbolic.dofCount())
        throw std::invalid_argument(
            base::concatenate({"MatrixFrontSource: the matrix has ", std::to_string(matrix.dimension()),
                               " equations and the symbolic factor ", std::to_string(symbolic.dofCount()),
                               "; give the matrix in the order of the symbolic factor it was analysed with"}));
}

void MatrixFrontSource::addTerms(FrontTerms& front)
{
    const auto offsets = m_matrix.columnOffsets();
    const auto rows = m_matrix.rowIndices();
    const auto values = m_matrix.values();
    for (std::int32_t j = 0; j < front.columnCount; ++j)
    {
        const auto column = static_cast<std::size_t>(front.firstColumn + j);
        for (auto k = static_cast<std::size_t>(offsets[column]); k < static_cast<std::size_t>(offsets[column + 1]); ++k)
        {
            const std::int32_t position = front.positionOf(rows[k]);
            if (position < 0)
                throw base::InternalError(base::concatenate(
                    {"MatrixFrontSource: row ", std::to_string(rows[k]), " of column ", std::to_string(column),
                     " is not in the front of supernode ", std::to_string(front.supernode),
                     " (a matrix outside its symbolic factor's structure: a bug of O_o_Solver, report it)"}));
            front.at(position, j) += m_scale * values[k];
        }
    }
}

void MatrixFrontSource::diagonal(std::int32_t supernode, std::span<double> values) const
{
    const std::int32_t first = m_symbolic.firstColumn(supernode);
    for (std::size_t j = 0; j < values.size(); ++j)
    {
        const auto column = static_cast<std::size_t>(first) + j;
        const auto begin = static_cast<std::size_t>(m_matrix.columnOffsets()[column]);
        const bool hasDiagonal = begin < static_cast<std::size_t>(m_matrix.columnOffsets()[column + 1]) &&
                                 m_matrix.rowIndices()[begin] == static_cast<std::int32_t>(column);
        values[j] = hasDiagonal ? m_scale * m_matrix.values()[begin] : 0.0;
    }
}

} // namespace oos::linalg
