#include "oos/linalg/eigen/PencilScaling.hpp"

#include <cmath>
#include <cstddef>

namespace oos::linalg
{
namespace
{

// The scale of an equation whose stiffness diagonal term is `diagonal`.
double scaleOfDiagonalTerm(double diagonal) noexcept
{
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
        return 1.0;
    // K_ii = m·2^e with m in [0.5, 1): the scale 2^(−e/2) leaves a diagonal term between 1/4 and 2.
    int exponent = 0;
    std::frexp(diagonal, &exponent);
    return std::ldexp(1.0, -(exponent / 2));
}

} // namespace

base::BudgetedBuffer<double> pencilScaleOf(std::span<const double> stiffnessDiagonal, base::MemoryBudget& budget)
{
    base::BudgetedBuffer<double> scale(budget, stiffnessDiagonal.size(), "eigenvalue pencil: scales");
    for (std::size_t j = 0; j < stiffnessDiagonal.size(); ++j)
        scale[j] = scaleOfDiagonalTerm(stiffnessDiagonal[j]);
    return scale;
}

void ScaledFrontSource::addTerms(FrontTerms& front)
{
    m_inner.addTerms(front);
    // The equation of each position of the front: its columns, then its rows.
    const std::int32_t order = front.order();
    const auto scaleAt = [&](std::int32_t position)
    {
        const std::int32_t equation = position < front.columnCount
                                          ? front.firstColumn + position
                                          : front.rows[static_cast<std::size_t>(position - front.columnCount)];
        return m_scale[static_cast<std::size_t>(equation)];
    };
    for (std::int32_t j = 0; j < order; ++j)
    {
        const double scaleJ = scaleAt(j);
        for (std::int32_t i = j; i < order; ++i)
        {
            double& term = front.at(i, j);
            term = scaleAt(i) * term * scaleJ;
        }
    }
}

void ScaledFrontSource::diagonal(std::int32_t supernode, std::span<double> values) const
{
    m_inner.diagonal(supernode, values);
    const auto first = static_cast<std::size_t>(m_symbolic.firstColumn(supernode));
    for (std::size_t k = 0; k < values.size(); ++k)
        values[k] = m_scale[first + k] * values[k] * m_scale[first + k];
}

base::BudgetedBuffer<double> pencilScaleOf(const SymmetricCsc& stiffness, base::MemoryBudget& budget)
{
    const auto dimension = static_cast<std::size_t>(stiffness.dimension());
    base::BudgetedBuffer<double> scale(budget, dimension, "eigenvalue pencil: scales");
    const auto offsets = stiffness.columnOffsets();
    const auto rows = stiffness.rowIndices();
    const auto values = stiffness.values();
    for (std::size_t j = 0; j < dimension; ++j)
    {
        // The rows of a column are ascending from its diagonal on: the diagonal is its first term when stored.
        const auto first = static_cast<std::size_t>(offsets[j]);
        const bool hasDiagonal = offsets[j] < offsets[j + 1] && static_cast<std::size_t>(rows[first]) == j;
        scale[j] = scaleOfDiagonalTerm(hasDiagonal ? values[first] : 0.0);
    }
    return scale;
}

void scaleRows(DenseMatrixView block, std::span<const double> scale)
{
    for (std::size_t j = 0; j < block.columns; ++j)
        for (std::size_t i = 0; i < block.rows; ++i)
            block(i, j) *= scale[i];
}

} // namespace oos::linalg
