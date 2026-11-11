#include "oos/linalg/eigen/BorderedEigenSystem.hpp"

#include "../dense/LdltArithmetic.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace oos::linalg
{
namespace
{

NodeGraph borderedGraph(const SymmetricCsc& matrix, base::MemoryBudget& budget)
{
    const auto n = static_cast<std::size_t>(matrix.dimension());
    base::BudgetedBuffer<std::int64_t> offsets(budget, n + 2, "eigenvector border graph offsets");
    std::fill(offsets.span().begin(), offsets.span().end(), 0);
    const auto starts = matrix.columnOffsets();
    const auto rows = matrix.rowIndices();
    for (std::size_t j = 0; j < n; ++j)
    {
        ++offsets[j + 1];
        ++offsets[n + 1];
        for (std::int64_t k = starts[j]; k < starts[j + 1]; ++k)
        {
            const auto i = static_cast<std::size_t>(rows[static_cast<std::size_t>(k)]);
            if (i != j)
            {
                ++offsets[i + 1];
                ++offsets[j + 1];
            }
        }
    }
    for (std::size_t j = 1; j < offsets.size(); ++j)
        offsets[j] += offsets[j - 1];
    base::BudgetedBuffer<std::int32_t> neighbours(budget, static_cast<std::size_t>(offsets[n + 1]),
                                                 "eigenvector border graph edges");
    base::BudgetedBuffer<std::int64_t> next(budget, n + 1, "eigenvector border graph cursors");
    std::copy_n(offsets.data(), n + 1, next.data());
    for (std::size_t j = 0; j < n; ++j)
    {
        neighbours[static_cast<std::size_t>(next[j]++)] = static_cast<std::int32_t>(n);
        neighbours[static_cast<std::size_t>(next[n]++)] = static_cast<std::int32_t>(j);
        for (std::int64_t k = starts[j]; k < starts[j + 1]; ++k)
        {
            const auto i = static_cast<std::size_t>(rows[static_cast<std::size_t>(k)]);
            if (i != j)
            {
                neighbours[static_cast<std::size_t>(next[j]++)] = static_cast<std::int32_t>(i);
                neighbours[static_cast<std::size_t>(next[i]++)] = static_cast<std::int32_t>(j);
            }
        }
    }
    for (std::size_t j = 0; j <= n; ++j)
        std::sort(neighbours.data() + offsets[j], neighbours.data() + offsets[j + 1]);
    return NodeGraph::fromAdjacency(static_cast<std::int32_t>(n + 1), offsets.span(), neighbours.span(), budget);
}

struct Entry
{
    std::int32_t row;
    double value;
};

// A truncated extraction can hide a repeated partner. Its pencil residual leaves a tiny, rather than exactly
// zero, pivot in the border. Compare pivots with their original K + |lambda| M diagonal scale, including 2x2
// blocks after pivoting. This costs O(n), without extracting extra roots or factorising a second pencil.
bool hasUnresolvedPivot(const SupernodalFactor& factor, std::span<const double> scale)
{
    constexpr double RELATIVE_PIVOT_TOLERANCE = 1e-10;
    for (std::int32_t node = 0; node < factor.supernodeCount(); ++node)
    {
        const auto panel = factor.panel(node);
        const auto pivots = panel.pivots();
        for (std::size_t i = 0; i < pivots.size; ++i)
        {
            const double firstScale = scale[static_cast<std::size_t>(panel.rows[i])];
            const double a = pivots.diagonal[i * pivots.diagonalStride] / firstScale;
            double smallest = std::abs(a);
            if (startsTwoByTwoBlock(pivots, i))
            {
                const double secondScale = scale[static_cast<std::size_t>(panel.rows[i + 1])];
                const double b = pivots.subdiagonal[i] / std::sqrt(firstScale) / std::sqrt(secondScale);
                const double c = pivots.diagonal[(i + 1) * pivots.diagonalStride] / secondScale;
                const double largest = 0.5 * (std::abs(a + c) + std::hypot(a - c, 2.0 * b));
                smallest = largest > 0.0 ? std::abs(a * c - b * b) / largest : 0.0;
                ++i;
            }
            if (!std::isfinite(smallest) || smallest <= RELATIVE_PIVOT_TOLERANCE)
                return true;
        }
    }
    return false;
}

base::BudgetedBuffer<double> pivotScales(const SymmetricCsc& stiffness, const SymmetricCsc& mass, double root,
                                        const NodePermutation& order, double borderScale, base::MemoryBudget& budget)
{
    const auto n = static_cast<std::size_t>(stiffness.dimension());
    base::BudgetedBuffer<double> scales(budget, n + 1, "modal derivative pivot scales");
    std::fill(scales.span().begin(), scales.span().end(), borderScale);
    const auto starts = stiffness.columnOffsets();
    for (std::size_t j = 0; j < n; ++j)
        for (std::int64_t k = starts[j]; k < starts[j + 1]; ++k)
        {
            const auto index = static_cast<std::size_t>(k);
            if (stiffness.rowIndices()[index] == static_cast<std::int32_t>(j))
            {
                const double value = std::abs(stiffness.values()[index]) + std::abs(root * mass.values()[index]);
                if (value > 0.0)
                    scales[static_cast<std::size_t>(order.positionOf(static_cast<std::int32_t>(j)))] = value;
                break;
            }
        }
    return scales;
}

SymmetricCsc borderedMatrix(const SymmetricCsc& stiffness, const SymmetricCsc& mass, double root,
                           std::span<const double> border, const NodePermutation& order, base::MemoryBudget& budget)
{
    const auto n = static_cast<std::size_t>(stiffness.dimension());
    const auto count = static_cast<std::size_t>(stiffness.nonZeroCount()) + n + 1;
    base::BudgetedBuffer<std::int64_t> offsets(budget, n + 2, "eigenvector border column offsets");
    base::BudgetedBuffer<Entry> entries(budget, count, "eigenvector border entries");
    std::fill(offsets.span().begin(), offsets.span().end(), 0);
    const auto visit = [&](const auto& add)
    {
        const auto starts = stiffness.columnOffsets();
        for (std::size_t j = 0; j < n; ++j)
        {
            const auto column = order.positionOf(static_cast<std::int32_t>(j));
            for (std::int64_t k = starts[j]; k < starts[j + 1]; ++k)
            {
                const auto index = static_cast<std::size_t>(k);
                const auto row = order.positionOf(stiffness.rowIndices()[index]);
                const double value = stiffness.values()[index] - root * mass.values()[index];
                add(std::min(row, column), std::max(row, column), value);
            }
            add(column, static_cast<std::int32_t>(n), border[j]);
        }
        add(static_cast<std::int32_t>(n), static_cast<std::int32_t>(n), 0.0);
    };
    visit([&](std::int32_t column, std::int32_t, double) { ++offsets[static_cast<std::size_t>(column) + 1]; });
    for (std::size_t j = 1; j < offsets.size(); ++j)
        offsets[j] += offsets[j - 1];
    base::BudgetedBuffer<std::int64_t> next(budget, n + 1, "eigenvector border column cursors");
    std::copy_n(offsets.data(), n + 1, next.data());
    visit([&](std::int32_t column, std::int32_t row, double value)
          { entries[static_cast<std::size_t>(next[static_cast<std::size_t>(column)]++)] = {row, value}; });
    for (std::size_t j = 0; j <= n; ++j)
        std::sort(entries.data() + offsets[j], entries.data() + offsets[j + 1],
                  [](const Entry& a, const Entry& b) { return a.row < b.row; });
    base::BudgetedBuffer<std::int32_t> rows(budget, count, "eigenvector border rows");
    base::BudgetedBuffer<double> values(budget, count, "eigenvector border values");
    for (std::size_t k = 0; k < count; ++k)
    {
        rows[k] = entries[k].row;
        values[k] = entries[k].value;
    }
    return SymmetricCsc::fromCompressedColumns(static_cast<std::int32_t>(n + 1), offsets.span(), rows.span(),
                                               values.span(), budget);
}

} // namespace

BorderedEigenSystem::BorderedEigenSystem(const SymmetricCsc& stiffness, const SymmetricCsc& mass, double root,
                                       std::span<const double> shape, const DenseKernels& kernels,
                                       base::MemoryBudget& budget)
    : m_kernels(kernels), m_budget(budget)
{
    const auto n = static_cast<std::size_t>(stiffness.dimension());
    if (!stiffness.hasSameStructure(mass) || shape.size() != n || n == 0 ||
        n >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("eigenvector derivatives require matching nonempty stiffness, mass and shape");
    base::BudgetedBuffer<double> border(budget, n, "eigenvector normalisation border");
    mass.multiply(shape, border.span());
    double largestBorder = 0.0, largestMatrix = 0.0;
    for (const double value : border.span())
        largestBorder = std::max(largestBorder, std::abs(value));
    for (std::size_t k = 0; k < stiffness.values().size(); ++k)
        largestMatrix = std::max(largestMatrix, std::abs(stiffness.values()[k]) + std::abs(root * mass.values()[k]));
    if (!(largestBorder > 0.0))
        throw std::invalid_argument("eigenvector derivatives require a mode with positive generalised mass");
    // Balance the border against the pencil to avoid a small mass pivot beside a large stiffness pivot.
    m_borderScale = std::max(largestMatrix, largestBorder) / largestBorder;
    for (double& value : border.span())
        value *= m_borderScale;
    const NodeGraph graph = borderedGraph(stiffness, budget);
    base::BudgetedBuffer<std::int32_t> weights(budget, n + 1, "eigenvector border node weights");
    std::fill(weights.span().begin(), weights.span().end(), 1);
    m_symbolic = SymbolicFactor::analyse(graph, weights.span(), NodePermutation::identity(
        static_cast<std::int32_t>(n + 1), budget), 1, {}, budget);
    const SymmetricCsc matrix = borderedMatrix(stiffness, mass, root, border.span(), m_symbolic.nodeOrder(), budget);
    FactorOptions options;
    options.pivoting = PivotingMode::Indefinite;
    m_factor = SupernodalFactor::factorise(m_symbolic, matrix, kernels, options, budget);
    const auto scales = pivotScales(stiffness, mass, root, m_symbolic.nodeOrder(),
                                     std::max(largestMatrix, largestBorder), budget);
    if (m_factor.isSingular() || hasUnresolvedPivot(m_factor, scales.span()))
        throw base::SingularityError(
            "modal derivative border is numerically singular: the selected root is repeated or the pencil "
            "has a mechanism; extract the neighbouring roots to inspect their separation");
}

double BorderedEigenSystem::solve(std::span<const double> weights, std::span<double> adjoint,
                                  double normalisationRate) const
{
    const auto n = static_cast<std::size_t>(m_factor.dimension() - 1);
    if (weights.size() != n || adjoint.size() != n)
        throw std::invalid_argument("eigenvector adjoint has the wrong equation count");
    DenseMatrix right(m_budget, n + 1, 1, "eigenvector adjoint solve");
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto equation = static_cast<std::size_t>(m_symbolic.nodeOrder().positionOf(static_cast<std::int32_t>(i)));
        right(equation, 0) = weights[i];
    }
    right(n, 0) = m_borderScale * normalisationRate;
    FactorSolve(m_factor, m_kernels).solve(right.view());
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto equation = static_cast<std::size_t>(m_symbolic.nodeOrder().positionOf(static_cast<std::int32_t>(i)));
        adjoint[i] = right(equation, 0);
    }
    return m_borderScale * right(n, 0);
}

} // namespace oos::linalg
