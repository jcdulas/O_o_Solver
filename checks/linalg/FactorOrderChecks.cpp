// The checks of the order of the tasks of the supernodal factorisation (linalg::FactorOrder): the postorder gives the
// bits of the parallel order with a smaller working storage, and the automatic order takes it when the parallel one
// does not fit the memory.

#include "CheckRunner.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/linalg/ordering/ApproximateMinimumDegree.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

using oos::linalg::DenseKernels;
using oos::linalg::DenseMatrix;
using oos::linalg::FactorOptions;
using oos::linalg::FactorOrder;
using oos::linalg::KernelKind;
using oos::linalg::NodeGraph;
using oos::linalg::PivotingMode;
using oos::linalg::SupernodalFactor;
using oos::linalg::SymbolicFactor;
using oos::linalg::SymmetricCsc;

namespace
{

// A grid of SIDE × SIDE scalar nodes, each next to its four neighbours: its tree of supernodes branches at every
// level of the ordering, so the factorisation cuts it into many subtrees.
constexpr std::int32_t SIDE = 220;
constexpr std::int32_t NODE_COUNT = SIDE * SIDE;
constexpr std::size_t COLUMNS = 2;
constexpr std::uint64_t MODEL_BUDGET_BYTES = std::uint64_t{1} << 30;

std::int32_t nodeAt(std::int32_t i, std::int32_t j) noexcept
{
    return i + SIDE * j;
}

SymbolicFactor analyseGrid(oos::base::MemoryBudget& budget)
{
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> neighbours;
    for (std::int32_t j = 0; j < SIDE; ++j)
        for (std::int32_t i = 0; i < SIDE; ++i)
        {
            if (j > 0)
                neighbours.push_back(nodeAt(i, j - 1));
            if (i > 0)
                neighbours.push_back(nodeAt(i - 1, j));
            if (i + 1 < SIDE)
                neighbours.push_back(nodeAt(i + 1, j));
            if (j + 1 < SIDE)
                neighbours.push_back(nodeAt(i, j + 1));
            offsets.push_back(static_cast<std::int64_t>(neighbours.size()));
        }
    const NodeGraph graph = NodeGraph::fromAdjacency(NODE_COUNT, offsets, neighbours, budget);
    const std::vector<std::int32_t> weights(static_cast<std::size_t>(NODE_COUNT), 1);
    const oos::linalg::NodePermutation order =
        oos::linalg::ApproximateMinimumDegree().order(graph, weights, {}, budget);
    return SymbolicFactor::analyse(graph, weights, order, 0, oos::linalg::AmalgamationThresholds{}, budget);
}

// The matrix of the grid in the equations of its symbolic factor: `diagonal` at every node, −1 between neighbours
// (4.5: positive definite; 3.3: indefinite, the shifted matrix of an eigenvalue solver).
SymmetricCsc gridMatrix(const SymbolicFactor& symbolic, double diagonal, oos::base::MemoryBudget& budget)
{
    const auto equationOf = [&](std::int32_t node) { return symbolic.nodeOrder().positionOf(node); };
    std::vector<std::vector<std::pair<std::int32_t, double>>> columns(static_cast<std::size_t>(NODE_COUNT));
    const auto couple = [&](std::int32_t a, std::int32_t b)
    {
        const std::int32_t first = equationOf(a);
        const std::int32_t second = equationOf(b);
        columns[static_cast<std::size_t>(std::min(first, second))].emplace_back(std::max(first, second), -1.0);
    };
    for (std::int32_t j = 0; j < SIDE; ++j)
        for (std::int32_t i = 0; i < SIDE; ++i)
        {
            const std::int32_t equation = equationOf(nodeAt(i, j));
            columns[static_cast<std::size_t>(equation)].emplace_back(equation, diagonal);
            if (i + 1 < SIDE)
                couple(nodeAt(i, j), nodeAt(i + 1, j));
            if (j + 1 < SIDE)
                couple(nodeAt(i, j), nodeAt(i, j + 1));
        }
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> rows;
    std::vector<double> values;
    for (std::vector<std::pair<std::int32_t, double>>& column : columns)
    {
        std::sort(column.begin(), column.end());
        for (const auto& [row, value] : column)
        {
            rows.push_back(row);
            values.push_back(value);
        }
        offsets.push_back(static_cast<std::int64_t>(rows.size()));
    }
    return SymmetricCsc::fromCompressedColumns(NODE_COUNT, offsets, rows, values, budget);
}

// What one factorisation and solve gave: the solution, the working storage of the factorisation and the peak of its
// budget.
struct Run
{
    std::vector<double> solution;
    oos::linalg::FactorWorkspaceUsage usage;
    std::uint64_t peakBytes = 0;
};

Run factoriseAndSolve(const SymbolicFactor& symbolic, const SymmetricCsc& matrix, PivotingMode pivoting,
                      FactorOrder order, std::uint64_t budgetBytes)
{
    oos::base::MemoryBudget budget(budgetBytes);
    const DenseKernels kernels(KernelKind::Reference);
    FactorOptions options;
    options.pivoting = pivoting;
    options.order = order;
    Run run;
    const SupernodalFactor factor = SupernodalFactor::factorise(symbolic, matrix, kernels, options, budget);
    run.usage = factor.workspaceUsage();
    // The peak of the factorisation alone: the right-hand sides come after it.
    run.peakBytes = budget.peak();
    DenseMatrix block(budget, static_cast<std::size_t>(NODE_COUNT), COLUMNS, "check: right-hand sides");
    for (std::size_t j = 0; j < COLUMNS; ++j)
        for (std::size_t i = 0; i < block.rows(); ++i)
            block(i, j) = static_cast<double>((7 * i + 3 * j) % 11) - 5.0;
    oos::linalg::FactorSolve(factor, kernels).solve(block.view());
    for (std::size_t j = 0; j < COLUMNS; ++j)
        run.solution.insert(run.solution.end(), &block(0, j), &block(0, j) + block.rows());
    return run;
}

bool haveSameBits(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

} // namespace

CHECK_CASE("supernodal factor: the postorder has the bits of the parallel order and a smaller working storage")
{
    struct Mode
    {
        const char* name;
        PivotingMode pivoting;
        double diagonal;
    };
    for (const Mode& mode : {Mode{"positive definite", PivotingMode::PositiveDefinite, 4.5},
                             Mode{"indefinite", PivotingMode::Indefinite, 3.3}})
    {
        const std::string name = mode.name;
        try
        {
            oos::base::MemoryBudget modelBudget(MODEL_BUDGET_BYTES);
            const SymbolicFactor symbolic = analyseGrid(modelBudget);
            const SymmetricCsc matrix = gridMatrix(symbolic, mode.diagonal, modelBudget);
            const Run parallel =
                factoriseAndSolve(symbolic, matrix, mode.pivoting, FactorOrder::Parallel, MODEL_BUDGET_BYTES);
            const Run postorder =
                factoriseAndSolve(symbolic, matrix, mode.pivoting, FactorOrder::Postorder, MODEL_BUDGET_BYTES);
            runner.expect(!parallel.usage.isPostorder && postorder.usage.isPostorder, name + ": the orders asked");
            runner.expect(haveSameBits(parallel.solution, postorder.solution),
                          name + ": the solve of the postorder has the bits of the parallel order's");
            runner.expect(postorder.usage.boundValues < parallel.usage.boundValues,
                          name + ": the postorder bounds " + std::to_string(postorder.usage.boundValues) +
                              " values of working storage, the parallel order " +
                              std::to_string(parallel.usage.boundValues));
            runner.expect(postorder.usage.peakValues <= postorder.usage.boundValues,
                          name + ": the working storage of the postorder stays within its bound");
            runner.expect(postorder.peakBytes < parallel.peakBytes, name + ": the postorder takes less memory");

            // With a memory between the two, the automatic order takes the postorder and gives the same bits; with
            // the memory of the parallel order it keeps that one.
            const Run roomy =
                factoriseAndSolve(symbolic, matrix, mode.pivoting, FactorOrder::Automatic, MODEL_BUDGET_BYTES);
            runner.expect(!roomy.usage.isPostorder, name + ": the automatic order is the parallel one when it fits");
            const std::uint64_t between = postorder.peakBytes + (parallel.peakBytes - postorder.peakBytes) / 2;
            const Run tight = factoriseAndSolve(symbolic, matrix, mode.pivoting, FactorOrder::Automatic, between);
            runner.expect(tight.usage.isPostorder, name + ": the automatic order is the postorder when short");
            runner.expect(haveSameBits(parallel.solution, tight.solution),
                          name + ": the automatic order short of memory has the same bits");
        }
        catch (const std::exception& error)
        {
            runner.expect(false, name + ": the check threw: " + error.what());
        }
    }
}
