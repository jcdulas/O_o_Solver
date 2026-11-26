// The checks of two edges of the supernodal factorisation: the ratio test flags a negative pivot whatever the sign of
// its matrix diagonal (linalg::SupernodalFactor, positive definite mode), and the solve of a factor out of core has the
// bits of the in-core solve wherever its panels are kept, a scratch file read without the system's cache included
// (linalg::OutOfCoreFactor).

#include "CheckRunner.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"
#include "oos/linalg/supernodal/OutOfCoreFactor.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using oos::linalg::DenseKernels;
using oos::linalg::DenseMatrix;
using oos::linalg::FactorOptions;
using oos::linalg::KernelKind;
using oos::linalg::NodeGraph;
using oos::linalg::NodePermutation;
using oos::linalg::PivotingMode;
using oos::linalg::PivotIssueKind;
using oos::linalg::SupernodalFactor;
using oos::linalg::SymbolicFactor;
using oos::linalg::SymmetricCsc;

namespace
{

constexpr std::uint64_t BUDGET_BYTES = std::uint64_t{1} << 28;

// A chain of `nodeCount` scalar nodes (node v next to v − 1 and v + 1), analysed in its own order.
SymbolicFactor analyseChain(std::int32_t nodeCount, oos::base::MemoryBudget& budget)
{
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> neighbours;
    for (std::int32_t v = 0; v < nodeCount; ++v)
    {
        if (v > 0)
            neighbours.push_back(v - 1);
        if (v + 1 < nodeCount)
            neighbours.push_back(v + 1);
        offsets.push_back(static_cast<std::int64_t>(neighbours.size()));
    }
    const NodeGraph graph = NodeGraph::fromAdjacency(nodeCount, offsets, neighbours, budget);
    const std::vector<std::int32_t> weights(static_cast<std::size_t>(nodeCount), 1);
    return SymbolicFactor::analyse(graph, weights, NodePermutation::identity(nodeCount, budget), 0,
                                   oos::linalg::AmalgamationThresholds{}, budget);
}

// The tridiagonal matrix of the chain in the equations of its symbolic factor (one equation a node, at the node's
// position): diagonal[v] at node v, coupling[v] between the nodes v and v + 1.
SymmetricCsc chainMatrix(const SymbolicFactor& symbolic, const std::vector<double>& diagonal,
                         const std::vector<double>& coupling, oos::base::MemoryBudget& budget)
{
    const std::size_t count = diagonal.size();
    const auto equationOf = [&](std::size_t node)
    { return symbolic.nodeOrder().positionOf(static_cast<std::int32_t>(node)); };
    // Per column, its terms of the lower triangle as (row, value).
    std::vector<std::vector<std::pair<std::int32_t, double>>> columns(count);
    for (std::size_t v = 0; v < count; ++v)
    {
        const std::int32_t equation = equationOf(v);
        columns[static_cast<std::size_t>(equation)].emplace_back(equation, diagonal[v]);
        if (v + 1 == count)
            continue;
        const std::int32_t next = equationOf(v + 1);
        columns[static_cast<std::size_t>(std::min(equation, next))].emplace_back(std::max(equation, next),
                                                                                coupling[v]);
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
    return SymmetricCsc::fromCompressedColumns(static_cast<std::int32_t>(count), offsets, rows, values, budget);
}

// A folder of its own under the temporary folder of the machine, removed with what it holds when the check ends.
class ScratchFolder
{
public:
    ScratchFolder()
    {
        std::random_device device;
        m_path = std::filesystem::temp_directory_path() /
                 ("oos-check-out-of-core-" + std::to_string(device()) + "-" + std::to_string(device()));
        std::filesystem::create_directories(m_path);
    }
    ScratchFolder(const ScratchFolder&) = delete;
    ScratchFolder& operator=(const ScratchFolder&) = delete;
    ~ScratchFolder()
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

private:
    std::filesystem::path m_path;
};

// The pivot issues of the positive definite factorisation of the 2 × 2 matrix [[2, 1], [1, lastDiagonal]], the run
// going on whatever they are (a negative bail-out).
oos::linalg::PivotDiagnostics issuesOfTwoByTwo(double lastDiagonal, std::size_t& negativePivots)
{
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    const SymbolicFactor symbolic = analyseChain(2, budget);
    // The diagonals by equation: the equation eliminated last takes lastDiagonal, whichever node it is.
    std::vector<double> diagonal(2, 2.0);
    diagonal[static_cast<std::size_t>(symbolic.nodeOrder().nodeAt(1))] = lastDiagonal;
    const SymmetricCsc matrix = chainMatrix(symbolic, diagonal, {1.0}, budget);
    FactorOptions options;
    options.pivotPolicy.bailout = -1;
    const SupernodalFactor factor = SupernodalFactor::factorise(symbolic, matrix, kernels, options, budget);
    negativePivots = factor.inertia().negative;
    return factor.pivotDiagnostics();
}

} // namespace

CHECK_CASE("supernodal factor: a negative pivot is an issue of the ratio test whatever its matrix diagonal")
{
    // [[2, 1], [1, d]]: the second pivot is d − 1/2. With d = 0 its ratio d/(d − 1/2) is −0, with d = −1 it is
    // positive (2/3): neither is below zero, and both pivots are negative.
    for (const double lastDiagonal : {0.0, -1.0})
    {
        const std::string name = "last diagonal " + std::to_string(lastDiagonal);
        try
        {
            std::size_t negativePivots = 0;
            const oos::linalg::PivotDiagnostics diagnostics = issuesOfTwoByTwo(lastDiagonal, negativePivots);
            runner.expect(negativePivots == 1, name + ": one negative pivot");
            runner.expect(diagnostics.issueCount() == 1,
                          name + ": " + std::to_string(diagnostics.issueCount()) + " issues, expected 1");
            if (!diagnostics.issues().empty())
            {
                runner.expect(diagnostics.issues().front().kind == PivotIssueKind::NegativePivot,
                              name + ": the issue is a negative pivot");
                runner.expect(diagnostics.issues().front().equation == 1, name + ": at the last equation");
                runner.expect(diagnostics.issues().front().factorDiagonal < 0.0, name + ": its pivot is negative");
            }
        }
        catch (const std::exception& error)
        {
            runner.expect(false, name + ": the factorisation threw: " + error.what());
        }
    }
    // A positive definite matrix keeps its clean test.
    try
    {
        std::size_t negativePivots = 0;
        const oos::linalg::PivotDiagnostics diagnostics = issuesOfTwoByTwo(2.0, negativePivots);
        runner.expect(negativePivots == 0 && diagnostics.issueCount() == 0, "a positive definite matrix has no issue");
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the positive definite factorisation threw: ") + error.what());
    }
}

CHECK_CASE("out-of-core factor: the solve has the bits of the in-core solve wherever the panels are kept")
{
    // An indefinite tridiagonal matrix of several solve units (more than SolvePlan's 1 024 rows a unit), its panels in
    // spool blocks of 64 kB: the units' records start anywhere in the blocks and run over several of them.
    constexpr std::int32_t DIMENSION = 3000;
    constexpr std::size_t COLUMNS = 3;
    constexpr double DIAGONAL = 2.0 - 0.37;
    constexpr std::size_t BLOCK_BYTES = std::size_t{1} << 16;
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    try
    {
        const SymbolicFactor symbolic = analyseChain(DIMENSION, budget);
        const std::vector<double> diagonal(static_cast<std::size_t>(DIMENSION), DIAGONAL);
        const std::vector<double> coupling(static_cast<std::size_t>(DIMENSION) - 1, -1.0);
        const SymmetricCsc matrix = chainMatrix(symbolic, diagonal, coupling, budget);
        FactorOptions options;
        options.pivoting = PivotingMode::Indefinite;
        const auto fillRightHandSides = [&](DenseMatrix& block)
        {
            for (std::size_t j = 0; j < COLUMNS; ++j)
                for (std::size_t i = 0; i < block.rows(); ++i)
                    block(i, j) = static_cast<double>((7 * i + 3 * j) % 11) - 5.0;
        };
        DenseMatrix reference(budget, static_cast<std::size_t>(DIMENSION), COLUMNS, "check: in-core solve");
        fillRightHandSides(reference);
        {
            const SupernodalFactor factor = SupernodalFactor::factorise(symbolic, matrix, kernels, options, budget);
            runner.expect(!factor.isSingular(), "the matrix of the check is not singular");
            oos::linalg::FactorSolve(factor, kernels).solve(reference.view());
        }

        const ScratchFolder scratch;
        struct Place
        {
            const char* name;
            bool isOnDisk;
            bool isUnbuffered;
        };
        const std::vector<Place> places{{"in memory", false, false},
                                        {"on disk through the system's cache", true, false},
                                        {"on disk without the system's cache", true, true}};
        for (const Place& place : places)
        {
            oos::linalg::ScratchSpool::Options spool;
            spool.isOnDisk = place.isOnDisk;
            spool.folder = scratch.path();
            spool.isUnbuffered = place.isUnbuffered;
            spool.blockBytes = BLOCK_BYTES;
            oos::linalg::OutOfCoreFactor factor =
                oos::linalg::OutOfCoreFactor::factorise(symbolic, matrix, kernels, options, spool, budget);
            DenseMatrix solution(budget, static_cast<std::size_t>(DIMENSION), COLUMNS, "check: out-of-core solve");
            fillRightHandSides(solution);
            factor.solve(solution.view(), kernels);
            bool hasSameBits = true;
            for (std::size_t j = 0; j < COLUMNS; ++j)
                hasSameBits = hasSameBits && std::memcmp(&solution(0, j), &reference(0, j),
                                                         static_cast<std::size_t>(DIMENSION) * sizeof(double)) == 0;
            runner.expect(hasSameBits, std::string("panels ") + place.name);
        }
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the check threw: ") + error.what());
    }
}
