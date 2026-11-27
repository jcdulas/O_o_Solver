// The checks of the scaling of the shifted matrices of the eigenvalue solvers (linalg::PencilScaling): a pencil whose
// equations have units a thousand apart is factorised without a delayed pivot, its Sturm counts are those of the
// unscaled problem, and its solves are those of the matrix itself.

#include "CheckRunner.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/PencilScaling.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

using oos::linalg::DenseKernels;
using oos::linalg::DenseMatrix;
using oos::linalg::KernelKind;
using oos::linalg::NodeGraph;
using oos::linalg::NodePermutation;
using oos::linalg::SymbolicFactor;
using oos::linalg::SymmetricCsc;

namespace
{

// A chain of COUNT equations that alternate two units, as the deflections and the rotations of a beam do: with
// u_i = UNIT for an even equation and 1 for an odd one,
//   K = U·T·U,  M = U·U,  T = tridiag(−0.9, 2, −0.9),  U = diag(u_i),
// so that K − σM = U·(T − σI)·U: the roots of the pencil are those of T, 2 − 1.8 cos(kπ/(COUNT + 1)), whatever the
// units, and the term that couples two neighbours is UNIT times the diagonal term of the odd one.
constexpr std::int32_t COUNT = 400;
constexpr double UNIT = 1000.0;
constexpr double DIAGONAL = 2.0;
constexpr double COUPLING = -0.9;
constexpr std::uint64_t BUDGET_BYTES = std::uint64_t{1} << 28;

double unitOf(std::int32_t equation) noexcept
{
    return equation % 2 == 0 ? UNIT : 1.0;
}

SymbolicFactor analyseChain(oos::base::MemoryBudget& budget)
{
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> neighbours;
    for (std::int32_t v = 0; v < COUNT; ++v)
    {
        if (v > 0)
            neighbours.push_back(v - 1);
        if (v + 1 < COUNT)
            neighbours.push_back(v + 1);
        offsets.push_back(static_cast<std::int64_t>(neighbours.size()));
    }
    const NodeGraph graph = NodeGraph::fromAdjacency(COUNT, offsets, neighbours, budget);
    const std::vector<std::int32_t> weights(static_cast<std::size_t>(COUNT), 1);
    return SymbolicFactor::analyse(graph, weights, NodePermutation::identity(COUNT, budget), 0,
                                   oos::linalg::AmalgamationThresholds{}, budget);
}

// U·(diagonal·I + coupling·(the neighbours))·U on the chain, in its own order (the identity ordering).
SymmetricCsc chainMatrix(double diagonal, double coupling, oos::base::MemoryBudget& budget)
{
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> rows;
    std::vector<double> values;
    for (std::int32_t j = 0; j < COUNT; ++j)
    {
        rows.push_back(j);
        values.push_back(unitOf(j) * diagonal * unitOf(j));
        if (j + 1 < COUNT)
        {
            rows.push_back(j + 1);
            values.push_back(unitOf(j + 1) * coupling * unitOf(j));
        }
        offsets.push_back(static_cast<std::int64_t>(rows.size()));
    }
    return SymmetricCsc::fromCompressedColumns(COUNT, offsets, rows, values, budget);
}

// The roots of T below `shift`.
std::size_t rootsBelow(double shift)
{
    std::size_t count = 0;
    for (std::int32_t k = 1; k <= COUNT; ++k)
        if (DIAGONAL + 2.0 * COUPLING * std::cos(k * std::numbers::pi / (COUNT + 1)) < shift)
            ++count;
    return count;
}

} // namespace

CHECK_CASE("pencil scaling: equations of units a thousand apart are factorised without a delayed pivot")
{
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    try
    {
        const SymbolicFactor symbolic = analyseChain(budget);
        const SymmetricCsc stiffness = chainMatrix(DIAGONAL, COUPLING, budget);
        const SymmetricCsc mass = chainMatrix(1.0, 0.0, budget);

        // The scales are powers of two that bring the diagonal of K between 1/4 and 2.
        const oos::base::BudgetedBuffer<double> scale = oos::linalg::pencilScaleOf(stiffness, budget);
        bool areScalesRight = true;
        for (std::int32_t j = 0; j < COUNT; ++j)
        {
            const double s = scale[static_cast<std::size_t>(j)];
            int exponent = 0;
            const double scaled = s * (unitOf(j) * DIAGONAL * unitOf(j)) * s;
            areScalesRight = areScalesRight && std::frexp(s, &exponent) == 0.5 && scaled >= 0.25 && scaled < 2.0;
        }
        runner.expect(areScalesRight, "the scales are powers of two and the scaled diagonal is of the order of one");

        oos::linalg::SparsePencil pencil(symbolic, stiffness, mass, kernels, oos::linalg::FactorOptions{}, budget);
        for (const double shift : {-1.0, 0.5, 1.0, 2.0, 3.5})
        {
            const std::string name = "shift " + std::to_string(shift);
            const oos::linalg::ShiftFactorisation made = pencil.factorise(shift);
            runner.expect(made.sturmCount() == rootsBelow(shift),
                          name + ": " + std::to_string(made.sturmCount()) + " roots counted below, expected " +
                              std::to_string(rootsBelow(shift)));
            runner.expect(pencil.factor().delayedPivotCount() == 0,
                          name + ": " + std::to_string(pencil.factor().delayedPivotCount()) + " delayed pivots");

            // (K − σM)·x = b for the x of the solve: the residual against b, equation by equation in its own unit.
            const auto n = static_cast<std::size_t>(COUNT);
            DenseMatrix block(budget, n, 1, "check: right-hand side");
            std::vector<double> rightHand(n);
            for (std::size_t i = 0; i < n; ++i)
                block(i, 0) = rightHand[i] = unitOf(static_cast<std::int32_t>(i)) * (static_cast<double>(i % 7) - 3.0);
            pencil.solve(block.view());
            const std::vector<double> x(&block(0, 0), &block(0, 0) + n);
            std::vector<double> kx(n);
            std::vector<double> mx(n);
            stiffness.multiply(x, kx);
            mass.multiply(x, mx);
            double worst = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                worst = std::max(worst, std::abs(kx[i] - shift * mx[i] - rightHand[i]) /
                                            unitOf(static_cast<std::int32_t>(i)));
            runner.expect(worst < 1.0e-8, name + ": the residual of the solve is " + std::to_string(worst));
        }
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the check threw: ") + error.what());
    }
}
