// The checks of the edges of the block Lanczos solver (linalg::BlockLanczos) on small diagonal pencils, whose roots
// are known: roots hidden by the locked ones at a shift are not taken for an exhausted space, a pencil with fewer
// finite roots than asked is still complete, a buckling bound that sits on a root is moved, and a root found below a
// Sturm point that counts none is refused.

#include "CheckRunner.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/DensePencil.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

using oos::linalg::BlockLanczos;
using oos::linalg::DenseKernels;
using oos::linalg::DenseMatrix;
using oos::linalg::DensePencil;
using oos::linalg::EigenRequest;
using oos::linalg::EigenSolution;
using oos::linalg::KernelKind;
using oos::linalg::SpectralTransform;

namespace
{

constexpr std::uint64_t BUDGET_BYTES = std::uint64_t{1} << 28;
// The relative agreement asked of a root of a diagonal pencil (the solver converges them far below).
constexpr double ROOT_TOLERANCE = 1.0e-9;

// The pencil (diag(stiffness), diag(rightHand)): its roots are stiffness[i] / rightHand[i], infinite where the
// right-hand term is zero.
DenseMatrix diagonalMatrix(const std::vector<double>& diagonal, oos::base::MemoryBudget& budget)
{
    DenseMatrix matrix(budget, diagonal.size(), diagonal.size(), "check: diagonal matrix");
    for (std::size_t i = 0; i < diagonal.size(); ++i)
        matrix(i, i) = diagonal[i];
    return matrix;
}

// A pencil whose first factorisation reports no negative pivot whatever it found: the unreliable Sturm count of a
// pencil close to singular, made on purpose.
class MiscountingPencil final : public oos::linalg::ShiftedPencil
{
public:
    MiscountingPencil(const std::vector<double>& stiffness, const DenseMatrix& rightHand, const DenseKernels& kernels,
                      oos::base::MemoryBudget& budget)
        : m_pencil(stiffness, rightHand.view(), SpectralTransform::Vibration, kernels, budget)
    {
    }

    [[nodiscard]] std::int32_t dimension() const noexcept override { return m_pencil.dimension(); }
    [[nodiscard]] oos::linalg::ShiftFactorisation factorise(double shift) override
    {
        oos::linalg::ShiftFactorisation factorisation = m_pencil.factorise(shift);
        if (m_isFirst)
        {
            factorisation.inertia.positive += factorisation.inertia.negative;
            factorisation.inertia.negative = 0;
            m_isFirst = false;
        }
        return factorisation;
    }
    void solve(oos::linalg::DenseMatrixView block) const override { m_pencil.solve(block); }
    void multiplyMass(oos::linalg::ConstDenseMatrixView x, oos::linalg::DenseMatrixView y) const override
    {
        m_pencil.multiplyMass(x, y);
    }
    [[nodiscard]] double diagonalRayleighBound() const override { return m_pencil.diagonalRayleighBound(); }

private:
    DensePencil m_pencil;
    bool m_isFirst = true;
};

} // namespace

CHECK_CASE("block Lanczos: roots hidden by the locked ones at a shift are found, not taken for an exhausted space")
{
    // Six roots at 1 and eighteen from 1e20 up. Once the six are locked, the next shift (1.5) is 1e20 times nearer
    // them than the others: Op applied to a random block is the locked vectors to rounding, and every start column
    // deflates. The eight lowest roots asked are the six and the first two of the far ones.
    constexpr std::size_t NEAR_ROOTS = 6;
    constexpr std::size_t FAR_ROOTS = 18;
    constexpr double FAR_ROOT_STEP = 1.0e20;
    constexpr std::size_t ASKED = 8;
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    std::vector<double> stiffness(NEAR_ROOTS, 1.0);
    for (std::size_t i = 1; i <= FAR_ROOTS; ++i)
        stiffness.push_back(FAR_ROOT_STEP * static_cast<double>(i));
    const DenseMatrix identity = diagonalMatrix(std::vector<double>(stiffness.size(), 1.0), budget);
    DensePencil pencil(stiffness, identity.view(), SpectralTransform::Vibration, kernels, budget);
    EigenRequest request;
    request.modeCount = ASKED;
    try
    {
        const EigenSolution solution = BlockLanczos().solve(pencil, request, kernels, budget);
        runner.expect(solution.isComplete, "the request is met");
        runner.expect(solution.eigenvalues.size() == ASKED,
                      std::to_string(solution.eigenvalues.size()) + " roots found, expected " + std::to_string(ASKED));
        for (std::size_t k = 0; k < solution.eigenvalues.size() && k < ASKED; ++k)
        {
            const double expected = k < NEAR_ROOTS ? 1.0 : FAR_ROOT_STEP * static_cast<double>(k - NEAR_ROOTS + 1);
            runner.expectNear(solution.eigenvalues[k] / expected, 1.0, ROOT_TOLERANCE, "root " + std::to_string(k));
        }
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the solve threw: ") + error.what());
    }
}

CHECK_CASE("block Lanczos: a pencil with fewer finite roots than asked is complete with those it has")
{
    // Four DOFs with mass (roots 1 to 4) and six without (infinite roots): eight roots asked, four exist. The probe
    // of the deflated operator finds nothing beyond the four locked vectors: the space is exhausted.
    constexpr std::size_t DIMENSION = 10;
    constexpr std::size_t FINITE_ROOTS = 4;
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    std::vector<double> stiffness;
    std::vector<double> mass(DIMENSION, 0.0);
    for (std::size_t i = 0; i < DIMENSION; ++i)
        stiffness.push_back(static_cast<double>(i + 1));
    for (std::size_t i = 0; i < FINITE_ROOTS; ++i)
        mass[i] = 1.0;
    const DenseMatrix massMatrix = diagonalMatrix(mass, budget);
    DensePencil pencil(stiffness, massMatrix.view(), SpectralTransform::Vibration, kernels, budget);
    EigenRequest request;
    request.modeCount = 8;
    try
    {
        const EigenSolution solution = BlockLanczos().solve(pencil, request, kernels, budget);
        runner.expect(solution.isComplete, "the request is met by the roots the pencil has");
        runner.expect(solution.eigenvalues.size() == FINITE_ROOTS,
                      std::to_string(solution.eigenvalues.size()) + " roots found, expected " +
                          std::to_string(FINITE_ROOTS));
        for (std::size_t k = 0; k < solution.eigenvalues.size() && k < FINITE_ROOTS; ++k)
            runner.expectNear(solution.eigenvalues[k], static_cast<double>(k + 1), ROOT_TOLERANCE,
                              "root " + std::to_string(k));
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the solve threw: ") + error.what());
    }
}

CHECK_CASE("block Lanczos: a buckling range whose upper bound sits on a root is solved")
{
    // K = I and B = diag(1/λ): the load factors 1, 2, 4, ... 2048. K − 2·B has an exactly zero pivot (1 − 2·0.5), so
    // the Sturm point of the upper bound 2 is moved a little beyond it; the range [0.5, 2] holds the root 1 and, on
    // its bound, the root 2.
    constexpr std::size_t DIMENSION = 12;
    constexpr double LOWER_BOUND = 0.5;
    constexpr double UPPER_BOUND = 2.0;
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    const std::vector<double> stiffness(DIMENSION, 1.0);
    std::vector<double> inverseRoots;
    double root = 1.0;
    for (std::size_t i = 0; i < DIMENSION; ++i, root *= 2.0)
        inverseRoots.push_back(1.0 / root);
    const DenseMatrix rightHand = diagonalMatrix(inverseRoots, budget);
    DensePencil pencil(stiffness, rightHand.view(), SpectralTransform::Buckling, kernels, budget);
    EigenRequest request;
    request.lowerBound = LOWER_BOUND;
    request.upperBound = UPPER_BOUND;
    try
    {
        const EigenSolution solution = BlockLanczos().solve(pencil, request, kernels, budget);
        runner.expect(solution.isComplete, "the range is verified");
        // The root on the bound is in or out of the range by its last bit: one or two roots.
        runner.expect(solution.eigenvalues.size() == 1 || solution.eigenvalues.size() == 2,
                      std::to_string(solution.eigenvalues.size()) + " roots found, expected 1 or 2");
        if (!solution.eigenvalues.empty())
            runner.expectNear(solution.eigenvalues.front(), 1.0, ROOT_TOLERANCE, "the first load factor");
        if (solution.eigenvalues.size() == 2)
            runner.expectNear(solution.eigenvalues.back(), UPPER_BOUND, ROOT_TOLERANCE, "the load factor on the bound");
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the solve threw: ") + error.what());
    }
}

CHECK_CASE("block Lanczos: a root found below a Sturm point that counts none is refused")
{
    // The roots −5 and 1 to 11; the factorisation below the spectrum (at −1e-6) wrongly counts no root below it. The
    // run there converges every root, −5 among them: the solve must not return it as verified.
    constexpr std::size_t DIMENSION = 12;
    constexpr double NEGATIVE_ROOT = -5.0;
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const DenseKernels kernels(KernelKind::Reference);
    std::vector<double> stiffness{NEGATIVE_ROOT};
    for (std::size_t i = 1; i < DIMENSION; ++i)
        stiffness.push_back(static_cast<double>(i));
    const DenseMatrix identity = diagonalMatrix(std::vector<double>(DIMENSION, 1.0), budget);
    MiscountingPencil pencil(stiffness, identity, kernels, budget);
    EigenRequest request;
    request.modeCount = DIMENSION;
    bool isRefused = false;
    try
    {
        const EigenSolution solution = BlockLanczos().solve(pencil, request, kernels, budget);
        static_cast<void>(solution);
    }
    catch (const oos::base::SolverError& error)
    {
        isRefused = std::string(error.what()).find("below the shift") != std::string::npos;
        runner.expect(isRefused, std::string("another error than the root below the Sturm point: ") + error.what());
    }
    catch (const std::exception& error)
    {
        runner.expect(false, std::string("the solve threw another kind of error: ") + error.what());
    }
    runner.expect(isRefused, "the root below the lowest Sturm point is refused with a SolverError");
}
