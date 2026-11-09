#include "oos/linalg/eigen/BlockLanczos.hpp"

#include "EigenvectorStore.hpp"
#include "LanczosRun.hpp"
#include "SolveSupport.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/eigen/SturmCheck.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

constexpr double INFINITE = std::numeric_limits<double>::infinity();

static_assert(LanczosOptions::DEFAULT_INNER_PRODUCT_PIECE_ROWS == INNER_PRODUCT_PIECE_ROWS,
              "the public default of the pieces of the inner products is the orthogonalizer's");

// The largest tolerance accepted: above it a converged vector would not even be good to two digits.
constexpr double MAXIMUM_CONVERGENCE_TOLERANCE = 1.0e-2;

std::size_t roundUp(std::size_t value, std::size_t multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

} // namespace

BlockLanczos::BlockLanczos(const LanczosOptions& options) : m_options(options)
{
    if (options.blockSize == 0 || options.modesPerShift == 0)
        throw std::invalid_argument(base::concatenate({"BlockLanczos: the block size (",
                                                       std::to_string(options.blockSize), ") and the modes per shift (",
                                                       std::to_string(options.modesPerShift), ") must be at least 1"}));
    if (!(options.convergenceTolerance > 0.0 && options.convergenceTolerance <= MAXIMUM_CONVERGENCE_TOLERANCE))
        throw std::invalid_argument(
            base::concatenate({"BlockLanczos: the convergence tolerance must be in (0, 1e-2], found ",
                               std::to_string(options.convergenceTolerance)}));
    if (options.maximumShifts == 0 || options.maximumRestarts == 0)
        throw std::invalid_argument("BlockLanczos: at least one shift and one restart must be allowed");
    if (options.innerProductPieceRows == 0)
        throw std::invalid_argument("BlockLanczos: the rows of a piece of the inner products must be at least 1");
}

BlockLanczos::BasisShape BlockLanczos::shapeFor(std::size_t dimension, const EigenRequest& request) const noexcept
{
    BasisShape shape;
    if (dimension == 0)
        return shape;
    // At least three blocks fit in the space: the restart keeps one and adds two.
    shape.blockSize = std::max<std::size_t>(1, std::min(m_options.blockSize, dimension / 3));
    shape.modesPerShift =
        request.modeCount > 0 ? std::min(m_options.modesPerShift, request.modeCount) : m_options.modesPerShift;
    const std::size_t wanted = roundUp(2 * shape.modesPerShift + 3 * shape.blockSize, shape.blockSize);
    const std::size_t room =
        std::max(dimension / shape.blockSize * shape.blockSize, std::min<std::size_t>(dimension, 3));
    shape.basisColumns = std::min(wanted, room);
    return shape;
}

std::uint64_t BlockLanczos::estimateMemoryBytes(std::size_t dimension, const EigenRequest& request,
                                                std::size_t modeCount) const noexcept
{
    const BasisShape shape = shapeFor(dimension, request);
    const auto rows = static_cast<std::uint64_t>(DenseMatrix::alignedLeadingDimension(dimension));
    // The basis and its blocks, the eigenvector matrix of the projected problem (SmallEigenSolver, at most
    // basisColumns square) while the Ritz pairs are computed, and the orthogonalizer's sums by pieces.
    const std::uint64_t projected =
        static_cast<std::uint64_t>(DenseMatrix::alignedLeadingDimension(shape.basisColumns)) * shape.basisColumns *
        sizeof(double);
    const std::uint64_t workspace = LanczosWorkspace::estimateBytes(dimension, shape.basisColumns, shape.blockSize) +
                                    projected +
                                    MassOrthogonalizer::estimateBytes(dimension, m_options.innerProductPieceRows);
    const std::uint64_t store = rows * (modeCount + shape.basisColumns) * sizeof(double);
    const std::uint64_t result = rows * modeCount * sizeof(double);
    return store + std::max(workspace, result);
}

EigenSolution BlockLanczos::solve(ShiftedPencil& pencil, const EigenRequest& request, const DenseKernels& kernels,
                                  base::MemoryBudget& budget) const
{
    if (pencil.transform() == SpectralTransform::Buckling)
        return solveBuckling(pencil, request, kernels, budget);
    const ShiftStrategy strategy(request, pencil.diagonalRayleighBound());
    const auto dimension = static_cast<std::size_t>(pencil.dimension());
    const base::OperationControl& control = m_options.control;
    EigenSolution solution;
    if (dimension == 0)
    {
        solution.isComplete = true;
        return solution;
    }
    const BasisShape shape = shapeFor(dimension, request);
    std::optional<LanczosWorkspace> workspace;
    workspace.emplace(budget, dimension, shape.basisColumns, shape.blockSize);
    EigenvectorStore store(budget, dimension,
                           (request.modeCount > 0 ? std::min(request.modeCount, dimension) : 0) + shape.basisColumns,
                           m_options.residentLockedColumns, m_options.lockedSpool, m_options.innerProductPieceRows);

    SturmCheck sturm;
    // A Sturm point at `shift`, moved off an eigenvalue (upward or downward) when the factor is singular.
    const auto factoriseAt = [&](double shift, ShiftPurpose purpose, bool isUpward)
    {
        double trial = shift;
        for (std::size_t attempt = 0; attempt <= MAX_SINGULAR_MOVES; ++attempt)
        {
            control.throwIfCancelled();
            const ShiftFactorisation factorisation = pencil.factorise(trial);
            ++solution.factorisationCount;
            if (!factorisation.isSingular())
            {
                sturm.addPoint({trial, factorisation.sturmCount()});
                ShiftRecord record;
                record.shift = trial;
                record.purpose = purpose;
                record.sturmCount = factorisation.sturmCount();
                solution.shifts.push_back(record);
                return trial;
            }
            trial = strategy.moveOffEigenvalue(shift, isUpward, attempt + 1);
        }
        throw base::SolverError(base::concatenate(
            {"eigenvalue analysis: K - sigma*M is singular at ", std::to_string(MAX_SINGULAR_MOVES + 1),
             " shifts around ", std::to_string(shift),
             ": a degree of freedom has neither mass nor stiffness (a massless mechanism); constrain it (single-point "
             "constraints) or give it a mass"}));
    };

    double lowerPoint = -INFINITE;
    double upperPoint = INFINITE;
    // The upper bound first, so that the factor left in the pencil is the one of the first Lanczos shift.
    if (request.hasUpperBound())
        upperPoint = factoriseAt(request.upperBound, ShiftPurpose::UpperBound, true);
    double current = 0.0;
    if (request.startsFromLowest())
    {
        current = factoriseAt(strategy.firstShift().shift, ShiftPurpose::LowestBound, false);
        for (std::size_t descents = 0; sturm.countBelow(current) > 0; ++descents)
        {
            if (descents == ShiftStrategy::MAX_DESCENTS)
                throw base::SolverError(base::concatenate(
                    {"eigenvalue analysis: ", std::to_string(sturm.countBelow(current)),
                     " eigenvalues lie below the shift ", std::to_string(current),
                     "; the stiffness has unbounded negative roots: check the model for negative stiffness or a "
                     "mass matrix that is not positive semi-definite"}));
            current = factoriseAt(strategy.descend(current), ShiftPurpose::LowestBound, false);
        }
    }
    else
    {
        current = factoriseAt(request.lowerBound, ShiftPurpose::LowerBound, false);
        lowerPoint = current;
    }
    const std::size_t lowestCount = sturm.countBelow(current);
    std::size_t wanted = request.modeCount;
    if (request.hasUpperBound())
    {
        const std::size_t inRange = sturm.countBelow(upperPoint) - lowestCount;
        wanted = wanted > 0 ? std::min(wanted, inRange) : inRange;
    }

    std::vector<double> lanczosShifts; // bounded by the shifts
    std::vector<double> sorted;        // bounded by the modes found
    SturmVerification verification = sturm.verify(sorted);
    bool isComplete = wanted == 0;
    for (std::size_t runIndex = 0; !isComplete; ++runIndex)
    {
        const auto inRequest = static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), upperPoint) -
                                                        std::lower_bound(sorted.begin(), sorted.end(), lowerPoint));
        const std::size_t remaining = wanted > inRequest ? wanted - inRequest : 0;
        LanczosRunSettings settings;
        settings.blockSize = shape.blockSize;
        settings.basisColumns = shape.basisColumns;
        settings.target = std::min(shape.modesPerShift, std::max(remaining, shape.blockSize));
        settings.convergenceTolerance = m_options.convergenceTolerance;
        settings.maximumRestarts = m_options.maximumRestarts;
        settings.seed = m_options.seed + SEED_STEP * (runIndex + 1);
        settings.innerProductPieceRows = m_options.innerProductPieceRows;
        LanczosRun run(pencil, kernels, *workspace, budget, settings, control);
        const LanczosRunResult result = run.run(current, store);
        for (auto record = solution.shifts.rbegin(); record != solution.shifts.rend(); ++record)
        {
            if (record->shift == current)
            {
                record->hasLanczosRun = true;
                record->blockSteps = result.blockSteps;
                record->restarts = result.restarts;
                record->convergedCount = result.convergedCount;
                break;
            }
        }
        lanczosShifts.push_back(current);
        control.advance(result.convergedCount);

        sorted = store.eigenvalues();
        std::sort(sorted.begin(), sorted.end());
        verification = sturm.verify(sorted);
        requireNoExtraEigenvalue(verification);
        if (strategy.isComplete(verification) || (result.isSpaceExhausted && !verification.hasMismatch))
        {
            isComplete = true;
            break;
        }
        if (solution.shifts.size() >= m_options.maximumShifts)
            break;
        const ShiftChoice next = strategy.nextShift(verification, sorted, result.ritzEstimates, lanczosShifts);
        current = factoriseAt(next.shift, next.purpose, true);
        verification = sturm.verify(sorted);
        requireNoExtraEigenvalue(verification);
        isComplete = strategy.isComplete(verification);
    }

    // The result, in increasing order (ties by the order found): the workspace is given back first.
    workspace.reset();
    const std::vector<double>& values = store.eigenvalues();
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
    const ShiftStrategy::Selection selection = strategy.select(sorted, lowerPoint, upperPoint);
    solution.eigenvectors = DenseMatrix(budget, dimension, selection.count, "Lanczos: eigenvectors");
    for (std::size_t k = 0; k < selection.count; ++k)
    {
        const std::size_t from = order[selection.first + k];
        solution.eigenvalues.push_back(values[from]);
        solution.residualBounds.push_back(store.residualBounds()[from]);
        store.copyColumn(from, &solution.eigenvectors(0, k));
    }
    solution.isComplete = isComplete;
    solution.verifiedCount = static_cast<std::size_t>(
        std::count_if(solution.eigenvalues.begin(), solution.eigenvalues.end(),
                      [&](double value) { return isComplete || value < verification.verifiedUpTo; }));
    return solution;
}

} // namespace oos::linalg
