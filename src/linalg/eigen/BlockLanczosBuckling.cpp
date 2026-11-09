#include "EigenvectorStore.hpp"
#include "LanczosRun.hpp"
#include "SolveSupport.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/SturmCheck.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::linalg
{
namespace
{

constexpr double INFINITE = std::numeric_limits<double>::infinity();

// One sign of the load factors. In the side's coordinate x = sign·λ >= 0 the
// negative pivots of K − σB at σ = sign·x are the number of roots in (0, x): a Sturm count from 0, which SturmCheck and
// ShiftStrategy use unchanged. The side asks for the roots with x in [lower, upper).
struct BucklingSide
{
    double sign = 1.0;
    double lower = 0.0;
    double upper = INFINITE;
    bool isActive = false;
    SturmCheck sturm;
    std::optional<ShiftStrategy> strategy;
    SturmVerification verification;
    std::vector<double> found; // the x of the roots found on this side, increasing

    [[nodiscard]] bool hasReachedUpper() const noexcept { return verification.verifiedUpTo >= upper; }
    [[nodiscard]] bool isInRange(double x) const noexcept { return x >= lower && x < upper; }
};

// The x of the values on the side of `sign` (x = sign·value > 0), increasing.
std::vector<double> onSide(const std::vector<double>& values, double sign)
{
    std::vector<double> result; // bounded by the modes found
    for (const double value : values)
    {
        if (sign * value > 0.0)
            result.push_back(sign * value);
    }
    std::sort(result.begin(), result.end());
    return result;
}

// How far above the modeCount-th verified root of the other sides the closing Sturm point goes (relative): well
// above the accuracy of a converged root (about 1e-12 relative), so that root is inside the radius, and small enough
// that the radius takes in almost nothing more of the lagging side (each root of it inside must be found).
constexpr double CLOSING_RADIUS_MARGIN = 1.0e-6;

// The Sturm point that closes the lagging side of a request by count (it replaces a slower walk).
// When the other active sides already hold modeCount verified roots in their ranges, every root the request can still
// take from the lagging side lies below the magnitude r of the modeCount-th of them: one Sturm point just above r on
// the lagging side either counts no root it has not found (the request is met: the ND nearest zero are known) or
// counts the missing ones in (0, r), which the next shifts then fill. Without it the lagging side's strategy advances
// by the spectrum's scale (a shift at −695 on the barrel, 445 roots below it) and walks back towards zero. Empty when
// the other sides do not hold enough verified roots (the lagging side is then searched by its own strategy, as
// before), or without a mode count (a range is verified to its bounds).
std::optional<double> closingRadius(const std::array<BucklingSide, 2>& sides, const BucklingSide& lagging,
                                    std::size_t modeCount)
{
    if (modeCount == 0)
        return std::nullopt;
    std::vector<double> verified; // the x of the other sides' verified roots in their ranges; bounded by the modes
    for (const BucklingSide& side : sides)
    {
        if (&side == &lagging || !side.isActive)
            continue;
        for (const double x : side.found)
        {
            if (side.isInRange(x) && (side.hasReachedUpper() || x < side.verification.verifiedUpTo))
                verified.push_back(x);
        }
    }
    if (verified.size() < modeCount)
        return std::nullopt;
    std::sort(verified.begin(), verified.end());
    const double radius = verified[modeCount - 1] * (1.0 + CLOSING_RADIUS_MARGIN);
    const double shift = lagging.strategy->clear(radius, lagging.found);
    return std::min(shift, lagging.upper);
}

} // namespace

EigenSolution BlockLanczos::solveBuckling(ShiftedPencil& pencil, const EigenRequest& request,
                                          const DenseKernels& kernels, base::MemoryBudget& budget) const
{
    const bool hasLowerBound = request.lowerBound > -INFINITE;
    if (request.modeCount == 0 && !(hasLowerBound && request.hasUpperBound()))
        throw std::invalid_argument(
            "buckling request: neither a mode count nor both bounds of the range; give a mode count, or "
            "both bounds (load factors)");
    if (!(request.upperBound > request.lowerBound))
        throw std::invalid_argument(
            base::concatenate({"buckling request: the upper bound ", std::to_string(request.upperBound),
                               " must be above the lower bound ", std::to_string(request.lowerBound),
                               "; check that the lower bound is below the upper bound"}));

    const auto dimension = static_cast<std::size_t>(pencil.dimension());
    const base::OperationControl& control = m_options.control;
    EigenSolution solution;
    if (dimension == 0)
    {
        solution.isComplete = true;
        return solution;
    }
    const double scale = pencil.diagonalRayleighBound();
    std::array<BucklingSide, 2> sides;
    BucklingSide& positive = sides[0];
    BucklingSide& negative = sides[1];
    positive.sign = 1.0;
    positive.isActive = request.upperBound > 0.0;
    positive.lower = std::max(request.lowerBound, 0.0);
    positive.upper = request.upperBound;
    negative.sign = -1.0;
    negative.isActive = request.lowerBound < 0.0;
    negative.lower = std::max(-request.upperBound, 0.0);
    negative.upper = -request.lowerBound;
    for (BucklingSide& side : sides)
    {
        if (side.isActive)
            side.strategy.emplace(EigenRequest{side.lower, side.upper, request.modeCount}, scale);
    }

    const BasisShape shape = shapeFor(dimension, request);
    std::optional<LanczosWorkspace> workspace;
    workspace.emplace(budget, dimension, shape.basisColumns, shape.blockSize);
    EigenvectorStore store(budget, dimension,
                           (request.modeCount > 0 ? std::min(request.modeCount, dimension) : 0) + shape.basisColumns,
                           m_options.residentLockedColumns, m_options.lockedSpool, m_options.innerProductPieceRows);

    // A Sturm point at λ = shift (on the side of its sign; at 0 on the sides that start there), moved away from 0
    // when the factor is singular.
    const auto factoriseAt = [&](double shift, ShiftPurpose purpose)
    {
        double trial = shift;
        for (std::size_t attempt = 0; attempt <= MAX_SINGULAR_MOVES; ++attempt)
        {
            control.throwIfCancelled();
            const ShiftFactorisation factorisation = pencil.factorise(trial);
            ++solution.factorisationCount;
            if (trial == 0.0 && (factorisation.isSingular() || factorisation.sturmCount() > 0))
                throw base::SolverError(base::concatenate(
                    {"buckling analysis: the stiffness matrix is not positive definite (",
                     std::to_string(factorisation.inertia.negative), " negative and ",
                     std::to_string(factorisation.inertia.zero),
                     " zero pivots); the load factors of K + lambda*K_d need a stable structure: constrain the "
                     "mechanisms (single-point constraints) and check the model for negative stiffness"}));
            if (!factorisation.isSingular())
            {
                for (BucklingSide& side : sides)
                {
                    if (!side.isActive)
                        continue;
                    if (trial == 0.0 && side.lower == 0.0)
                        side.sturm.addPoint({0.0, 0});
                    else if (side.sign * trial > 0.0)
                        side.sturm.addPoint({side.sign * trial, factorisation.sturmCount()});
                }
                ShiftRecord record;
                record.shift = trial;
                record.purpose = purpose;
                record.sturmCount = factorisation.sturmCount();
                solution.shifts.push_back(record);
                return trial;
            }
            const double sign = shift > 0.0 ? 1.0 : -1.0;
            const ShiftStrategy& strategy = sides[sign > 0.0 ? 0 : 1].strategy.value();
            trial = sign * strategy.moveOffEigenvalue(std::abs(shift), true, attempt + 1);
        }
        throw base::SolverError(base::concatenate(
            {"buckling analysis: K + sigma*K_d is singular at ", std::to_string(MAX_SINGULAR_MOVES + 1),
             " shifts around ", std::to_string(shift),
             ": the load factors are not separated there (a nearly singular pencil); move the bounds of the range"}));
    };

    // The first Lanczos shift: 0 when a side starts there (θ = 1/λ: the roots nearest 0 of both signs converge first),
    // else the lower bound of the one active side. It is factorised last, so that its factor stays in the pencil; the
    // factorisation at 0 (the check that K is positive definite, on which every count relies) comes first otherwise.
    const bool startsAtZero =
        (positive.isActive && positive.lower == 0.0) || (negative.isActive && negative.lower == 0.0);
    const BucklingSide& single = positive.isActive ? positive : negative;
    const double firstShift = startsAtZero ? 0.0 : single.sign * single.lower;
    if (!startsAtZero)
        factoriseAt(0.0, ShiftPurpose::LowestBound);
    for (const BucklingSide& side : sides)
    {
        if (side.isActive && side.upper < INFINITE)
            factoriseAt(side.sign * side.upper, ShiftPurpose::UpperBound);
    }
    double current = factoriseAt(firstShift, startsAtZero ? ShiftPurpose::LowestBound : ShiftPurpose::LowerBound);

    // The roots asked: the mode count, or the Sturm count of the range when it is smaller or there is no count.
    std::size_t wanted = request.modeCount;
    std::size_t inRange = 0;
    bool isRangeCounted = true;
    for (const BucklingSide& side : sides)
    {
        if (!side.isActive)
            continue;
        if (side.upper == INFINITE)
        {
            isRangeCounted = false;
            continue;
        }
        inRange += side.sturm.countBelow(side.upper) - side.sturm.points().front().countBelow;
    }
    if (isRangeCounted)
        wanted = wanted > 0 ? std::min(wanted, inRange) : inRange;

    // Updates the verification of every side from the eigenvalues found; true when the request is met.
    std::vector<double> sorted; // every λ found, increasing; bounded by the modes found
    const auto verifySides = [&]()
    {
        for (BucklingSide& side : sides)
        {
            if (!side.isActive)
                continue;
            side.found = onSide(sorted, side.sign);
            side.verification = side.sturm.verify(side.found);
            requireNoExtraEigenvalue(side.verification);
        }
        bool isEverySideAtUpper = true;
        double radius = INFINITE;
        for (const BucklingSide& side : sides)
        {
            if (!side.isActive || side.hasReachedUpper())
                continue;
            isEverySideAtUpper = false;
            radius = std::min(radius, side.verification.verifiedUpTo);
        }
        if (isEverySideAtUpper)
            return true;
        if (request.modeCount == 0)
            return false;
        // Every root of magnitude below the radius is known on both sides: when there are enough of them, the
        // smallest modeCount are found.
        std::size_t verified = 0;
        for (const BucklingSide& side : sides)
        {
            if (!side.isActive)
                continue;
            const double end = std::min(radius, side.upper);
            verified += static_cast<std::size_t>(std::count_if(side.found.begin(), side.found.end(),
                                                               [&](double x) { return x >= side.lower && x < end; }));
        }
        return verified >= request.modeCount;
    };

    // The side that lags: the active side not at its upper bound with the smallest verified radius.
    const auto laggingSide = [&]() -> BucklingSide*
    {
        BucklingSide* lagging = nullptr;
        for (BucklingSide& side : sides)
        {
            if (side.isActive && !side.hasReachedUpper() &&
                (lagging == nullptr || side.verification.verifiedUpTo < lagging->verification.verifiedUpTo))
                lagging = &side;
        }
        return lagging;
    };

    std::vector<double> lanczosShifts; // bounded by the shifts
    std::vector<double> ritzEstimates; // of the last run; bounded by its basis
    std::size_t runCount = 0;
    bool isComplete = wanted == 0 || verifySides();
    while (!isComplete)
    {
        BucklingSide* lagging = laggingSide();
        // A run converges the roots nearest its shift, on the shift's side (both at 0). When the other sides already
        // hold the roots the count asks and only the lagging side is left to close (closingRadius), a run at a shift
        // of another side finds roots of no use to the request (the positive side's Sturm point that verified
        // the ND roots nearest 0, the negative side still to close): the lagging side's next shift is chosen without
        // it. Otherwise the run is made, as before (it may still find roots the request takes).
        const bool runsHere = runCount == 0 || current == 0.0 || lagging->sign * current > 0.0 ||
                              !closingRadius(sides, *lagging, request.modeCount).has_value();
        if (runsHere)
        {
            std::size_t inRequest = 0;
            for (const BucklingSide& side : sides)
            {
                if (side.isActive)
                    inRequest += static_cast<std::size_t>(std::count_if(side.found.begin(), side.found.end(),
                                                                        [&](double x) { return side.isInRange(x); }));
            }
            const std::size_t remaining = wanted > inRequest ? wanted - inRequest : 0;
            LanczosRunSettings settings;
            settings.blockSize = shape.blockSize;
            settings.basisColumns = shape.basisColumns;
            settings.target = std::min(shape.modesPerShift, std::max(remaining, shape.blockSize));
            settings.convergenceTolerance = m_options.convergenceTolerance;
            settings.maximumRestarts = m_options.maximumRestarts;
            settings.seed = m_options.seed + SEED_STEP * (runCount + 1);
            settings.innerProductPieceRows = m_options.innerProductPieceRows;
            LanczosRun run(pencil, kernels, *workspace, budget, settings, control);
            const LanczosRunResult result = run.run(current, store);
            ++runCount;
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
            ritzEstimates = result.ritzEstimates;
            control.advance(result.convergedCount);

            sorted = store.eigenvalues();
            std::sort(sorted.begin(), sorted.end());
            isComplete = verifySides();
            const bool hasMismatch = std::any_of(sides.begin(), sides.end(), [](const BucklingSide& side)
                                                 { return side.isActive && side.verification.hasMismatch; });
            if (isComplete || (result.isSpaceExhausted && !hasMismatch))
            {
                isComplete = true;
                break;
            }
            lagging = laggingSide();
        }
        if (solution.shifts.size() >= m_options.maximumShifts)
            break;
        std::vector<double> ritz = onSide(ritzEstimates, lagging->sign);
        std::vector<double> shifts; // the Lanczos shifts in the side's coordinate (0 included)
        for (const double shift : lanczosShifts)
            shifts.push_back(lagging->sign * shift);
        ShiftChoice next = lagging->strategy->nextShift(lagging->verification, lagging->found, ritz, shifts);
        if (const std::optional<double> radius = closingRadius(sides, *lagging, request.modeCount);
            radius && *radius > lagging->verification.verifiedUpTo && *radius < next.shift)
            next = {*radius, ShiftPurpose::Advance};
        current = factoriseAt(lagging->sign * next.shift, next.purpose);
        isComplete = verifySides();
    }

    // The result: the roots of the range, by increasing magnitude (the positive first on a tie), the modeCount
    // smallest when a count is set. The workspace is given back first.
    workspace.reset();
    const std::vector<double>& values = store.eigenvalues();
    std::vector<std::size_t> order;
    for (std::size_t k = 0; k < values.size(); ++k)
    {
        const BucklingSide& side = values[k] > 0.0 ? positive : negative;
        if (side.isActive && side.isInRange(std::abs(values[k])))
            order.push_back(k);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b)
                     {
                         const double magnitudeA = std::abs(values[a]);
                         const double magnitudeB = std::abs(values[b]);
                         return magnitudeA < magnitudeB || (magnitudeA == magnitudeB && values[a] > values[b]);
                     });
    if (request.modeCount > 0 && order.size() > request.modeCount)
        order.resize(request.modeCount);
    solution.eigenvectors = DenseMatrix(budget, dimension, order.size(), "Lanczos: eigenvectors");
    for (std::size_t k = 0; k < order.size(); ++k)
    {
        solution.eigenvalues.push_back(values[order[k]]);
        solution.residualBounds.push_back(store.residualBounds()[order[k]]);
        store.copyColumn(order[k], &solution.eigenvectors(0, k));
    }
    solution.isComplete = isComplete;
    solution.verifiedCount = static_cast<std::size_t>(
        std::count_if(solution.eigenvalues.begin(), solution.eigenvalues.end(),
                      [&](double value)
                      {
                          const BucklingSide& side = value > 0.0 ? positive : negative;
                          return isComplete || std::abs(value) < side.verification.verifiedUpTo;
                      }));
    return solution;
}

} // namespace oos::linalg
