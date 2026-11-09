#include "oos/linalg/eigen/ShiftStrategy.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::linalg
{
namespace
{

constexpr double INFINITE = std::numeric_limits<double>::infinity();

// Two shifts closer than this (relative) are the same shift for the purpose of not repeating a Lanczos run: a run
// there would converge the same modes, all locked already.
constexpr double SAME_SHIFT_TOLERANCE = 1.0e-3;

// The clearance passes allowed: each moves the shift above one eigenvalue, so a cluster of that many eigenvalues within
// the clearance of each other is the most it can cross.
constexpr std::size_t MAX_CLEARANCE_PASSES = 64;

} // namespace

ShiftStrategy::ShiftStrategy(const EigenRequest& request, double scale) : m_request(request)
{
    if (request.modeCount == 0 && !request.hasUpperBound())
        throw std::invalid_argument(
            "eigenvalue request: neither a mode count nor an upper bound; give a mode count, or an upper bound to "
            "bound the range");
    if (request.hasUpperBound() && !request.startsFromLowest() && !(request.upperBound > request.lowerBound))
        throw std::invalid_argument(
            base::concatenate({"eigenvalue request: the upper bound ", std::to_string(request.upperBound),
                               " must be above the lower bound ", std::to_string(request.lowerBound),
                               "; check that the lower bound is below the upper bound"}));
    if (request.hasUpperBound() && request.startsFromLowest() && !(request.upperBound > 0.0))
        throw std::invalid_argument(
            base::concatenate({"eigenvalue request: the upper bound ", std::to_string(request.upperBound),
                               " must be above 0; check the upper bound"}));
    m_scale = std::isfinite(scale) && scale > 0.0 ? scale : 1.0;
}

ShiftChoice ShiftStrategy::firstShift() const noexcept
{
    if (m_request.startsFromLowest())
        return {-RIGID_SHIFT_FRACTION * m_scale, ShiftPurpose::LowestBound};
    return {m_request.lowerBound, ShiftPurpose::LowerBound};
}

double ShiftStrategy::descend(double shift) const noexcept
{
    return shift < 0.0 ? shift * DESCENT_FACTOR : -RIGID_SHIFT_FRACTION * m_scale;
}

double ShiftStrategy::moveOffEigenvalue(double shift, bool isUpward, std::size_t attempt) const noexcept
{
    const double step =
        SINGULAR_SHIFT_MOVE * static_cast<double>(attempt) * std::max(std::abs(shift), RIGID_SHIFT_FRACTION * m_scale);
    return isUpward ? shift + step : shift - step;
}

bool ShiftStrategy::isComplete(const SturmVerification& verification) const noexcept
{
    if (m_request.modeCount > 0 && verification.verifiedCount >= m_request.modeCount)
        return true;
    return m_request.hasUpperBound() && verification.verifiedUpTo >= m_request.upperBound;
}

ShiftStrategy::Selection ShiftStrategy::select(std::span<const double> sortedEigenvalues, double lowerPoint,
                                               double upperPoint) const noexcept
{
    const auto first = std::lower_bound(sortedEigenvalues.begin(), sortedEigenvalues.end(), lowerPoint);
    const auto end = std::lower_bound(first, sortedEigenvalues.end(), upperPoint);
    Selection selection;
    selection.first = static_cast<std::size_t>(first - sortedEigenvalues.begin());
    selection.count = static_cast<std::size_t>(end - first);
    if (m_request.modeCount > 0)
        selection.count = std::min(selection.count, m_request.modeCount);
    return selection;
}

double ShiftStrategy::clearanceAt(double value, double shift) const noexcept
{
    return SHIFT_CLEARANCE * std::max({std::abs(value), std::abs(shift), RIGID_SHIFT_FRACTION * m_scale});
}

double ShiftStrategy::clear(double shift, std::span<const double> sortedEigenvalues) const noexcept
{
    for (std::size_t pass = 0; pass < MAX_CLEARANCE_PASSES; ++pass)
    {
        bool isClear = true;
        for (const double value : sortedEigenvalues)
        {
            if (std::abs(shift - value) < clearanceAt(value, shift))
            {
                shift = value + clearanceAt(value, shift);
                isClear = false;
            }
        }
        if (isClear)
            break;
    }
    return shift;
}

bool ShiftStrategy::isNearShift(double shift, std::span<const double> lanczosShifts) const noexcept
{
    return std::any_of(lanczosShifts.begin(), lanczosShifts.end(),
                       [&](double used)
                       {
                           return std::abs(shift - used) <=
                                  SAME_SHIFT_TOLERANCE *
                                      std::max({std::abs(used), std::abs(shift), RIGID_SHIFT_FRACTION * m_scale});
                       });
}

ShiftChoice ShiftStrategy::nextShift(const SturmVerification& verification, std::span<const double> sortedEigenvalues,
                                     std::span<const double> ritzEstimates, std::span<const double> lanczosShifts) const
{
    const double a = verification.verifiedUpTo;
    const double b = verification.hasMismatch ? verification.mismatch.upper : INFINITE;
    const ShiftPurpose purpose = verification.hasMismatch ? ShiftPurpose::FillGap : ShiftPurpose::Advance;
    const auto firstFound = std::lower_bound(sortedEigenvalues.begin(), sortedEigenvalues.end(), a);
    const auto endFound = std::lower_bound(firstFound, sortedEigenvalues.end(), b);
    const auto ritzAbove = [&](double value)
    {
        const auto at = std::upper_bound(ritzEstimates.begin(), ritzEstimates.end(), value);
        return at != ritzEstimates.end() && *at < b ? *at : INFINITE;
    };

    double shift = 0.0;
    // Modes are missing between two Sturm points: an unconverged Ritz estimate inside the interval points at one (a
    // root seen by the last run but not converged, often the missing copy of a repeated root).
    if (verification.hasMismatch)
    {
        for (const double estimate : ritzEstimates)
        {
            if (estimate > a && estimate < b && !isNearShift(estimate, lanczosShifts))
                return {clear(estimate, sortedEigenvalues), purpose};
        }
    }
    if (firstFound != endFound)
    {
        const double highest = *(endFound - 1);
        const double estimate = ritzAbove(highest + clearanceAt(highest, highest));
        if (estimate < INFINITE)
            shift = 0.5 * (highest + estimate);
        else
        {
            // At least the clearance zone of the start: an interval starting at σ₀ below rigid-body roots has a
            // highest root about 0, whose extent from σ₀ is the rigid-body zone itself.
            const double extent = std::max(highest - a, RIGID_SHIFT_FRACTION * m_scale);
            shift = highest + STEP_FRACTION * extent;
            if (b < INFINITE)
                shift = std::min(shift, 0.5 * (highest + b));
        }
    }
    else
    {
        const double estimate = ritzAbove(a);
        if (estimate < INFINITE)
            shift = estimate;
        else
            shift = b < INFINITE ? 0.5 * (a + b) : a + std::max(std::abs(a), m_scale);
    }

    if (isNearShift(shift, lanczosShifts))
    {
        // The widest gap between the ends of the interval and the eigenvalues found in it (a finite end above the
        // highest found when the interval is open).
        std::vector<double> ends; // bounded by the eigenvalues of one interval
        ends.push_back(a);
        ends.insert(ends.end(), firstFound, endFound);
        const double top = ends.back();
        ends.push_back(b < INFINITE ? b : top + std::max(top - a, m_scale));
        std::size_t widest = 0;
        for (std::size_t k = 1; k + 1 < ends.size(); ++k)
        {
            if (ends[k + 1] - ends[k] > ends[widest + 1] - ends[widest])
                widest = k;
        }
        shift = 0.5 * (ends[widest] + ends[widest + 1]);
        if (isNearShift(shift, lanczosShifts))
            shift = ends[widest] + 0.25 * (ends[widest + 1] - ends[widest]);
    }
    return {clear(shift, sortedEigenvalues), purpose};
}

} // namespace oos::linalg
