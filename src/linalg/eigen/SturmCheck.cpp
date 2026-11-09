#include "oos/linalg/eigen/SturmCheck.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

std::string describe(const SturmPoint& point)
{
    return base::concatenate(
        {"shift ", std::to_string(point.shift), " (", std::to_string(point.countBelow), " eigenvalues below)"});
}

} // namespace

void SturmCheck::addPoint(SturmPoint point)
{
    const auto at = std::lower_bound(m_points.begin(), m_points.end(), point.shift,
                                     [](const SturmPoint& p, double shift) { return p.shift < shift; });
    const bool isSameShift = at != m_points.end() && at->shift == point.shift;
    const bool isBelowNext =
        at == m_points.end() || (isSameShift ? at->countBelow == point.countBelow : at->countBelow >= point.countBelow);
    const bool isAbovePrevious = at == m_points.begin() || std::prev(at)->countBelow <= point.countBelow;
    if (!isBelowNext || !isAbovePrevious)
    {
        const SturmPoint& other = (!isBelowNext) ? *at : *std::prev(at);
        throw base::SolverError(base::concatenate(
            {"eigenvalue analysis: the Sturm counts contradict each other: ", describe(point), " against ",
             describe(other),
             "; the factorisations of K - sigma*M are not reliable (a nearly singular pencil, or a mass matrix that "
             "is not positive semi-definite): check the model for mechanisms without mass and negative masses"}));
    }
    if (!isSameShift)
        m_points.insert(at, point);
}

std::size_t SturmCheck::countBelow(double shift) const
{
    for (const SturmPoint& point : m_points)
    {
        if (point.shift == shift)
            return point.countBelow;
    }
    throw std::invalid_argument(base::concatenate(
        {"SturmCheck::countBelow: no Sturm point at the shift ", std::to_string(shift), "; add it first"}));
}

SturmVerification SturmCheck::verify(std::span<const double> sortedEigenvalues) const
{
    if (m_points.empty())
        throw std::invalid_argument("SturmCheck::verify: no Sturm point; factorise at a shift first");
    SturmVerification result;
    result.verifiedUpTo = m_points.front().shift;
    // The eigenvalues found in [points[k], points[k + 1]), by a merge of the two sorted sequences.
    auto next = std::lower_bound(sortedEigenvalues.begin(), sortedEigenvalues.end(), m_points.front().shift);
    for (std::size_t k = 0; k + 1 < m_points.size(); ++k)
    {
        const SturmPoint& lower = m_points[k];
        const SturmPoint& upper = m_points[k + 1];
        const auto end = std::lower_bound(next, sortedEigenvalues.end(), upper.shift);
        const auto found = static_cast<std::size_t>(end - next);
        const std::size_t expected = upper.countBelow - lower.countBelow;
        if (found != expected)
        {
            result.hasMismatch = true;
            result.mismatch = {lower.shift, upper.shift, expected, found};
            return result;
        }
        result.verifiedUpTo = upper.shift;
        result.verifiedCount += found;
        next = end;
    }
    return result;
}

} // namespace oos::linalg
