#include "oos/linalg/dense/IterativeRefinement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::linalg
{

IterativeRefinement::Step IterativeRefinement::record(double correction)
{
    const bool isFirst = m_corrections.empty();
    const double previous = isFirst ? 0.0 : m_corrections.back();
    m_corrections.push_back(correction);
    Step step;
    step.isAccepted = isFirst || correction < previous;
    const bool isHalved = isFirst || correction <= DECREASE * previous;
    step.continues = step.isAccepted && isHalved && correction > TOLERANCE && m_corrections.size() < MAX_STEPS;
    return step;
}

double IterativeRefinement::largestRelative(ConstDenseMatrixView corrections, ConstDenseMatrixView displacements)
{
    if (corrections.rows != displacements.rows || corrections.columns != displacements.columns)
        throw std::invalid_argument(base::concatenate(
            {"IterativeRefinement::largestRelative: expected corrections of the displacements' shape ",
             std::to_string(displacements.rows), " x ", std::to_string(displacements.columns), ", found ",
             std::to_string(corrections.rows), " x ", std::to_string(corrections.columns)}));
    double largest = 0.0;
    for (std::size_t b = 0; b < corrections.columns; ++b)
    {
        double correction = 0.0;
        double displacement = 0.0;
        for (std::size_t e = 0; e < corrections.rows; ++e)
        {
            correction = std::max(correction, std::abs(corrections(e, b)));
            displacement = std::max(displacement, std::abs(displacements(e, b)));
        }
        if (displacement > 0.0)
            largest = std::max(largest, correction / displacement);
    }
    return largest;
}

} // namespace oos::linalg
