#include "oos/assembly/singularities/SingularBlock.hpp"

#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/assembly/singularities/SmallSymmetricEigen.hpp"
#include "oos/base/math/MatN.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace oos::assembly
{

namespace
{

// Two subsets whose |det V_C| differ by less than this are a tie (the lower components are chosen): the determinants
// are cosines computed to round-off.
constexpr double CHOICE_TIE_TOLERANCE = 1.0e-12;

// The determinant of the square sub-matrix of V (|F| × s, column-major in `vectors`, columns `singular`) on the rows
// `chosen`.
double subsetDeterminant(const base::MatN<3>& vectors, const std::array<std::size_t, 3>& singular,
                         const std::array<std::size_t, 3>& chosen, std::size_t size)
{
    const auto at = [&](std::size_t i, std::size_t j) { return vectors(chosen[i], singular[j]); };
    if (size == 1)
        return at(0, 0);
    if (size == 2)
        return at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);
    return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) -
           at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
           at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
}

} // namespace

SingularBlock SingularBlock::analyse(std::span<const double, 9> block, const std::array<bool, 3>& isFree,
                                     double pivotZeroThreshold)
{
    SingularBlock analysis;
    std::array<std::size_t, 3> freeComponents{};
    std::size_t freeCount = 0;
    for (std::size_t c = 0; c < 3; ++c)
        if (isFree[c])
            freeComponents[freeCount++] = c;
    if (freeCount == 0)
        return analysis;
    // The eigenpairs of the free sub-block (its own size: 1, 2 or 3), the vectors in the first rows and columns of a
    // 3 × 3.
    std::array<double, 3> values{};
    base::MatN<3> vectors;
    const auto termOf = [&](std::size_t i, std::size_t j)
    {
        return 0.5 *
               (block[freeComponents[i] * 3 + freeComponents[j]] + block[freeComponents[j] * 3 + freeComponents[i]]);
    };
    if (freeCount == 1)
    {
        values[0] = termOf(0, 0);
        vectors(0, 0) = 1.0;
    }
    else if (freeCount == 2)
    {
        base::MatN<2> sub;
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                sub(i, j) = termOf(i, j);
        const SmallEigenDecomposition<2> eigen = SmallSymmetricEigen<2>::decompose(sub);
        for (std::size_t j = 0; j < 2; ++j)
        {
            values[j] = eigen.values[j];
            for (std::size_t i = 0; i < 2; ++i)
                vectors(i, j) = eigen.vectors(i, j);
        }
    }
    else
    {
        base::MatN<3> sub;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                sub(i, j) = termOf(i, j);
        const SmallEigenDecomposition<3> eigen = SmallSymmetricEigen<3>::decompose(sub);
        values = eigen.values;
        vectors = eigen.vectors;
    }
    double largest = 0.0;
    for (std::size_t k = 0; k < freeCount; ++k)
        largest = std::max(largest, values[k]);
    if (!(largest > 0.0))
    {
        // No stiffness at all in the free components: every one is singular, along its own axis.
        for (std::size_t i = 0; i < freeCount; ++i)
            analysis.ratios[freeComponents[i]] = 0.0;
        return analysis;
    }
    std::array<std::size_t, 3> singular{};
    std::array<double, 3> singularRatio{};
    std::size_t singularCount = 0;
    for (std::size_t k = 0; k < freeCount; ++k)
    {
        const double ratio = values[k] / largest;
        if (ratio < pivotZeroThreshold)
        {
            singular[singularCount] = k;
            singularRatio[singularCount] = std::max(ratio, 0.0);
            ++singularCount;
        }
    }
    if (singularCount == 0)
        return analysis;
    // The components C maximising |det V_C|, subsets of the free rows in increasing (lexicographic) order.
    std::array<std::size_t, 3> best{};
    double bestDeterminant = -1.0;
    std::array<std::size_t, 3> chosen{};
    for (std::size_t mask = 1; mask < (1U << freeCount); ++mask)
    {
        std::size_t count = 0;
        for (std::size_t i = 0; i < freeCount; ++i)
            if ((mask >> i) & 1U)
                chosen[count++] = i;
        if (count != singularCount)
            continue;
        // The masks run through the subsets of each size in lexicographic order: on a tie the first, lowest, stays.
        const double determinant = std::abs(subsetDeterminant(vectors, singular, chosen, count));
        if (determinant > bestDeterminant + CHOICE_TIE_TOLERANCE)
        {
            bestDeterminant = determinant;
            best = chosen;
        }
    }
    analysis.isAligned = bestDeterminant >= GridPointSingularities::ALIGNMENT_COSINE;
    for (std::size_t c = 0; c < singularCount; ++c)
    {
        // The ratio of a component: that of the singular direction with the largest part along it.
        std::size_t pair = 0;
        for (std::size_t j = 1; j < singularCount; ++j)
            if (std::abs(vectors(best[c], singular[j])) > std::abs(vectors(best[c], singular[pair])))
                pair = j;
        analysis.ratios[freeComponents[best[c]]] = singularRatio[pair];
    }
    return analysis;
}

} // namespace oos::assembly
