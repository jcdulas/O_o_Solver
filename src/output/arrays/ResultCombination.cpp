#include "oos/output/arrays/ResultCombination.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::output
{

std::size_t ResultCombination::batchCaseCount(std::size_t rows, std::size_t quantities,
                                              std::uint64_t availableBytes) const
{
    const std::uint64_t perCase = static_cast<std::uint64_t>(std::max<std::size_t>(rows, 1)) *
                                  std::max<std::size_t>(quantities, 1) * sizeof(double);
    std::uint64_t cases = std::max<std::uint64_t>(1, BATCH_BYTES / perCase);
    if (cases < MIN_BATCH_CASES)
    {
        // PERFORMANCE: a large model's batch raised towards MIN_BATCH_CASES when the memory left allows it, so that the
        // basis results are read once per batch rather than once per case.
        // Measured: on a 1 M DOF plate with 64 cases.
        const std::uint64_t affordable = availableBytes / MIN_BATCH_SHARE_OF_AVAILABLE / perCase;
        cases = std::max(cases, std::min<std::uint64_t>(MIN_BATCH_CASES, affordable));
    }
    return static_cast<std::size_t>(std::min<std::uint64_t>(cases, std::max<std::size_t>(caseCount(), 1)));
}

void ResultCombination::combine(linalg::ConstDenseMatrixView basisResults, std::size_t first, std::size_t count,
                                linalg::DenseMatrixView out, base::MemoryBudget& budget) const
{
    if (basisResults.columns != basisSize() || first + count > caseCount() || out.rows != basisResults.rows ||
        out.columns < count)
        throw std::invalid_argument(
            base::concatenate({"ResultCombination::combine: expected basis results of ", std::to_string(basisSize()),
                               " columns, cases within ", std::to_string(caseCount()), " and an output of ",
                               std::to_string(basisResults.rows), " rows; found ", std::to_string(basisResults.columns),
                               " columns, cases ", std::to_string(first), " + ", std::to_string(count), ", output ",
                               std::to_string(out.rows), " × ", std::to_string(out.columns)}));
    for (std::size_t j = 0; j < count; ++j)
        for (std::size_t i = 0; i < out.rows; ++i)
            out(i, j) = 0.0;
    if (basisSize() == 0 || count == 0 || out.rows == 0)
        return;
    // B = −A(:, batch)ᵀ (count × k), so that C ← C − X·Bᵀ is X·A(:, batch).
    linalg::DenseMatrix negatedTranspose(budget, count, basisSize(), "result combination coefficients");
    for (std::size_t j = 0; j < count; ++j)
        for (std::size_t i = 0; i < basisSize(); ++i)
            negatedTranspose(j, i) = -m_coefficients(i, first + j);
    m_kernels->updateRectangular(out.block(0, 0, out.rows, count), basisResults, negatedTranspose.view(), {});
}

} // namespace oos::output
