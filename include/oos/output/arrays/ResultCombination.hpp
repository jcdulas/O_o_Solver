#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::output
{

// The results of the load cases from those of the load basis (combined, not recomputed):
// the quantities linear in the loads (displacements, SPC and MPC forces, applied loads, the reduced loads and the
// residual of the checks) are computed for the k basis vectors, and one matrix product gives them for a batch of cases,
//     Y(:, batch) = X·A(:, batch),    X (rows × k) the results of the basis, A (k × cases) the coefficients,
// through the dense kernels (C ← C − X·D·Bᵀ with B = −A(:, batch)ᵀ): every term of Y is one chain over the basis
// vectors in a fixed order, so the bits do not depend on the threads.
//
// The quantities that are not linear (von Mises and principal stresses, margins, failure indices, strain energy,
// envelopes) and the element results are computed case by case from the combined displacements (StaticSolution): an
// element's recovery costs about as much as combining its k basis results once k exceeds a few tens, and recovering
// from the combined displacements keeps every column of every element type exact without a per-family rule for the
// non-linear columns.
//
// Thread-safety: const methods are safe concurrently. Ownership: owns the coefficients; borrows the kernels.
class ResultCombination
{
public:
    // The memory a batch of combined results may take: the cases of a batch are chosen so that the combined matrices
    // of all the quantities stay within it (and at least one case).
    static constexpr std::uint64_t BATCH_BYTES = 64ULL << 20;
    // A batch holds at least this many cases (or all of them) when the memory left allows it: within BATCH_BYTES a
    // large model would combine one case at a time, reading the basis results again for every case (k·g values per
    // case and quantity: the combination of 64 cases of a 1 M DOF model grew with the square of the cases), and the
    // element recovery of a block of cases would get blocks of one. The bits do not depend on the batch (every term
    // of Y is its own chain), so the memory may decide it.
    static constexpr std::size_t MIN_BATCH_CASES = 32;
    // The share of the memory left that a batch grown to MIN_BATCH_CASES may take.
    static constexpr std::uint64_t MIN_BATCH_SHARE_OF_AVAILABLE = 8;

    ResultCombination(const linalg::DenseKernels& kernels, linalg::DenseMatrix coefficients)
        : m_kernels(&kernels), m_coefficients(std::move(coefficients))
    {
    }

    [[nodiscard]] std::size_t basisSize() const noexcept { return m_coefficients.rows(); }
    [[nodiscard]] std::size_t caseCount() const noexcept { return m_coefficients.columns(); }
    [[nodiscard]] const linalg::DenseMatrix& coefficients() const noexcept { return m_coefficients; }

    // The cases of one batch for `quantities` matrices of `rows` rows: those that fit BATCH_BYTES, raised towards
    // MIN_BATCH_CASES within 1/MIN_BATCH_SHARE_OF_AVAILABLE of `availableBytes` (the memory left; 0 keeps BATCH_BYTES),
    // at least one and at most every case.
    [[nodiscard]] std::size_t batchCaseCount(std::size_t rows, std::size_t quantities,
                                             std::uint64_t availableBytes = 0) const;

    // out (rows × count) = basisResults (rows × k) · A(:, first : first + count). Throws std::invalid_argument for
    // sizes that disagree; MemoryBudgetError.
    void combine(linalg::ConstDenseMatrixView basisResults, std::size_t first, std::size_t count,
                 linalg::DenseMatrixView out, base::MemoryBudget& budget) const;

private:
    const linalg::DenseKernels* m_kernels;
    linalg::DenseMatrix m_coefficients;
};

} // namespace oos::output
