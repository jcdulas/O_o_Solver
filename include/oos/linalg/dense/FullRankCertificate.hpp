#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>

namespace oos::linalg
{

// A proof that the smallest singular value of a matrix A (m × n, m ≥ n) is at least a bound, by its unpivoted
// Householder QR (the rank test of the load basis, assembly::LoadBasis, which needs to know that the
// column-pivoted QR of thousands of independent load cases cannot stop early, without running it):
//
//     σ_min(A) = σ_min(R) = 1/‖R⁻¹‖₂ ≥ 1/‖R⁻¹‖_F,
//
// R the triangular factor of A = Q·R (Golub and Van Loan, Matrix Computations, §2.4 and §5.2). The QR is blocked
// (Schreiber and Van Loan, SIAM J. Sci. Stat. Comput. 10, 1989: the compact WY form Q = I − V·T·Vᵀ of each panel of
// PANEL_COLUMNS reflectors, the trailing columns updated by two matrix products of the dense kernels), so that it runs
// at the rate of the kernels, where a column-pivoted QR must update the norms of every column after every reflector
// and stays bound by memory (LAPACK xGEQP3 does half its work in matrix-vector products). ‖R⁻¹‖_F is formed by blocks
// of columns of R⁻ᵀ = D⁻¹·L⁻¹, L = Rᵀ·D⁻¹ unit lower triangular and D the diagonal of R (forward substitutions of the
// dense kernels, the blocks summed in a fixed order).
//
// The QR stops as soon as a diagonal |r_kk| is below the bound: σ_min(R) ≤ min_k |r_kk|, so no proof is possible then,
// and a matrix of low rank costs only the columns up to its first dependent one.
//
// The rounding: R is the exact factor of A + E with ‖E‖ of the order of n·ε·‖A‖ (Higham, Accuracy and Stability of
// Numerical Algorithms, §19.3), so the bound proves σ_min(A) ≥ 1/‖R⁻¹‖_F − ‖E‖: the caller's bound carries that margin.
//
// Determinism: the products are those of the dense kernels (bits independent of the threads), the sums of the blocks
// of ‖R⁻¹‖_F combined in block order. Thread-safety: stateless. Ownership: none kept.
class FullRankCertificate
{
public:
    // The reflectors of a panel: wide enough for the matrix products to run near the kernels' rate, narrow enough
    // that the panel factorisation (2·m·PANEL_COLUMNS² operations per panel, one thread) stays a small share.
    static constexpr std::size_t PANEL_COLUMNS = 32;
    // The columns of R⁻ᵀ one task forms (a block of the identity solved with L).
    static constexpr std::size_t INVERSE_BLOCK_COLUMNS = 64;

    struct Outcome
    {
        bool isProved = false;           // σ_min(A) ≥ 1/‖R⁻¹‖_F ≥ bound
        std::size_t factoredColumns = 0; // the columns of the QR made before it stopped (n when it did not)
        double singularValueBound = 0.0; // 1/‖R⁻¹‖_F when every column was factored, 0 otherwise
    };

    // `a` (m × n, m ≥ n, column-major) is overwritten (its QR, then L below the diagonal). Throws std::invalid_argument
    // when m < n; MemoryBudgetError for the work blocks (a panel, its T and the products, the blocks of R⁻ᵀ).
    [[nodiscard]] static Outcome prove(DenseMatrixView a, double bound, const DenseKernels& kernels,
                                       base::MemoryBudget& budget);
};

} // namespace oos::linalg
