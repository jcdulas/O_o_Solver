#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>

namespace oos::reduction
{

// What the semi-definiteness test found.
struct SemiDefiniteResult
{
    bool isSemiDefinite = true;
    // The pivots above the tolerance: the rank of the matrix to the tolerance (a K* with its six rigid-body motions has
    // rank n − 6).
    std::size_t rank = 0;
    // The smallest pivot, relative to the largest term of the matrix: the last pivot taken, or, when the factorisation
    // stops, the most negative diagonal left (an indefinite remainder of zero diagonals counts as −its largest term).
    // Negative below −tolerance: a negative eigenvalue.
    double smallestPivot = 0.0;
    // The row of the matrix (in its own order) that pivot was on.
    std::size_t row = 0;
};

// The test of positive semi-definiteness of a dense symmetric matrix by its factorisation ("the positive
// semi-definiteness (by its factorisation)"): LDLᵀ with symmetric pivoting on the largest remaining diagonal, the
// Cholesky factorisation with complete pivoting of a semi-definite matrix (Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., §10.3): at step k the largest diagonal d of the Schur complement S_k is the pivot; a matrix is
// semi-definite if and only if every S_k has d ≥ 0, and S_k = 0 once d = 0 (for a semi-definite matrix
// |s_ij| ≤ max(s_ii, s_jj)). In floating point the factorisation stops when d ≤ tolerance·s, s the largest |a_ij|; the
// matrix is then taken as semi-definite when the remainder is negligible: no diagonal below −tolerance·s and no term
// above tolerance·s (a remainder [[0, b], [b, 0]] has the eigenvalue −|b|).
//
// Only the lower triangle of `symmetric` is read. Cost: n³/6 multiply-adds on a copy of n² doubles (budgeted).
// Thread-safety: a pure function. Throws std::invalid_argument for a matrix that is not square, MemoryBudgetError.
[[nodiscard]] SemiDefiniteResult testSemiDefinite(linalg::ConstDenseMatrixView symmetric, double tolerance,
                                                  base::MemoryBudget& budget);

} // namespace oos::reduction
